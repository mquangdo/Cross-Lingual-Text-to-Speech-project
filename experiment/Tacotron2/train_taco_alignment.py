import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import set_seed
from accelerate import Accelerator
import matplotlib.pyplot as plt

from model import Tacotron2, Tacotron2Config
from dataset import TTSDataset, TTSCollator, BatchSampler, denormalize
from tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    ### SETUP CONFIG ###
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--working_directory", type=str, required=True)
    parser.add_argument("--save_audio_gen", type=str, required=True)
    parser.add_argument("--path_to_train_manifest", type=str, required=True)
    parser.add_argument("--path_to_val_manifest", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    ### TRAINING CONFIG ###
    parser.add_argument("--training_epochs", type=int, default=500)
    parser.add_argument("--console_out_iters", type=int, default=5)
    parser.add_argument("--wandb_log_iters", type=int, default=5)
    parser.add_argument("--checkpoint_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--start_decay_epochs", type=int, default=None)

    # === Guided Attention Loss (Alignment Loss) ===
    parser.add_argument("--use_guided_attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guided_attn_weight", type=float, default=5.0)
    parser.add_argument("--guided_attn_sigma", type=float, default=0.4)
    parser.add_argument("--guided_attn_warmup_epochs", type=int, default=50)

    # Loss weights (giữ flexible)
    parser.add_argument("--refined_mel_weight", type=float, default=0.5)
    parser.add_argument("--stop_weight", type=float, default=1.0)

    ### MODEL CONFIG ###
    parser.add_argument("--character_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_kernel_size", type=int, default=5)
    parser.add_argument("--encoder_n_convolutions", type=int, default=3)
    parser.add_argument("--encoder_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_dropout_p", type=float, default=0.5)
    parser.add_argument("--decoder_rnn_embed_dim", type=int, default=1024)
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_prenet_dim", type=int, default=256)
    parser.add_argument("--decoder_prenet_depth", type=int, default=2)
    parser.add_argument("--decoder_prenet_dropout_p", type=float, default=0.5)
    parser.add_argument("--decoder_postnet_num_convs", type=int, default=5)
    parser.add_argument("--decoder_postnet_n_filters", type=int, default=512)
    parser.add_argument("--decoder_postnet_kernel_size", type=int, default=5)
    parser.add_argument("--decoder_postnet_dropout_p", type=float, default=0.5)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--attention_dropout_p", type=float, default=0.1)
    parser.add_argument("--attention_location_n_filters", type=int, default=32)
    parser.add_argument("--attention_location_kernel_size", type=int, default=31)

    ### DATASET CONFIG ###
    parser.add_argument("--sampling_rate", type=int, default=22050)
    parser.add_argument("--num_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    parser.add_argument("--min_db", type=float, default=-100.0)
    parser.add_argument("--max_scaled_abs", type=float, default=1.0)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction)

    return parser.parse_args()


def _ensure_attn_B_Tdec_Tenc(attn: torch.Tensor) -> torch.Tensor:
    """
    Normalize attention shape to [B, T_dec, T_enc].

    Common variants:
      - [B, T_dec, T_enc]  (already OK)
      - [B, T_enc, T_dec]  (transpose)
    """
    if attn is None or attn.dim() != 3:
        return None

    B, A, C = attn.shape

    # Heuristic: encoder length (chars) typically smaller than decoder length (mel frames).
    # If A < C -> likely [B, T_enc, T_dec], transpose to [B, T_dec, T_enc]
    if A < C:
        return attn.transpose(1, 2).contiguous()
    return attn


def guided_attention_loss(attn: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor, sigma: float = 0.4):
    """
    Guided attention loss encourages diagonal alignment.

    attn: [B, T_dec, T_enc]
    text_lens: [B] (encoder steps)
    mel_lens:  [B] (decoder steps)
    """
    if attn is None:
        return attn.new_tensor(0.0) if isinstance(attn, torch.Tensor) else torch.tensor(0.0)

    device = attn.device
    dtype = attn.dtype
    B, T_dec, T_enc = attn.shape

    W = torch.zeros((B, T_dec, T_enc), device=device, dtype=dtype)

    for b in range(B):
        t_len = int(mel_lens[b].item())
        s_len = int(text_lens[b].item())
        if t_len <= 0 or s_len <= 0:
            continue

        t = torch.arange(t_len, device=device, dtype=dtype) / max(t_len, 1)
        s = torch.arange(s_len, device=device, dtype=dtype) / max(s_len, 1)

        tt = t.unsqueeze(1)
        ss = s.unsqueeze(0)

        Wb = 1.0 - torch.exp(-((tt - ss) ** 2) / (2 * (sigma ** 2)))
        W[b, :t_len, :s_len] = Wb

    # valid region mask
    valid = torch.zeros((B, T_dec, T_enc), device=device, dtype=torch.bool)
    for b in range(B):
        t_len = int(mel_lens[b].item())
        s_len = int(text_lens[b].item())
        if t_len > 0 and s_len > 0:
            valid[b, :t_len, :s_len] = True

    if valid.sum() == 0:
        return attn.new_tensor(0.0)

    return (attn[valid] * W[valid]).mean()


def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
    accelerator = Accelerator(project_dir=path_to_experiment, log_with="wandb" if args.log_wandb else None)

    if args.log_wandb:rè
        accelerator.init_trackers(project_name=args.experiment_name, init_kwargs={"wandb": {"name": args.run_name}})

    accelerator.print(args)

    if accelerator.is_main_process:
        os.makedirs(args.save_audio_gen, exist_ok=True)

    tokenizer = Tokenizer()

    config = Tacotron2Config(
        num_mels=args.num_mels,
        num_chars=tokenizer.vocab_size,
        character_embed_dim=args.character_embed_dim,
        pad_token_id=tokenizer.pad_token_id,
        encoder_kernel_size=args.encoder_kernel_size,
        encoder_n_convolutions=args.encoder_n_convolutions,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_dropout_p=args.encoder_dropout_p,
        decoder_embed_dim=args.decoder_rnn_embed_dim,
        decoder_dropout_p=args.decoder_dropout_p,
        decoder_prenet_dim=args.decoder_prenet_dim,
        decoder_prenet_depth=args.decoder_prenet_depth,
        decoder_prenet_dropout_p=args.decoder_prenet_dropout_p,
        decoder_postnet_num_convs=args.decoder_postnet_num_convs,
        decoder_postnet_n_filters=args.decoder_postnet_n_filters,
        decoder_postnet_kernel_size=args.decoder_postnet_kernel_size,
        decoder_postnet_dropout_p=args.decoder_postnet_dropout_p,
        attention_dim=args.attention_dim,
        attention_dropout_p=args.attention_dropout_p,
        attention_location_n_filters=args.attention_location_n_filters,
        attention_location_kernel_size=args.attention_location_kernel_size,
    )

    model = Tacotron2(config)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Total Trainable Parameters: {total_trainable_params}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.adam_eps
    )

    trainset = TTSDataset(
        args.path_to_train_manifest,
        sample_rate=args.sampling_rate,
        n_fft=args.n_fft,
        window_size=args.window_size,
        hop_size=args.hop_size,
        fmin=args.fmin,
        fmax=args.fmax,
        num_mels=args.num_mels,
        min_db=args.min_db,
        max_scaled_abs=args.max_scaled_abs,
    )

    testset = TTSDataset(
        args.path_to_val_manifest,
        sample_rate=args.sampling_rate,
        n_fft=args.n_fft,
        window_size=args.window_size,
        hop_size=args.hop_size,
        fmin=args.fmin,
        fmax=args.fmax,
        num_mels=args.num_mels,
        min_db=args.min_db,
        max_scaled_abs=args.max_scaled_abs,
    )

    collator = TTSCollator()
    train_sampler = BatchSampler(trainset, batch_size=args.batch_size, drop_last=accelerator.num_processes > 1)

    trainloader = DataLoader(trainset, batch_sampler=train_sampler, num_workers=args.num_workers, collate_fn=collator)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collator)

    model, optimizer, trainloader, testloader = accelerator.prepare(model, optimizer, trainloader, testloader)

    using_scheduler = False
    if args.start_decay_epochs is not None:
        accelerator.print("Using LR Scheduler!!")
        using_scheduler = True

        init_lr = args.learning_rate
        min_lr = args.min_learning_rate
        decay_epochs = args.training_epochs - args.start_decay_epochs
        decay_gamma = (min_lr / init_lr) ** (1 / max(decay_epochs, 1))

        def lr_lambda(epoch):
            if epoch < args.start_decay_epochs:
                return 1.0
            return decay_gamma ** (epoch - args.start_decay_epochs)

    if args.resume_from_checkpoint is not None:
        path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
        with accelerator.main_process_first():
            accelerator.load_state(path_to_checkpoint)

        completed_epochs = int(args.resume_from_checkpoint.split("_")[-1]) + 1
        completed_steps = completed_epochs * len(trainloader)
        accelerator.print(f"Resuming from Epoch: {completed_epochs}")

        if using_scheduler:
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=completed_epochs - 1)
    else:
        completed_epochs = 0
        completed_steps = 0
        if using_scheduler:
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(completed_epochs, args.training_epochs):
        accelerator.print(f"Epoch: {epoch}")

        model.train()
        for texts, text_lens, mels, stops, encoder_mask, decoder_mask in trainloader:
            texts = texts.to(accelerator.device)
            mels = mels.to(accelerator.device)
            stops = stops.to(accelerator.device)
            encoder_mask = encoder_mask.to(accelerator.device)
            decoder_mask = decoder_mask.to(accelerator.device)

            # mel lengths from mask (your build_padding_mask: 1=pad)
            mel_lens = (decoder_mask == 0).sum(dim=1).to(text_lens.device)

            # Forward (need attention for guided loss)
            mels_out, mels_postnet_out, stop_preds, attention_weights = model(
                texts, text_lens.to("cpu"), mels, encoder_mask, decoder_mask
            )

            # Losses
            mel_loss = F.mse_loss(mels_out, mels)
            refined_mel_loss = F.mse_loss(mels_postnet_out, mels)
            stop_loss = F.binary_cross_entropy_with_logits(stop_preds.reshape(-1, 1), stops.reshape(-1, 1))

            loss = mel_loss + refined_mel_loss * args.refined_mel_weight + stop_loss * args.stop_weight # cuonglp1

            # Guided attention loss (warmup only)
            ga = None
            if args.use_guided_attn and (epoch < args.guided_attn_warmup_epochs):
                attn = _ensure_attn_B_Tdec_Tenc(attention_weights)
                if attn is not None:
                    ga = guided_attention_loss(
                        attn=attn,
                        text_lens=text_lens.to(attn.device),
                        mel_lens=mel_lens.to(attn.device),
                        sigma=args.guided_attn_sigma,
                    )
                    loss = loss + args.guided_attn_weight * ga # cuonglp1

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Metrics
            loss_item = torch.mean(accelerator.gather_for_metrics(loss)).item()
            mel_item = torch.mean(accelerator.gather_for_metrics(mel_loss)).item()
            rmel_item = torch.mean(accelerator.gather_for_metrics(refined_mel_loss)).item()
            stop_item = torch.mean(accelerator.gather_for_metrics(stop_loss)).item()

            if ga is not None:
                ga_item = torch.mean(accelerator.gather_for_metrics(ga)).item()
            else:
                ga_item = 0.0

            if completed_steps % args.console_out_iters == 0:
                accelerator.print(
                    "Completed Steps {}/{} | Loss {:.4f} | Mel {:.4f} | RMel {:.4f} | Stop {:.4f} | GA {:.4f}".format(
                        completed_steps,
                        args.training_epochs * len(trainloader),
                        loss_item,
                        mel_item,
                        rmel_item,
                        stop_item,
                        ga_item,
                    )
                )

            if completed_steps % args.wandb_log_iters == 0:
                if args.log_wandb:
                    accelerator.log(
                        {
                            "mel_loss": mel_item,
                            "refined_mel_loss": rmel_item,
                            "stop_loss": stop_item,
                            "guided_attn_loss": ga_item,
                            "total_loss": loss_item,
                        },
                        step=completed_steps,
                    )

            completed_steps += 1

        accelerator.wait_for_everyone()

        # === VALIDATION ===
        model.eval()
        accelerator.print("--VALIDATION--")

        val_mel_loss, val_rmel_loss, val_stop_loss, val_ga_loss = 0, 0, 0, 0
        num_losses = 0
        save_first = True

        for texts, text_lens, mels, stops, encoder_mask, decoder_mask in testloader:
            texts = texts.to(accelerator.device)
            mels = mels.to(accelerator.device)
            stops = stops.to(accelerator.device)
            encoder_mask = encoder_mask.to(accelerator.device)
            decoder_mask = decoder_mask.to(accelerator.device)

            mel_lens = (decoder_mask == 0).sum(dim=1).to(text_lens.device)

            with torch.no_grad():
                mels_out, mels_postnet_out, stop_preds, attention_weights = model(
                    texts, text_lens.to("cpu"), mels, encoder_mask, decoder_mask
                )

            mel_loss = F.mse_loss(mels_out, mels)
            refined_mel_loss = F.mse_loss(mels_postnet_out, mels)
            stop_loss = F.binary_cross_entropy_with_logits(stop_preds.reshape(-1, 1), stops.reshape(-1, 1))

            # guided attn metric (log only)
            ga = None
            if args.use_guided_attn:
                attn = _ensure_attn_B_Tdec_Tenc(attention_weights)
                if attn is not None:
                    ga = guided_attention_loss(
                        attn=attn,
                        text_lens=text_lens.to(attn.device),
                        mel_lens=mel_lens.to(attn.device),
                        sigma=args.guided_attn_sigma,
                    )

            val_mel_loss += mel_loss
            val_rmel_loss += refined_mel_loss
            val_stop_loss += stop_loss
            if ga is not None:
                val_ga_loss += ga
            num_losses += 1

            if accelerator.is_main_process and save_first:
                true_mel = denormalize(mels[0].T.to("cpu"))
                pred_mel = denormalize(mels_postnet_out[0].T.to("cpu"))
                attention = attention_weights[0].T.to("cpu") if attention_weights is not None else None

                fig, axes = plt.subplots(3, 1, figsize=(8, 12))

                im0 = axes[0].imshow(true_mel, aspect="auto", origin="lower", interpolation="none")
                axes[0].set_title("True Mel")
                axes[0].set_ylabel("Mel bins")
                fig.colorbar(im0, ax=axes[0])

                im1 = axes[1].imshow(pred_mel, aspect="auto", origin="lower", interpolation="none")
                axes[1].set_title("Predicted Mel")
                axes[1].set_ylabel("Mel bins")
                fig.colorbar(im1, ax=axes[1])

                if attention is not None:
                    im2 = axes[2].imshow(attention, aspect="auto", origin="lower", interpolation="none")
                    axes[2].set_title("Alignment")
                    axes[2].set_ylabel("Character Index")
                    axes[2].set_xlabel("Decoder Mel Timesteps")
                    fig.colorbar(im2, ax=axes[2])
                else:
                    axes[2].set_title("Alignment (None)")

                plt.tight_layout()
                plt.savefig(os.path.join(args.save_audio_gen, f"epoch_{epoch}_result.png"))
                plt.close()

            save_first = False

        val_mel = torch.mean(accelerator.gather_for_metrics(val_mel_loss)).item() / num_losses
        val_rmel = torch.mean(accelerator.gather_for_metrics(val_rmel_loss)).item() / num_losses
        val_stop = torch.mean(accelerator.gather_for_metrics(val_stop_loss)).item() / num_losses

        if args.use_guided_attn:
            val_ga = torch.mean(accelerator.gather_for_metrics(val_ga_loss)).item() / num_losses
        else:
            val_ga = 0.0

        val_total = val_mel + val_rmel + val_stop  # không cộng GA để so sánh với run cũ (cuonglp1)
        val_mel_stop =  val_mel + 0.3 * val_rmel + 3.0 * val_stop # sinh thêm score cho mel & stop only (cuonglp1)

        accelerator.print(
            "Loss {:.4f} | Mel+Stop {:.4f} | Mel {:.4f} | RMel {:.4f} | Stop {:.4f} | ValGA {:.4f}".format(
                val_total, val_mel_stop, val_mel, val_rmel, val_stop, val_ga
            )
        )

        if args.log_wandb:
            accelerator.log(
                {
                    "val_mel_loss": val_mel,
                    "val_refined_mel_loss": val_rmel,
                    "val_stop_loss": val_stop,
                    "val_guided_attn_loss": val_ga,
                    "val_mel_stop": val_mel_stop,
                    "val_total_loss": val_total,
                },
                step=completed_steps,
            )

        if completed_epochs % args.checkpoint_epochs == 0:
            accelerator.print("Saving Checkpoint!")
            path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_epochs}")
            accelerator.save_state(output_dir=path_to_checkpoint, safe_serialization=False)

        completed_epochs += 1

        if using_scheduler:
            scheduler.step(epoch=completed_epochs)
            accelerator.print(f"Learning Rate: {scheduler.get_last_lr()[0]}")

    accelerator.save_state(os.path.join(path_to_experiment, "final_checkpoint"), safe_serialization=False)
    accelerator.end_training()


if __name__ == "__main__":
    main()