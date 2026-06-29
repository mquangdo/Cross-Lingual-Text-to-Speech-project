import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import torch
import matplotlib.pyplot as plt
# from IPython.display import Audio
import numpy as np
import json
from scipy.io.wavfile import write
from scipy.ndimage import gaussian_filter1d
import soundfile as sf
import torchaudio

from model import Tacotron2Config, Tacotron2
from tokenizer import Tokenizer
from dataset import AudioMelConversions, load_wav

from infer_models import Generator
from env import AttrDict

MAX_WAV_VALUE = 32768.0
mel_gt_stats = np.load("/home/app/translate-topic/cuonglp1/TRAIN_phase_02/hifi_gan/data/mel_gt_stats.npz")
MEL_GT_MEAN = mel_gt_stats["mean"]
MEL_GT_STD  = mel_gt_stats["std"]
sr = 22050


#Tacotron2
config = Tacotron2Config()
model = Tacotron2(config)
tokenizer = Tokenizer()
a2m = AudioMelConversions()

state_dict = torch.load('/home/app/translate-topic/cuonglp1/TRAIN_phase_01/tacotron2/best_checkpoint/ver_4_0_0_1200/pytorch_model.bin', map_location=torch.device('cuda'))
model.load_state_dict(state_dict)
model.eval()

tacotron2_total_params = sum(p.numel() for p in model.parameters())
print("Total Tacotron2 parameters:", tacotron2_total_params)

#Hifi-GAN
with open("/home/app/translate-topic/cuonglp1/TRAIN_phase_02/hifi_gan/config_v3.json") as f:
    vocoder_config = f.read()
    
h = AttrDict(json.loads(vocoder_config))

generator = Generator(h).to(torch.device('cuda'))
state_dict_g = torch.load(
    "/home/app/translate-topic/cuonglp1/TRAIN_phase_02/hifi_gan/best_checkpoint/g_00355000",
    # "TRAIN_phase_02/hifi_gan/ckpts/g_epoch_2000",
    map_location=torch.device('cuda')
)
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()

generator_total_params = sum(p.numel() for p in generator.parameters())
print("Total Hifi-GAN parameters:", generator_total_params)


def refine_mel_for_hifigan(
    mel_taco: np.ndarray,          # [80, T]
    mel_gt_mean: np.ndarray,       # [80]
    mel_gt_std: np.ndarray,        # [80]
    *,
    freq_smooth_sigma: float = 0.4,
    clamp_min: float = -11.5,
    clamp_max: float = 0.6,
):
    """
    Refine Tacotron2 mel → HiFi-GAN-compatible mel
    using precomputed GT mel statistics.

    This version is SAFE for HiFi-GAN:
    - per-bin mean/std matching
    - NO dynamic-range expansion (avoids hiss / metallic noise)
    - light frequency smoothing
    - soft clamp to GT acoustic bounds

    Args:
        mel_taco: mel inferred from Tacotron2 [80, T]
        mel_gt_mean: per-bin mean from GT mel [80]
        mel_gt_std: per-bin std from GT mel [80]
        freq_smooth_sigma: Gaussian sigma for frequency smoothing (0.3–0.5)
        clamp_min: lower bound of GT mel (≈ -11.5)
        clamp_max: upper bound of GT mel (≈ 0.5–0.8)
    """

    assert mel_taco.ndim == 2 and mel_taco.shape[0] == 80, \
        f"Expected mel shape [80, T], got {mel_taco.shape}"

    # ---------- Step 1: per-bin mean / std matching ----------
    src_mean = mel_taco.mean(axis=1, keepdims=True)
    src_std  = mel_taco.std(axis=1, keepdims=True)

    ref_mean = mel_gt_mean[:, None]
    ref_std  = mel_gt_std[:, None]

    mel = (mel_taco - src_mean) / (src_std + 1e-6)
    mel = mel * ref_std + ref_mean

    # ---------- Step 2: light frequency smoothing ----------
    if freq_smooth_sigma is not None and freq_smooth_sigma > 0:
        mel = gaussian_filter1d(mel, sigma=freq_smooth_sigma, axis=0)

    # ---------- Step 3: enforce GT acoustic bounds ----------
    mel = np.clip(mel, clamp_min, clamp_max)

    return mel.astype(np.float32)

def infer(text, output_path):
    tokens = tokenizer.encode(text).unsqueeze(0)

    with torch.no_grad():
        output, alignments = model.inference(tokens)
        mel_taco = output[0].T
        mel_taco_array = mel_taco.detach().cpu().numpy()

        # refine mel infer
        mel_refined = refine_mel_for_hifigan(
            mel_taco_array,
            MEL_GT_MEAN,
            MEL_GT_STD,
            freq_smooth_sigma=0.4,
            clamp_min=-11.5,
            clamp_max=0.6,
        )

        mel_refined = torch.from_numpy(mel_refined).unsqueeze(0).to("cuda")  # [1,80,T]

        y_g_hat = generator(mel_refined)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.detach().cpu()
        audio = audio.numpy().astype('int16')

        sf.write(output_path, audio, 22050)        

    return audio
    
def inference_griffinlim(text: str, output_path: str):
    print(f"Input Text: {text}")

    # true_mel = a2m.audio2mel(load_wav(audio_path), do_norm=False)
    # print(true_mel.shape)
    # true_audio = a2m.mel2audio(true_mel.squeeze(0), do_denorm=False)
    # display(Audio(true_audio, rate=22050))
    
    # device = next(model.parameters()).device
    # dtype  = next(generator.parameters()).dtype
    
    tokens = tokenizer.encode(text).unsqueeze(0)

    with torch.no_grad():
        output, alignments = model.inference(tokens)
        mel_taco = output[0].T
        # mel_taco_array = mel_taco.detach().cpu().numpy()

    plt.tight_layout()
    plt.show()

    # without hifigan
    print('Without HifiGAN')
    mel_audio = a2m.mel2audio(mel_taco, do_denorm=True)
    sf.write(output_path, mel_audio, 22050)        

def infer_tacotron2_hifigan(count: int):
    df = pd.read_csv('data_309/metadata_test.csv')
    save_dir = f'infer_tacotron-hifigan/data{count}'
    os.makedirs(save_dir, exist_ok=True)

    for (i, row) in df.iterrows():
        print('Infering sample: ', i)
        audio_path = row[0]
        tts_text = row[1]

        output_audio_path = os.path.join(save_dir, audio_path)
        infer(tts_text, output_audio_path)

    # print(df.shape[0])
    # print(len(os.listdir('data_309/wavs')))

if __name__ == "__main__":
    infer('xin chào địa phương anh ba đang ở địa phương', 'results/test28.wav')


