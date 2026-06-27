import os
import numpy as np
from dataset import AudioMelConversions, load_wav
from hifigan import HIFIGANConfig, HIFIGAN
import pandas as pd
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file
import scipy.io.wavfile as wf
import soundfile as sf
from tqdm import tqdm

from model import Tacotron2Config, Tacotron2
from tokenizer import Tokenizer

#Tacotron2
config = Tacotron2Config()
tacotron2 = Tacotron2(config)
tokenizer = Tokenizer()

state_dict = torch.load('tacotron2/final_checkpoint/pytorch_model.bin', map_location=torch.device('cuda')) 
tacotron2.load_state_dict(state_dict)
tacotron2.eval().to('cuda')


#HifiGAN
config = HIFIGANConfig()
hifigan = HIFIGAN(config)

file_path = "../HifiGAN/work_dir/hifigan/final_checkpoint/model.safetensors"
state_dict = load_file(file_path, device="cuda")

hifigan.load_state_dict(state_dict)
hifigan.eval().to('cuda')

def inference(text, output_path):
    print(f"Input Text: {text}")

    tokens = tokenizer.encode(text).unsqueeze(0).to('cuda')
    output, alignments  = tacotron2.inference(tokens)
    output = output.transpose(1,2)

    ### HIFIGAN ###
    print("HIFIGAN")
    with torch.no_grad():
        gen_audio = hifigan.generator(output).squeeze().cpu().numpy()
    sf.write(output_path, gen_audio, 22050)

def inference_tacotron2_hifigan(input_csv: str, output_folder: str, num: int):
    df = pd.read_csv(
        input_csv,
        sep="|",
        header=None,
        names=["id", "text", "normalized_text"],
        dtype=str,
    )

    # Trường hợp metadata chỉ có 2 cột
    if df.shape[1] == 2:
        df.columns = ["id", "text"]
        df["normalized_text"] = df["text"]

    save_dir = f"{output_folder}_{num}"
    os.makedirs(save_dir, exist_ok=True)

    total = len(df)

    for idx, (_, row) in enumerate(
        tqdm(df.iterrows(), total=total, desc="Infer"), start=1
    ):
        utt_id = row["id"]

        text = row["normalized_text"]
        if pd.isna(text) or text == "":
            text = row["text"]

        output_path = os.path.join(save_dir, f"{utt_id}.wav")

        print("=" * 80)
        print(f"[{idx}/{total}]")
        print(f"ID         : {utt_id}")
        print(f"Text       : {text}")
        print(f"Output     : {output_path}")
        print("=" * 80)

        inference(text, output_path)

    print(f"\nDone! Results saved to: {save_dir}")
    
    
if __name__ == "__main__":
    inference_tacotron2_hifigan('../benchmarking/test/metadata.csv', '../benchmarking/infer_tacotron2_hifigan', 3)
    # inference('xin chào địa phương', 'results/test17.wav')