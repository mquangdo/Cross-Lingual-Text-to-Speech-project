import numpy as np
from dataset import AudioMelConversions, load_wav
from model import HIFIGANConfig, HIFIGAN
import pandas as pd
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file
import scipy.io.wavfile as wf

from tacotron2 import Tacotron2Config, Tacotron2
from tokenizer import Tokenizer

#Tacotron2
config = Tacotron2Config()
tacotron2 = Tacotron2(config)
tokenizer = Tokenizer()

state_dict = torch.load('../Tacotron2/tacotron2/final_checkpoint/pytorch_model.bin', map_location=torch.device('cpu')) 
tacotron2.load_state_dict(state_dict)
tacotron2.eval()


#HifiGAN
config = HIFIGANConfig()
hifigan = HIFIGAN(config)

file_path = "hifigan/model.safetensors"
state_dict = load_file(file_path, device="cpu")

hifigan.load_state_dict(state_dict)
hifigan.eval()

def inference(text, output_path):
    print(f"Input Text: {text}")

    tokens = tokenizer.encode(text).unsqueeze(0)
    output, alignments  = tacotron2.inference(tokens)
    output = output.transpose(1,2)

    ### HIFIGAN ###
    print("HIFIGAN")
    with torch.no_grad():
        gen_audio = hifigan.generator(output).squeeze().numpy()
    sf.write(output_path, gen_audio, 22050)
    
    
if __name__ == "__main__":
    inference("Deep Learning is the future of science", "test.wav")