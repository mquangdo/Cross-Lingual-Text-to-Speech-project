import os
import platform
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

from cli.SparkTTS import SparkTTS


#########################################
# Load model (chỉ load 1 lần)
#########################################

MODEL_DIR = "finetuned_models/Spark-TTS-0.5B"

if platform.system() == "Darwin" and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

sparktts = SparkTTS(MODEL_DIR, device)


#########################################
# Infer một câu
#########################################

def inference(
    text,
    output_path,
    prompt_speech_path=None,
    prompt_text=None,
    gender=None,
    pitch=None,
    speed=None,
):
    print(f"Input Text: {text}")

    with torch.no_grad():
        wav = sparktts.inference(
            text=text,
            prompt_speech_path=prompt_speech_path,
            prompt_text=prompt_text,
            gender=gender,
            pitch=pitch,
            speed=speed,
        )

    sf.write(output_path, wav, 16000)


#########################################
# Infer cả dataset
#########################################

def inference_sparktts(
    input_csv,
    output_folder,
    num,
    prompt_speech_path=None,
    prompt_text=None,
    gender=None,
    pitch=None,
    speed=None,
):
    """
    input_csv: metadata.csv (LJSpeech)
    """

    df = pd.read_csv(
        input_csv,
        sep="|",
        header=None,
        names=["id", "text", "normalized_text"],
        dtype=str,
    )

    if df.shape[1] == 2:
        df.columns = ["id", "text"]
        df["normalized_text"] = df["text"]

    save_dir = f"{output_folder}_{num}"
    os.makedirs(save_dir, exist_ok=True)

    total = len(df)

    for idx, (_, row) in enumerate(
        tqdm(df.iterrows(), total=total, desc="SparkTTS"),
        start=1,
    ):
        utt_id = os.path.splitext(row["id"])[0]

        text = row["normalized_text"]
        if pd.isna(text) or text == "":
            text = row["text"]

        output_path = os.path.join(save_dir, f"{utt_id}.wav")

        print("=" * 80)
        print(f"[{idx}/{total}]")
        print(f"ID     : {utt_id}")
        print(f"Text   : {text}")
        print(f"Output : {output_path}")
        print("=" * 80)

        inference(
            text=text,
            output_path=output_path,
            prompt_speech_path=prompt_speech_path,
            prompt_text=prompt_text,
            gender=gender,
            pitch=pitch,
            speed=speed,
        )

    print(f"\nDone! Results saved to {save_dir}")

if __name__ == "__main__":
    inference_sparktts('../benchmarking/test/metadata.csv', '../benchmarking/infer_sparktts', prompt_speech_path='speech_prompts_muong/0.wav', num=0)