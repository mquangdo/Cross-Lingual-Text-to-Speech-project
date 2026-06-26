from datasets import load_dataset
import os
import soundfile as sf
import shutil
import numpy as np

def convert_to_ljspeech(hf_name: str,
                        output_dir: str,
                        wav_folder_name: str = "wavs",
                        metadata_filename: str = "metadata.csv",
                        id_field: str = "id",
                        audio_field: str = "audio",
                        text_field: str = "text",
                        normalized_text_field: str = None):
    """
    Load a HuggingFace dataset and export to LJSpeech format.

    Args:
        hf_name: dataset identifier on HF hub (e.g. "strongpear/viet_muong_vtv5_001")
        output_dir: where to write the converted dataset
        wav_folder_name: subfolder under output_dir where WAVs are saved
        metadata_filename: name of metadata CSV file
        id_field: field name for unique id in the HF dataset
        audio_field: field name for the audio data or path
        text_field: field name for transcription text
        normalized_text_field: optional field name for normalized text (if present)
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_dir = os.path.join(output_dir, wav_folder_name)
    os.makedirs(wav_dir, exist_ok=True)

    ds = load_dataset(hf_name, split="train")  # adjust split name if different
    meta_lines = []

    for idx, example in enumerate(ds):
        uid = example[id_field]
        # get text
        txt = example.get(text_field, "").strip()
        if normalized_text_field:
            norm = example.get(normalized_text_field, "").strip()
        else:
            norm = txt  # fallback: use same as text

        # Prepare WAV file
        audio = example[audio_field]
        # audio_dec is AudioDecoder
        # get waveform array + sampling rate
        samples = audio.get_all_samples()
        # samples.data is e.g. a torch.Tensor with shape (channels, num_samples)
        data = samples.data  # likely a torch.Tensor or numpy-compatible
        sr = samples.sample_rate

        # If stereo or multi-channel, convert to mono by averaging
        if hasattr(data, "numpy"):
            arr = data.numpy()
        else:
            arr = np.array(data)

        # If multiple channels, average them to mono
        if arr.ndim > 1:
            arr_mono = np.mean(arr, axis=0)
        else:
            arr_mono = arr

        wav_path = os.path.join(wav_dir, f"{idx}.wav")
        # Write as PCM 16-bit wav (float32 → pcm automatically handled by soundfile)
        sf.write(wav_path, arr_mono, sr, subtype="PCM_16")
        # `audio` may be e.g. a dict with path / array depending on dataset. We'll try to handle both
        # if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        #     wav = audio["array"]
        #     sr = audio["sampling_rate"]
        #     wav_path = os.path.join(wav_dir, f"{uid}.wav")
        #     sf.write(wav_path, wav, sr, subtype="PCM_16")
        # elif isinstance(audio, dict) and "path" in audio:
        #     src = audio["path"]
        #     wav_path = os.path.join(wav_dir, f"{uid}.wav")
        #     shutil.copy(src, wav_path)
        # elif isinstance(audio, (bytes, bytearray)):
        #     wav_path = os.path.join(wav_dir, f"{uid}.wav")
        #     with open(wav_path, "wb") as f:
        #         f.write(audio)
        # else:
        #     raise ValueError(f"Cannot interpret audio field for example {uid}")

        # Build metadata line: id|text|normalized_text (or id|text if you prefer)
        meta_line = f"{wav_path}|{txt}|{norm}"
        meta_lines.append(meta_line)

    # Write metadata.csv
    meta_path = os.path.join(output_dir, metadata_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        for l in meta_lines:
            f.write(l + "\n")

    print("Done. Wrote {} wavs and metadata to {}".format(len(meta_lines), output_dir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_path", type=str, required=True)

    args = parser.parse_args()
    
    convert_to_ljspeech(
        hf_name=args.hf_dataset_path,
        output_dir="data",
        wav_folder_name="wavs",
        metadata_filename="metadata.csv",
        id_field="id",
        audio_field="audio",          # might be e.g. "speech" or similar
        text_field="text",
        normalized_text_field=None    # or name of normalized text field if exists
    )