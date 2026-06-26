import os
from datasets import load_dataset, Audio
from pathlib import Path
import csv
import soundfile as sf
import numpy as np
import io

dataset = load_dataset(
    "strongpear/viet_muong_merged_0_200_denoise_silence_speaker101"
)

# Cast audio
dataset["train"] = dataset["train"].cast_column(
    "audio",
    Audio(
        sampling_rate=None,
        decode=False,
    ),
)

# Split cố định
dataset = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42,
)

train_ds = dataset["train"]
test_ds = dataset["test"]

print("Num train:", train_ds.num_rows)
print("Num test :", test_ds.num_rows)


def decode_audio(sample_audio):
    raw_bytes = sample_audio.get("bytes")
    path = sample_audio.get("path")

    if raw_bytes:
        array, sr = sf.read(
            io.BytesIO(raw_bytes),
            dtype="float32",
        )
    elif path and os.path.exists(path):
        array, sr = sf.read(
            path,
            dtype="float32",
        )
    else:
        raise ValueError(
            f"Cannot find audio: {sample_audio.keys()}"
        )

    if array.ndim > 1:
        array = array.mean(axis=1)

    return array.astype(np.float32), sr


def export_ljspeech_split(
    split_ds,
    output_dir,
):
    output_dir = Path(output_dir)

    wav_dir = output_dir / "wavs"

    wav_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows = []

    for idx, sample in enumerate(split_ds):

        file_id = f"{idx:05d}"

        text = sample["text"].strip()

        audio, sr = decode_audio(
            sample["audio"]
        )

        wav_path = wav_dir / f"{file_id}.wav"

        sf.write(
            wav_path,
            audio,
            sr,
        )

        metadata_rows.append(
            [
                file_id,
                text,
                text,
            ]
        )

        if (idx + 1) % 100 == 0:
            print(
                f"{output_dir.name}: {idx + 1} samples exported"
            )

    metadata_path = output_dir / "metadata.csv"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
            delimiter="|",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )

        writer.writerows(
            metadata_rows
        )

    print(
        f"{output_dir.name}: saved {len(metadata_rows)} samples"
    )


def main():
    export_ljspeech_split(
        train_ds,
        "data/train",
    )

    export_ljspeech_split(
        test_ds,
        "data/test",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()