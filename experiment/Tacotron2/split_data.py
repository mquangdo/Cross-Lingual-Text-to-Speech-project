import os
from datasets import load_dataset, Audio
from pathlib import Path
import csv
import soundfile as sf
import numpy as np
import io
import pandas as pd

dataset = load_dataset(
    "strongpear/viet_muong_merged_0_200_denoise_silence_speaker101"
)

# Cast TRƯỚC khi split để reset audio decoder về soundfile
dataset["train"] = dataset["train"].cast_column(
    "audio",
    Audio(
        sampling_rate=None,
        decode=False,
    ),
)

dataset = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42,
)

train = dataset["train"]
test = dataset["test"]

print("Num train:", train.num_rows)
print("Num test:", test.num_rows)


def decode_audio(sample_audio: dict) -> tuple[np.ndarray, int]:
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
            f"Không tìm thấy audio: {sample_audio.keys()}"
        )

    if array.ndim > 1:
        array = array.mean(axis=1)

    return array.astype(np.float32), sr


def split_data(
    data_dir: str = "data",
    split_dir: str = "data_split",
):
    data_dir = Path(data_dir)
    wav_dir = data_dir / "wavs"

    wav_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(split_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows = []

    train_rows = []
    test_rows = []

    global_idx = 0

    #
    # Export toàn bộ dataset ra data/
    #
    for split_name, split_ds in {
        "train": train,
        "test": test,
    }.items():

        for sample in split_ds:

            file_id = f"{global_idx:05d}"

            text = sample["text"].strip()

            array, sr = decode_audio(
                sample["audio"]
            )

            wav_path = wav_dir / f"{file_id}.wav"

            sf.write(
                str(wav_path),
                array,
                sr,
            )

            #
            # Đường dẫn tuyệt đối tới file wav
            #
            audio_path = str(
                wav_path.resolve()
            )

            #
            # metadata.csv (LJSpeech format)
            #
            metadata_rows.append(
                [
                    audio_path,
                    text,
                    text,
                ]
            )

            #
            # train/test metadata
            #
            train_test_row = {
                "file_path": audio_path,
                "normalized_transcript": text,
            }

            if split_name == "train":
                train_rows.append(
                    train_test_row
                )
            else:
                test_rows.append(
                    train_test_row
                )

            global_idx += 1

            if global_idx % 100 == 0:
                print(
                    f"{global_idx} samples exported..."
                )

    #
    # metadata.csv
    #
    metadata_path = data_dir / "metadata.csv"

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

        writer.writerow(
            [
                "file_path",
                "raw_transcription",
                "normalized_transcript",
            ]
        )

        writer.writerows(
            metadata_rows
        )

    #
    # train_metadata.csv
    #
    train_df = pd.DataFrame(
        train_rows
    )

    test_df = pd.DataFrame(
        test_rows
    )

    train_df.to_csv(
        Path(split_dir)
        / "train_metadata.csv",
        index=False,
    )

    test_df.to_csv(
        Path(split_dir)
        / "test_metadata.csv",
        index=False,
    )

    print()
    print(
        f"Total samples: {global_idx}"
    )
    print(
        f"Train samples: {len(train_rows)}"
    )
    print(
        f"Test samples : {len(test_rows)}"
    )
    print(
        f"LJSpeech data saved to: {data_dir}"
    )
    print(
        f"Split metadata saved to: {split_dir}"
    )


if __name__ == "__main__":
    split_data()