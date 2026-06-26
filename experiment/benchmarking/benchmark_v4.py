from ssr_eval_utils import *
from ssr_eval import AudioMetrics
import os
import time
import numpy as np
from pymcd.mcd import Calculate_MCD


def mcd_eval(ref_path, pred_path):
    return mcd_toolbox.calculate_mcd(ref_path, pred_path)


def lsd_ssim_eval(ref_path, pred_path):
    result = au.evaluation(ref_path, pred_path)
    return result['lsd'], result['ssim']


def compute_ci(values):
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # sample std
    n = len(values)
    ci = 1.96 * std / np.sqrt(n)
    return mean, ci


if __name__ == '__main__':
    au = AudioMetrics(rate=22050)
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")

    ref_folder = "test/wavs"
    pred_folder = "infer_tacotron2_hifigan_0"

    print('Working on folder:', pred_folder)

    # ===== lọc file wav =====
    ref_list = sorted([
        f for f in os.listdir(ref_folder)
        if f.lower().endswith('.wav')
    ])

    pred_list = sorted([
        f for f in os.listdir(pred_folder)
        if f.lower().endswith('.wav')
    ])

    # match file chung
    common_files = sorted(set(ref_list) & set(pred_list))
    print(f"Total matched files: {len(common_files)}")

    # ===== lưu giá trị từng sample =====
    mcd_list = []
    lsd_list = []
    ssim_list = []

    for i, f in enumerate(common_files):
        print('Processing sample:', i, f)

        ref_path = os.path.join(ref_folder, f)
        pred_path = os.path.join(pred_folder, f)

        mcd_res = mcd_eval(ref_path, pred_path)
        lsd_res, ssim_res = lsd_ssim_eval(ref_path, pred_path)

        mcd_list.append(mcd_res)
        lsd_list.append(lsd_res)
        ssim_list.append(ssim_res)

    #tính mean + CI
    mcd_mean, mcd_ci = compute_ci(mcd_list)
    lsd_mean, lsd_ci = compute_ci(lsd_list)
    ssim_mean, ssim_ci = compute_ci(ssim_list)

    n = len(common_files)

    print(f"\nResults:")
    print(f"MCD: {mcd_mean:.4f} ± {mcd_ci:.4f}")
    print(f"LSD: {lsd_mean:.4f} ± {lsd_ci:.4f}")
    print(f"SSIM: {ssim_mean:.4f} ± {ssim_ci:.4f}")

    #lưu file
    pred_folder_name = os.path.basename(pred_folder.rstrip("/"))
    output_file = os.path.join(pred_folder, f"{pred_folder_name}.txt")

    with open(output_file, "w") as f:
        f.write(f"Folder: {pred_folder_name}\n")
        f.write(f"Num samples: {n}\n\n")

        f.write(f"MCD: {mcd_mean:.6f} ± {mcd_ci:.6f}\n")
        f.write(f"LSD: {lsd_mean:.6f} ± {lsd_ci:.6f}\n")
        f.write(f"SSIM: {ssim_mean:.6f} ± {ssim_ci:.6f}\n")

    print(f"\nSaved results to: {output_file}")