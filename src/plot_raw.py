"""
Day 1：读一条 eegmmidb 数据，画原始波形图 + 功率谱(PSD)

⚠️ 口径提醒：eegmmidb 仅含「左/右手」与「双手/双脚」MI，**无左右腿**。
   这里用 run 3（左/右手 MI，真实执行）演示完整「读取 -> 标准化 -> 可视化」流程，
   建立一套可复用模板，后续直接套到 Zuo2025 的左右腿数据上。

用法：
    python src/plot_raw.py
产出：
    figures/day1_raw.png  （前 8 通道、前 5 秒原始波形）
    figures/day1_psd.png  （1-45Hz Welch 功率谱）
"""

import os

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能存图
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "eegmmidb")
FIG_DIR = os.path.join(ROOT, "figures")

SUBJECT = 1
RUN = 3  # 左/右手 MI（真实执行）


def load_one_run(subject, run, data_dir):
    files = eegbci.load_data(subject, run, path=data_dir, update_path=True)
    raw = mne.io.read_raw_edf(files[0], preload=True)
    eegbci.standardize(raw)  # 统一通道命名
    # MNE 1.14 起 standard_1005 更名为 colin27_1005，做一次兼容
    try:
        montage = mne.channels.make_standard_montage("colin27_1005")
    except (ValueError, KeyError):
        montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)
    return raw


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    raw = load_one_run(SUBJECT, RUN, DATA_DIR)
    print(raw.info)

    # 1) 原始波形（取前 5 秒、前 8 通道）
    fig = raw.plot(duration=5, n_channels=8, show=False, block=False)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(FIG_DIR, "day1_raw.png"), dpi=120)
    plt.close(fig)

    # 2) 功率谱密度（Welch, 1-45Hz）
    raw.filter(1, 45, verbose="warning")
    spectrum = raw.compute_psd(method="welch", fmin=1, fmax=45, n_fft=2048)
    psd_fig = spectrum.plot()
    if isinstance(psd_fig, (list, tuple)):
        psd_fig = psd_fig[0]
    psd_fig.savefig(os.path.join(FIG_DIR, "day1_psd.png"), dpi=120)
    plt.close(psd_fig)

    print("已保存：figures/day1_raw.png, figures/day1_psd.png")


if __name__ == "__main__":
    main()
