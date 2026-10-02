"""
数据下载脚本（Day 1 / Day 2）

主数据集：PhysioNet EEG Motor Movement/Imagery (eegmmidb)
  ⚠️ 关键事实：eegmmidb 只有「左拳/右拳」和「双手/双脚」，**没有"左右腿"标签**。
     它的 T1/T2 含义确实随 run 变化（详见 README 第 9 节对照表）。
     因此本数据集用于「方法练手 + 与 Zuo2025 对照」，真正的左右腿 MI 任务用 Zuo2025。

eegmmidb 由 mne 原生支持：mne.datasets.eegbci
"""

import os

import mne
from mne.datasets import eegbci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "eegmmidb")


def download_eegmmidb(subjects=(1,), runs=(3, 4, 7, 8, 11, 12)):
    """下载指定被试、指定 run 的 EDF 文件。

    runs 默认选「左/右手运动想象」相关 run：
        3,7,11  = 真实执行（T1=左拳, T2=右拳）
        4,8,12  = 想象执行（T1=左拳, T2=右拳）
    返回本地文件路径列表。
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    files = eegbci.load_data(list(subjects), list(runs), path=DATA_DIR, update_path=True)
    print(f"已下载 {len(files)} 个文件：")
    for f in files:
        print(" -", f)
    return files


def download_zuo2025():
    """Zuo2025：真正的 left_leg vs right_leg 运动想象数据集
    （Scientific Data 2025, CC BY 4.0，DOI: 10.1038/s41597-025-05767-2）。

    这是本项目的「主数据集」，用于复现 left_leg vs right_leg 的运动想象二分类任务。
    接入方式：通过 MOABB 的 Zuo2025 适配器。使用前需先 pip install moabb，
    并按数据集说明在 Figshare 完成下载授权。

    返回 moabb 数据集对象；未安装 moabb 时返回 None 并给出指引。
    """
    try:
        from moabb.datasets import Zuo2025
    except ImportError:
        print("未安装 moabb。请先：pip install moabb")
        print("参考 https://github.com/NeuroTechX/moabb 完成 Zuo2025 的下载授权。")
        return None
    ds = Zuo2025()
    print("Zuo2025 已装配完成，调用 ds.get_data(subjects=[1]) 会按需触发下载。")
    return ds


if __name__ == "__main__":
    # Day 1 先跑这个，把 eegmmidb 练手数据拉到本地
    download_eegmmidb()
