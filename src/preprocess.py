"""
Day 2: 预处理 + 按事件提取运动想象(左右拳)片段
============================================
目标: 跑通 带通滤波 -> ICA 去眼电 -> 切 epoch 的最小闭环。

关键说明:
- 本数据集 PhysioNet eegmmidb **只有左右拳 / 双手双脚**, **没有左右腿**。
  * runs 3,7,11 = 执行(左拳/右拳)
  * runs 4,8,12 = 想象(左拳/右拳)
  * 每 run 内 T1=左拳, T2=右拳
- 「左右腿」请用 Zuo2025 数据集(download_data.py 已封装), 那是真正带 left_leg/right_leg 标签的。
- 本脚本用左右拳做方法练手, 不要对外说成"左右腿"。

用法:
  .venv/Scripts/python.exe src/preprocess.py                 # 默认 subj=1, runs=[3,4,7,8,11,12]
  .venv/Scripts/python.exe src/preprocess.py --runs 3 4      # 快速冒烟测试
  .venv/Scripts/python.exe src/preprocess.py --subjects 1 2 3 # 多被试
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # 无头环境, 不弹窗
import numpy as np
import mne
from mne.datasets import eegbci
from mne.preprocessing import ICA

DATA_DIR = "data"
OUT_DIR = "processed"
# 左右拳 runs: 执行(3,7,11) + 想象(4,8,12)。每 run 内 T1=左拳, T2=右拳
DEFAULT_RUNS = [3, 4, 7, 8, 11, 12]
# eegbci annotations: T1=左拳, T2=右拳 (run3/7/11 执行, run4/8/12 想象)
TMIN, TMAX = -0.5, 4.0          # epoch 窗口(相对事件起始)
LOPASS, HIPASS = 40.0, 1.0       # 带通(ICA 建议高通>=1Hz)


def build_montage(raw):
    # MNE 1.14 起 standard_1005 改名为 colin27_1005, 做兼容
    try:
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    except Exception:
        raw.set_montage(mne.channels.make_standard_montage("colin27_1005"))


def load_runs(subject, runs):
    files = eegbci.load_data(subject, runs, path=DATA_DIR, update_path=True)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in files]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)           # 统一通道名为标准 10-20
    build_montage(raw)
    raw.rename_channels(lambda x: x.strip("."))
    return raw


def bandpass(raw):
    # 复制一份再滤波, 保留原始 raw
    filt = raw.copy().filter(HIPASS, LOPASS, fir_design="firwin",
                             skip_by_annotation="edge")
    return filt


def remove_eog(filt_raw):
    # ICA 去眼电: 找与 Fp1/Fp2(眼电最强) 相关的成分剔除
    ica = ICA(n_components=15, random_state=42, max_iter="auto")
    ica.fit(filt_raw)
    eog_inds, _ = ica.find_bads_eog(filt_raw, ch_name=["Fp1", "Fp2"], threshold=3.0)
    print(f"  [ICA] 识别为眼电的成分为: {eog_inds}")
    ica.exclude = eog_inds
    clean = ica.apply(filt_raw.copy())
    return clean


def epoch(clean_raw):
    # 本数据集事件存于 annotations (无 STI 014 刺激通道), 用 events_from_annotations 提取
    # eegbci: run3/7/11(执行) 与 run4/8/12(想象) 内 T1=左拳, T2=右拳
    events, ev_id = mne.events_from_annotations(clean_raw)
    if "T1" not in ev_id or "T2" not in ev_id:
        raise RuntimeError(f"未找到 T1/T2 事件, 当前 ev_id={ev_id}, 请检查 runs 参数")
    events = events[np.isin(events[:, 2], [ev_id["T1"], ev_id["T2"]])]
    event_id = {"left_fist": ev_id["T1"], "right_fist": ev_id["T2"]}
    epochs = mne.Epochs(clean_raw, events, event_id,
                         tmin=TMIN, tmax=TMAX,
                         baseline=(None, 0), preload=True)
    return epochs, event_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", type=int, default=[1])
    ap.add_argument("--runs", nargs="+", type=int, default=DEFAULT_RUNS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    for subj in args.subjects:
        print(f"\n=== 被试 {subj} | runs={args.runs} ===")
        raw = load_runs(subj, args.runs)
        print(f"  [load] {raw.info['nchan']} 通道, {raw.times[-1]:.1f}s, sfreq={raw.info['sfreq']:.0f}Hz")

        filt = bandpass(raw)
        print("  [filter] 带通 %.1f-%.1f Hz 完成" % (HIPASS, LOPASS))

        clean = remove_eog(filt)

        epochs, event_id = epoch(clean)
        print(f"  [epoch] 共 {len(epochs)} 段, 每段 {epochs.times[-1]-epochs.times[0]:.1f}s")
        print(f"  [epoch] 各类别数量: { {k: int((epochs.events[:,2]==v).sum()) for k,v in event_id.items()} }")

        out = os.path.join(OUT_DIR, f"sub{subj:03d}_epochs-epo.fif")
        epochs.save(out)
        print(f"  [save] -> {out}")

        # 输出 epoch 平均波形图(Day 2 图产出)
        try:
            os.makedirs("figures", exist_ok=True)
            fig = epochs.average().plot(show=False)
            fig_path = os.path.join("figures", f"sub{subj:03d}_epochs_avg.png")
            fig.savefig(fig_path)
            print(f"  [fig ] -> {fig_path}")
        except Exception as e:
            print(f"  [warn] 平均波形图生成跳过: {e}")


if __name__ == "__main__":
    main()
