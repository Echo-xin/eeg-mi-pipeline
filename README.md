# 运动想象 EEG 信号处理复现

> 目标：用公开数据集跑通「加载数据 → 预处理 → 提取 MI 片段 → 特征提取 → 分类 → LSL 接收示例」的最小闭环，
> 覆盖运动想象（Motor Imagery, MI）脑电信号处理的基础流程。

---

## 目录结构

```
eeg-reproduce/
├── .venv/              # Python 虚拟环境（D 盘，隔离安装）
├── data/
│   ├── eegmmidb/       # PhysioNet 练手数据（方法对照，无左右腿）
│   └── zuo2025/        # 左右腿 MI 主数据集（待接入）
├── src/
│   ├── download_data.py    # 数据下载（eegmmidb / Zuo2025）
│   ├── plot_raw.py         # Day1：读一条数据画原始波形 + PSD
│   ├── lsl_receive_demo.py # Day1：LSL 接收示例 + mock 自测
│   └── preprocess.py       # Day2：带通滤波 + ICA 去眼电 + 切 MI epoch
├── notebooks/          # 分步实验 notebook
├── figures/            # 产出图（day1_raw.png / sub001_epochs_avg.png ...）
├── processed/          # 中间产物（epoch 的 .fif，不入库）
└── models/             # 训练好的模型
```

---

## 环境准备

```bash
cd D:\WorkBuddy_Projects\eeg-reproduce
.venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ 创建 venv 时若用绝对路径 `/d/...` 失败，请先 `cd` 进项目目录再用相对路径 `python -m venv .venv`。

---

## Day 1 运行

```bash
# 1) 下载 eegmmidb 练手数据（左/右手 MI 相关 run）
python src/download_data.py

# 2) 画一条数据的原始波形 + 功率谱
python src/plot_raw.py
# 产出：figures/day1_raw.png, figures/day1_psd.png

# 3) LSL 接收示例（一条命令自测：先发后收）
python src/lsl_receive_demo.py --mode selftest
```

---

## Day 2 运行

```bash
# 带通滤波 1-40 Hz -> ICA 去眼电 -> 按 T1/T2 切 epoch（左拳 vs 右拳）
# 快速冒烟测试（只跑 runs 3 4，验证链路）：
python src/preprocess.py --runs 3 4

# 正式执行（左右拳融合：执行 run3/7/11 + 想象 run4/8/12，可多被试）：
python src/preprocess.py --subjects 1 2 3
# 产出：processed/sub00X_epochs-epo.fif, figures/sub00X_epochs_avg.png
```

---

## 数据集对照

| 数据集 | 有没有左右腿 | 用途 | 备注 |
|--------|--------------|------|------|
| **PhysioNet eegmmidb** | ❌ 只有左拳/右拳、双手/双脚 | 方法练手 + 与 Zuo2025 对照 | 64 通道；T1/T2 含义随 run 变 |
| **Zuo2025** | ✅ left_leg vs right_leg | **主数据集** | 30 被试 / 30ch / 500Hz；CC BY 4.0；baseline=CSP+LDA/FBCSP+SVM/EEGNet |

> 说明：eegmmidb 提供的是左右手分类数据（部分 run 为双手/双脚），**不含左右腿**；
> 左右腿二分类请使用 Zuo2025（见 `src/download_data.py::download_zuo2025`；该数据集需另装 MOABB + Figshare 授权后接入）。

---

## Day 1 实测结果（已跑通）

| 项目 | 结果 |
|------|------|
| venv | `D:\WorkBuddy_Projects\eeg-reproduce\.venv`（Python 3.13.14，pip 26.2.1） |
| 依赖 | mne 1.13.2 / numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1 / pylsl 1.18.5 / pywavelets 1.10.0 等，安装成功 |
| 数据 | `data/eegmmidb/.../S001/S001R03.edf`，**64 通道 / 160 Hz / 125 s** |
| 波形图 | `figures/day1_raw.png`（FC5/FC3/FC1/FCz/FC2/FC4/FC6/C5，前 5 秒） |
| 功率谱 | `figures/day1_psd.png`（Welch 1-45 Hz，10 Hz 附近可见 alpha 隆起） |
| LSL 自测 | `--mode selftest` 通过，接收端连上 `MockEEG`，共收到 1037 个样本（8 通道） |

**踩过的坑（已修，供复用时注意）**
1. 建 venv 时用绝对路径 `/d/...` 会**静默失败**（返回 0 但不落地）→ 必须先 `cd` 进目录再用相对路径 `python -m venv .venv`。
2. `eegbci.load_data(...)` 会**交互式提问**「是否设为默认路径」，非交互环境会 EOFError → 加参数 `update_path=True`。
3. `pylsl` 1.18 的 API 是 `resolve_streams()`（复数，返回列表），不是旧版的 `resolve_stream(prop, val)`。
4. `standard_1005` montage 在 MNE 1.14 会改名 `colin27_1005` → 代码里已做 try/except 兼容。

> 环境说明：本机 mne 下载较慢（约 30 kB/s），eegmmidb 单个 run 文件很小（约 2-3 MB），可接受。

---

## Day 2 实测结果（脚本已跑通）

链路：`eegmmidb 加载 -> 带通滤波 1-40 Hz -> ICA 去眼电 -> events_from_annotations -> Epochs(左拳/右拳)`

| 项目 | 结果 |
|------|------|
| 冒烟测试 | `--runs 3 4`，被试 1，共 **30 段** epoch（左拳 16 / 右拳 14），0 坏段 |
| epoch 窗口 | tmin=-0.5s, tmax=4.0s，baseline=(-0.5, 0s) |
| ICA | 基于 Fp1/Fp2 相关识别出成分 **[0]** 并剔除（眼电） |
| 产出 | `processed/sub001_epochs-epo.fif`、`figures/sub001_epochs_avg.png`（64 通道平均，0.3-0.5s 有诱发峰） |

**Day 2 新增的坑（已修）**
5. **eegmmidb 事件不在 `STI 014` 刺激通道里**，而是存在 **annotations**（标签 `T0/T1/T2`）→
   必须用 `mne.events_from_annotations(raw)`；`events_from_annotations` 会把 `T1` 映射为 2、`T2` 映射为 3，切 epoch 时用返回的 `event_id` 字典，别写死数字。
6. 滤波/ICA 之间：mne 的 `ICA.find_bads_eog` 内部会用 **1-10 Hz** 带通做相关（日志里那段 1-10 Hz 是它，不是主滤波，别被误导）。

---

## 进度

- [x] Day 1：环境 + eegmmidb 读取 + 波形图 + LSL 示例
- [x] Day 2：带通滤波 + ICA 去眼电 + 按事件提取 MI 片段（脚本已跑通，待跑全 runs/多被试）
- [ ] Day 3：alpha/beta PSD + CSP 特征 + LDA/SVM 分类 + 混淆矩阵
- [ ] Day 4：模块化整理 + README + 接入 Zuo2025 左右腿
