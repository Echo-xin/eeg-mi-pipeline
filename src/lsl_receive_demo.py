"""
LSL 接收示例代码（Day 1）

演示脑机实验里最常用的一段链路：
    resolve_streams  ->  StreamInlet  ->  pull_chunk

同时也内置一个 mock sender，在没有真实脑电设备时，可以本地自测整条接收链路
（便于在没有真实脑电设备时本地验证整条接收链路）。

用法：
    # 终端1：启动 mock 数据流（模拟设备推流）
    python src/lsl_receive_demo.py --mode send
    # 终端2：接收并打印
    python src/lsl_receive_demo.py --mode receive

    # 或一条命令自测（线程内先发后收）
    python src/lsl_receive_demo.py --mode selftest
"""

import argparse
import threading
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams

STREAM_NAME = "MockEEG"
N_CHANNELS = 8
SFREQ = 250


def start_sender(duration_s: int = 10):
    """模拟一台 8 通道 EEG 设备，以 250Hz 往 LSL 推流。"""
    info = StreamInfo(
        name=STREAM_NAME,
        type="EEG",
        channel_count=N_CHANNELS,
        nominal_srate=SFREQ,
        channel_format="float32",
        source_id="mockuid1234",
    )
    outlet = StreamOutlet(info)
    print(f"[sender] 推流中：{N_CHANNELS} 通道 @ {SFREQ}Hz，持续 {duration_s}s ...")
    start = time.time()
    while time.time() - start < duration_s:
        # 模拟 8 通道 EEG：10Hz 正弦 + 噪声
        sample = (
            np.sin(2 * np.pi * 10 * time.time()) + np.random.randn(N_CHANNELS) * 0.1
        ).tolist()
        outlet.push_sample(sample)
        time.sleep(1.0 / SFREQ)
    print("[sender] 结束")


def start_receiver(timeout_s: int = 12):
    """查找流 -> 建立 inlet -> 循环拉取 chunk。"""
    print("[receiver] 正在查找流 ...")
    streams = resolve_streams(wait_time=2.0)
    streams = [s for s in streams if s.name() == STREAM_NAME]
    inlet = StreamInlet(streams[0])
    print(f"[receiver] 已连接：{inlet.info().name()}")

    t0 = time.time()
    total = 0
    while time.time() - t0 < timeout_s:
        # pull_chunk：一次拉取本采集周期内到达的所有样本（不丢时间戳）
        samples, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=SFREQ)
        if timestamps:
            arr = np.asarray(samples)
            total += len(timestamps)
            print(
                f"[receiver] 收到 {len(timestamps)} 样本，形状={arr.shape}，"
                f"均值={arr.mean():.3f}"
            )
    print(f"[receiver] 结束，共接收 {total} 样本")


def selftest():
    """先发 8 秒，1.5 秒后开始收，验证链路通。"""
    t = threading.Thread(target=start_sender, args=(8,), daemon=True)
    t.start()
    time.sleep(1.5)  # 等流在 LSL 网络里注册好
    start_receiver(timeout_s=8)


def main():
    ap = argparse.ArgumentParser(description="LSL 接收 / 发送 演示")
    ap.add_argument(
        "--mode",
        choices=["send", "receive", "selftest"],
        default="selftest",
        help="send=只发流；receive=只收流；selftest=一条命令自测",
    )
    args = ap.parse_args()
    if args.mode == "send":
        start_sender()
    elif args.mode == "receive":
        start_receiver()
    else:
        selftest()


if __name__ == "__main__":
    main()
