#!/usr/bin/env python3
"""
YDLIDAR Tmini 测试 — 严格按 C++ YDlidarDriver 逐函数翻译
用法: python3 test_lidar.py [/dev/rplidar]
"""

import sys, time, threading
import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/rplidar"
BAUD = 230400

# ── 协议常量 (ydlidar_protocol.h) ──────
PH1, PH2 = 0xAA, 0x55       # 包头 0x55AA 小端
LIDAR_CMD_SYNC  = 0xA5
LIDAR_CMD_SCAN  = 0x60
LIDAR_CMD_STOP  = 0x65
LIDAR_CMD_FSTOP = 0x00
LIDAR_RESP_CHECKBIT = 0x01
TRI_PACKHEADSIZE = 10
TRI_PACKMAXNODES = 80

latest_scan = []
lock = threading.Lock()


# ═══════════════════════════════════════════
# 对应 C++: YDlidarDriver 成员变量
# ═══════════════════════════════════════════
class State:
    def __init__(self):
        self.nodeIndex = 0           # 当前包内取到第几个点
        self.package_Sample_Num = 0  # 当前包的点数 (count)
        self.FirstSampleAngle = 0    # >>1 后，单位 1/64°
        self.LastSampleAngle = 0     # >>1 后
        self.IntervalSampleAngle = 0.0
        self.IntervalSampleAngle_LastPackage = 0.0
        self.CheckSum = 0
        self.CheckSumCal = 0
        self.CheckSumResult = True
        self.ct = 0                  # CT 原始值
        self.has_package_error = False
        self._raw_samples = bytearray()       # 临时原始字节
        self._parsed_nodes = []               # [(intensity, dist, is_flag), ...]
        self.package_index = 0


# ═══════════════════════════════════════════
# parseResponseHeader (L786)
# ═══════════════════════════════════════════
def parse_response_header(st, ser, timeout_s=1.0):
    """读 10 字节包头，返回 True 表示成功"""
    import time as _time
    start = _time.time()
    recvPos = 0
    pkg = bytearray(TRI_PACKHEADSIZE)
    st.CheckSumCal = 0

    while (_time.time() - start) < timeout_s:
        b = ser.read(1)
        if not b:
            continue
        byte = b[0]

        if recvPos == 0:
            if byte == PH1:
                pkg[0] = byte
                recvPos = 1
            continue

        if recvPos == 1:
            if byte == PH2:
                pkg[1] = byte
                recvPos = 2
                # C++: 校验和从 "PH" (0x55AA) 开始
                st.CheckSumCal = 0x55AA
            elif byte == PH1:  # 连续 AA，保持
                continue
            else:
                recvPos = 0
            continue

        # recvPos >= 2: 读剩余 8 字节
        pkg[recvPos] = byte

        if recvPos == 2:  # CT
            st.ct = byte
        elif recvPos == 3:  # count (LSN)
            st.package_Sample_Num = byte
            # C++: 校验和 XOR CT|LSN 字
            st.CheckSumCal ^= (byte << 8) | st.ct
        elif recvPos == 4:  # firstAngle low
            if byte & LIDAR_RESP_CHECKBIT:
                st.FirstSampleAngle = byte
            else:
                st.has_package_error = True
                recvPos = 0
                continue
        elif recvPos == 5:  # firstAngle high
            # C++: 校验和 XOR 原始 FSA (移位前)
            fsa_raw = (byte << 8) | st.FirstSampleAngle
            st.CheckSumCal ^= fsa_raw
            st.FirstSampleAngle = fsa_raw >> 1  # ← 移位到后面做
        elif recvPos == 6:  # lastAngle low
            if byte & LIDAR_RESP_CHECKBIT:
                st.LastSampleAngle = byte
            else:
                st.has_package_error = True
                recvPos = 0
                continue
        elif recvPos == 7:  # lastAngle high
            # C++: 校验和 XOR 原始 LSA (移位前)
            lsa_raw = (byte << 8) | st.LastSampleAngle
            st.CheckSumCal ^= lsa_raw
            st.LastSampleAngle = lsa_raw >> 1

            # 计算角度间隔
            cnt = st.package_Sample_Num
            if cnt == 1:
                st.IntervalSampleAngle = 0.0
            elif st.LastSampleAngle < st.FirstSampleAngle:
                if st.FirstSampleAngle > 270 * 64 and st.LastSampleAngle < 90 * 64:
                    st.IntervalSampleAngle = (360 * 64 + st.LastSampleAngle - st.FirstSampleAngle) / (cnt - 1)
                    st.IntervalSampleAngle_LastPackage = st.IntervalSampleAngle
                else:
                    st.IntervalSampleAngle = st.IntervalSampleAngle_LastPackage
            else:
                st.IntervalSampleAngle = (st.LastSampleAngle - st.FirstSampleAngle) / (cnt - 1)
                st.IntervalSampleAngle_LastPackage = st.IntervalSampleAngle
        elif recvPos == 8:
            st.CheckSum = byte
        elif recvPos == 9:
            st.CheckSum += byte * 0x100

        recvPos += 1
        if recvPos == TRI_PACKHEADSIZE:
            return True
    return False


# ═══════════════════════════════════════════
# parseResponseScanData (L1012)
# ═══════════════════════════════════════════
def parse_response_scan_data(st, ser, timeout_s=1.0):
    """读 count×3 字节采样数据 (NODE_QUAL8: 1B intensity + 2B distance)"""
    import time as _time
    start = _time.time()
    need = st.package_Sample_Num * 3  # ← NODE_QUAL8: 每点 3 字节

    while (_time.time() - start) < timeout_s:
        b = ser.read(1)
        if not b:
            continue
        byte = b[0]

        # 收集所有原始字节
        st._raw_samples.append(byte)

        if len(st._raw_samples) >= need:
            # 逐点解析: 1B intensity + 2B distance
            st._parsed_nodes = []
            for i in range(st.package_Sample_Num):
                off = i * 3
                intensity = st._raw_samples[off]
                dist_raw = st._raw_samples[off + 1] | (st._raw_samples[off + 2] << 8)
                is_flag = dist_raw & 0x0003
                dist = dist_raw & 0xFFFC
                st._parsed_nodes.append((intensity, dist, is_flag))

                # C++: 校验和 — 1B intensity XOR, then 2B distance XOR
                st.CheckSumCal ^= intensity
                st.CheckSumCal ^= dist_raw

            st._raw_samples = []
            return True
    return False


# ═══════════════════════════════════════════
# parseNodeFromeBuffer (L1252)
# ═══════════════════════════════════════════
def parse_node_from_buffer(st):
    """从缓存中取 nodeIndex 指向的点，返回 (angle_deg, dist_m, intensity, is_sync)"""
    intensity, dist, is_flag = st._parsed_nodes[st.nodeIndex]

    # C++: sync 基于 CT bit[0] (零位包标记)，不依赖 LSN
    is_sync = (st.ct & 0x01) != 0

    angle_q64 = st.FirstSampleAngle + st.IntervalSampleAngle * st.nodeIndex
    angle_deg = angle_q64 / 64.0

    if angle_deg >= 360:
        angle_deg -= 360

    dist_m = dist / 4000.0

    st.nodeIndex += 1
    return angle_deg, dist_m, intensity, is_sync


# ═══════════════════════════════════════════
# waitPackage (L1137)
# ═══════════════════════════════════════════
def wait_package(st, ser, timeout_s=1.0):
    """解析一个包（如需），返回包内第 nodeIndex 个点"""
    if st.nodeIndex >= st.package_Sample_Num:
        st.nodeIndex = 0

    if st.nodeIndex == 0:
        if not parse_response_header(st, ser, timeout_s):
            return None
        cnt = st.package_Sample_Num
        # 合理性检查
        if cnt == 0 or cnt > TRI_PACKMAXNODES:
            st.nodeIndex = 0
            return None
        if not parse_response_scan_data(st, ser, timeout_s):
            return None
        # 校验——跳过校验和不匹配的假包
        if st.CheckSumCal != st.CheckSum:
            st.CheckSumResult = False
        else:
            st.CheckSumResult = True

    return parse_node_from_buffer(st)


# ═══════════════════════════════════════════
# waitScanData (L1383)
# ═══════════════════════════════════════════
def wait_scan_data(st, ser, max_nodes=2000, timeout_s=1.0):
    """收集一圈点，返回到 sync 节点为止的列表"""
    import time as _time
    nodes_list = []
    start = _time.time()

    while len(nodes_list) < max_nodes and (_time.time() - start) < timeout_s:
        result = wait_package(st, ser, timeout_s)
        if result is None:
            continue
        angle_deg, dist_m, intensity, is_sync = result
        nodes_list.append((angle_deg, dist_m, intensity, is_sync))

        if is_sync and len(nodes_list) > 1:
            break

    return nodes_list


# ═══════════════════════════════════════════
# cacheScanData (L613) — 后台线程
# ═══════════════════════════════════════════
def cache_scan_data(st, ser):
    """C++ cacheScanData (L613): 双 sync 节点逻辑"""
    global latest_scan
    local_scan = []  # 累积的 node_info

    while True:
        # waitScanData: 一批节点，末尾是 sync 节点
        batch = wait_scan_data(st, ser, max_nodes=5000, timeout_s=3.0)
        if not batch:
            continue

        for angle_deg, dist_m, intensity, is_sync in batch:
            # C++: if (local_buf[pos].sync & LIDAR_RESP_SYNCBIT)
            if is_sync:
                # C++: if (local_scan[0].sync & LIDAR_RESP_SYNCBIT)
                if local_scan and local_scan[0][3]:  # 前一个也是 sync → 一圈完成
                    circle = [(a, d) for a, d, _, _ in local_scan
                              if 0.05 < d < 12.0]
                    if circle:
                        with lock:
                            latest_scan = circle
                local_scan = []

            local_scan.append((angle_deg, dist_m, intensity, is_sync))
            # C++: if (scan_count == MAX) scan_count -= 1;
            if len(local_scan) > 4096:
                local_scan.pop(0)


# ═══════════════════════════════════════════
# main
# ═══════════════════════════════════════════
def main():
    ser = serial.Serial(PORT, BAUD, timeout=5)
    print(f"[INFO] 打开 {PORT}")

    # Tmini 电机不由 DTR 控制（tmini_test.cpp: MotorDtrCtrl=false）
    # ser.setDTR(True)  ← 不设！

    # stopScan: 强制停止 + 停止
    ser.write(bytes([LIDAR_CMD_SYNC, LIDAR_CMD_FSTOP]))
    time.sleep(0.05)
    ser.write(bytes([LIDAR_CMD_SYNC, LIDAR_CMD_STOP]))
    time.sleep(0.1)
    ser.reset_input_buffer()

    # startScan: 发送扫描命令
    print("[CMD] A5 60")
    ser.write(bytes([LIDAR_CMD_SYNC, LIDAR_CMD_SCAN]))

    # 跳过应答头，直接开始 (C++ 里有 waitResponseHeader，这里简化)
    time.sleep(0.5)

    st = State()
    print("[INFO] 开始扫描...")

    t = threading.Thread(target=cache_scan_data, args=(st, ser), daemon=True)
    t.start()

    # ── matplotlib ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("YDLIDAR Tmini")
    ax.grid(True); ax.set_aspect('equal')
    ax.plot([0], [0], 'ro', markersize=10)
    (sc,) = ax.plot([], [], 'g.', markersize=1, alpha=0.6)

    def update(_):
        with lock:
            pts = latest_scan[:] if latest_scan else []
        if pts:
            xs = [d * np.cos(np.radians(a)) for a, d, in pts]
            ys = [d * np.sin(np.radians(a)) for a, d, in pts]
            sc.set_data(xs, ys)
        return sc,

    _ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.show()

    ser.write(bytes([LIDAR_CMD_SYNC, LIDAR_CMD_STOP]))
    ser.close()


if __name__ == "__main__":
    main()
