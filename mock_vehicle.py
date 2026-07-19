#!/usr/bin/env python3
"""
mock_vehicle.py — 2D 小车模拟器
连接后持续发送 pose 数据，验证 WebSocket 通信。
"""

import asyncio
import json
import signal
import time

import websockets
from websockets.asyncio.server import serve

HOST = "0.0.0.0"
PORT = 9090


async def handler(websocket):
    addr = websocket.remote_address
    print(f"[+] client connected: {addr}")

    x = 10.0
    y = 10.0
    pose_count = 0

    # 先发地图
    import random
    random.seed(42)
    voxels = []
    size = 256  # 完整 Chunk
    for gx in range(size):
        for gy in range(size):
            is_wall = random.random() < 0.05
            voxels.append({"gx": gx, "gy": gy, "gz": 0, "state": 1 if is_wall else 0, "conf": 1.0})
    map_msg = json.dumps({"type": "map_full", "ts": time.time(), "voxels": voxels})
    print(f"[→] sending map_full ({len(map_msg)} bytes, {len(voxels)} cells)")
    await websocket.send(map_msg)
    print(f"[✓] map_full sent")

    try:
        while True:
            msg = json.dumps({
                "type": "pose",
                "ts": time.time(),
                "x": x,
                "y": y,
                "z": 0.0,
                "yaw": 0.0,
                "vx": 0.5,
                "vy": 0.0,
            })
            await websocket.send(msg)
            x += 0.5
            pose_count += 1
            print(f"[→] pose #{pose_count}: x={x:.1f} y={y:.1f}")
            await asyncio.sleep(1.0)

    except Exception as e:
        print(f"[!] error: {e}")
    print(f"[-] client disconnected: {addr}")


async def main():
    stop = asyncio.Event()

    def _sig_handler(signum, frame):
        print("\n[!] shutting down...")
        stop.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    async with serve(handler, HOST, PORT):
        print(f"Mock Vehicle Server listening on ws://{HOST}:{PORT}")
        print("Waiting for Godot to connect...\n")
        await stop.wait()


if __name__ == "__main__":
    asyncio.run(main())
