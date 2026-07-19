#!/usr/bin/env python3
"""
mock_visual.py — Mock Server Pygame 可视化测试

24×24 地图，圆形车辆 (r=0.5)，W/S/A/D 驾驶，实时碰撞检测。

用法:
  cd test_tool && python mock_visual.py
"""

import math
import random
import sys

import pygame

from mock_map_grid import MapGrid
from mock_collision import is_circle_passable

# ── 配置 ────────────────────────────────────────────────

MAP_W = 24
MAP_H = 24
CELL = 25                # px/cell
WALL_DENSITY = 0.10      # 10% 墙密度
R = 0.5                  # 车辆半径 (m)
MOVE_SPEED = 0.2         # 每帧移动距离 (m)
TURN_SPEED = math.radians(5)  # 每帧转向角度

# 窗口
PANEL_H = 40             # 底部状态栏高度
WINDOW_W = MAP_W * CELL
WINDOW_H = MAP_H * CELL + PANEL_H
FPS = 60

# 颜色
C_BG = (30, 30, 30)
C_GRID = (60, 60, 60)
C_FREE = (200, 200, 200)
C_WALL = (100, 100, 100)
C_VEHICLE = (0, 140, 255)
C_VEHICLE_HIT = (255, 60, 60)
C_DIR = (255, 255, 255)
C_TEXT = (220, 220, 220)

# ── 辅助 ────────────────────────────────────────────────

def world_to_screen(wx: float, wy: float) -> tuple[float, float]:
    return (wx * CELL, wy * CELL)


def generate_map() -> MapGrid:
    """生成 24×24 随机地图"""
    random.seed(42)
    walls = set()
    for gx in range(MAP_W):
        for gy in range(MAP_H):
            if random.random() < WALL_DENSITY:
                walls.add((gx, gy))
    # 保证起点 (2, 2) 周围 3×3 无障碍
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            walls.discard((2 + dx, 2 + dy))
    return MapGrid.from_wall_set(MAP_W, MAP_H, walls)


# ── 车辆 ────────────────────────────────────────────────

class Vehicle:
    def __init__(self, x: float, y: float, yaw: float = 0.0):
        self.x = x
        self.y = y
        self.yaw = yaw      # rad, 0 = 右
        self.hit = False

    def reset(self):
        self.x, self.y = 2.5, 2.5
        self.yaw = 0.0
        self.hit = False

    def try_move(self, grid: MapGrid, distance: float) -> bool:
        """沿 yaw 前进 distance，碰撞则不动"""
        nx = self.x + distance * math.cos(self.yaw)
        ny = self.y + distance * math.sin(self.yaw)
        if is_circle_passable(grid, nx, ny, R):
            self.x, self.y = nx, ny
            self.hit = False
            return True
        self.hit = True
        return False

    def try_strafe(self, grid: MapGrid, distance: float) -> bool:
        """垂直于 yaw 平移"""
        nx = self.x + distance * math.cos(self.yaw + math.pi / 2)
        ny = self.y + distance * math.sin(self.yaw + math.pi / 2)
        if is_circle_passable(grid, nx, ny, R):
            self.x, self.y = nx, ny
            self.hit = False
            return True
        self.hit = True
        return False


# ── 渲染 ────────────────────────────────────────────────

def draw_map(screen: pygame.Surface, grid: MapGrid):
    for gy in range(MAP_H):
        for gx in range(MAP_W):
            sx, sy = gx * CELL, gy * CELL
            if grid.is_wall(gx, gy):
                pygame.draw.rect(screen, C_WALL, (sx, sy, CELL, CELL))
            else:
                pygame.draw.rect(screen, C_FREE, (sx, sy, CELL, CELL))
            pygame.draw.rect(screen, C_GRID, (sx, sy, CELL, CELL), 1)


def draw_vehicle(screen: pygame.Surface, v: Vehicle):
    sx, sy = world_to_screen(v.x, v.y)
    r_px = int(R * CELL)
    color = C_VEHICLE_HIT if v.hit else C_VEHICLE
    pygame.draw.circle(screen, color, (int(sx), int(sy)), r_px)

    # 方向线
    end_x = sx + r_px * 1.5 * math.cos(v.yaw)
    end_y = sy + r_px * 1.5 * math.sin(v.yaw)
    pygame.draw.line(screen, C_DIR, (int(sx), int(sy)), (int(end_x), int(end_y)), 2)


def draw_status(screen: pygame.Surface, v: Vehicle, font: pygame.font.Font):
    y = MAP_H * CELL + 5
    texts = [
        f"Pose: ({v.x:.1f}, {v.y:.1f})  Yaw: {math.degrees(v.yaw):.0f}°",
        f"Collision: {'YES 🔴' if v.hit else 'NO'}",
        "[W/S] 前进/后退  [A/D] 转向  [Q/E] 平移  [R] 重置  [ESC] 退出",
    ]
    for i, txt in enumerate(texts):
        surf = font.render(txt, True, C_TEXT)
        screen.blit(surf, (10, y + i * 18))


# ── 主循环 ──────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Mock Server — Pygame Visual (24x24, r=0.5)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    grid = generate_map()
    vehicle = Vehicle(2.5, 2.5, yaw=0.0)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # ── 事件 ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    vehicle.reset()

        # ── 输入 ──
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            vehicle.try_move(grid, MOVE_SPEED)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            vehicle.try_move(grid, -MOVE_SPEED)
        if keys[pygame.K_a]:
            vehicle.yaw -= TURN_SPEED
        if keys[pygame.K_d]:
            vehicle.yaw += TURN_SPEED
        if keys[pygame.K_q]:
            vehicle.try_strafe(grid, -MOVE_SPEED)
        if keys[pygame.K_e]:
            vehicle.try_strafe(grid, MOVE_SPEED)

        # ── 渲染 ──
        screen.fill(C_BG)
        draw_map(screen, grid)
        draw_vehicle(screen, vehicle)
        draw_status(screen, vehicle, font)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
