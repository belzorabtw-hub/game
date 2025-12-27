from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from config import (
    BALL_COLOR,
    BALL_RADIUS,
    BALL_SPEED,
    BRICK_COLS,
    BRICK_COLOR,
    BRICK_GAP,
    BRICK_HEIGHT,
    BRICK_ROWS,
    BRICK_SIDE_MARGIN,
    BRICK_TOP_MARGIN,
    PADDLE_COLOR,
    PADDLE_HEIGHT,
    PADDLE_SPEED,
    PADDLE_WIDTH,
    PADDLE_Y_OFFSET,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    R_BREAK_BRICK,
    R_HIT_PADDLE,
    R_LOSE,
    R_WIN,
    CURRICULUM_LEVELS,
)
from game.entities import Ball, Brick, Paddle
from game.physics import reflect_ball_from_brick, reflect_ball_from_paddle, reflect_ball_from_walls


class HeadlessBreakoutEnv:
    """
    Headless-версия среды для быстрых эпизодов без рендера.
    API совпадает по смыслу с BreakoutEnv: reset/step и obs (5 значений), actions 0/1/2.
    Obs: (ball_x, ball_y, ball_vx, ball_vy, paddle_x) в [-1..1]
    """

    def __init__(self, fixed_dt: float = 1 / 120) -> None:
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.fixed_dt = fixed_dt

        self.paddle: Paddle | None = None
        self.ball: Ball | None = None
        self.bricks: List[Brick] = []

        self.is_game_over = False
        self.is_win = False

        self.difficulty_level = 0
        self._level_paddle_width = PADDLE_WIDTH
        self._level_ball_speed = BALL_SPEED
        self._level_brick_rows = BRICK_ROWS
        self._level_brick_cols = BRICK_COLS
        self._level_angle_min_deg = 35
        self._level_angle_max_deg = 145
        
        # Анти-эксплойт: счетчик для рандомизации стартового угла
        self._start_angle_counter = 0

    def set_difficulty(self, level: int) -> None:
        if level < 0:
            level = 0
        if level >= len(CURRICULUM_LEVELS):
            level = len(CURRICULUM_LEVELS) - 1

        self.difficulty_level = level

        paddle_width, ball_speed_mult, brick_rows, brick_cols, a_min, a_max = CURRICULUM_LEVELS[level]

        self._level_paddle_width = float(paddle_width)
        self._level_ball_speed = float(BALL_SPEED) * float(ball_speed_mult)
        self._level_brick_rows = int(brick_rows)
        self._level_brick_cols = int(brick_cols)
        self._level_angle_min_deg = int(a_min)
        self._level_angle_max_deg = int(a_max)

    def reset(self) -> Tuple[float, float, float, float, float]:
        self.is_game_over = False
        self.is_win = False

        # Используем текущие настройки сложности
        self.paddle = Paddle(
            x=self.width / 2,
            y=PADDLE_Y_OFFSET,
            width=self._level_paddle_width,
            height=PADDLE_HEIGHT,
            speed=PADDLE_SPEED,
            color=PADDLE_COLOR,
        )

        # Анти-эксплойт: рандомизация стартового угла с небольшим смещением
        self._start_angle_counter += 1
        
        # Базовый угол из допустимого диапазона
        base_min_angle = math.radians(self._level_angle_min_deg)
        base_max_angle = math.radians(self._level_angle_max_deg)
        
        # Добавляем небольшую случайную вариацию (±10 градусов)
        variation = math.radians(random.uniform(-10, 10))
        
        # Основной случайный угол
        angle = random.uniform(base_min_angle, base_max_angle) + variation
        
        # Ограничиваем угол разумными пределами (20-160 градусов)
        angle = max(math.radians(20), min(math.radians(160), angle))
        
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Гарантируем, что мяч летит вверх
        if dy <= 0:
            dy = abs(dy) + 0.1
        
        # Нормализация скорости
        speed = math.sqrt(dx*dx + dy*dy)
        dx /= speed
        dy /= speed

        self.ball = Ball(
            x=self.width / 2,
            y=self.paddle.top + BALL_RADIUS + 6,
            radius=BALL_RADIUS,
            dx=dx,
            dy=dy,
            speed=self._level_ball_speed,
            color=BALL_COLOR,
        )
        # Убедимся, что скорость нормализована
        self.ball.normalize_velocity()

        self.bricks = self._build_bricks()
        return self.get_obs()

    def _build_bricks(self) -> List[Brick]:
        bricks: List[Brick] = []
        available_width = self.width - (2 * BRICK_SIDE_MARGIN)
        total_gap = (self._level_brick_cols - 1) * BRICK_GAP
        brick_width = (available_width - total_gap) / self._level_brick_cols

        for row in range(self._level_brick_rows):
            for col in range(self._level_brick_cols):
                x = BRICK_SIDE_MARGIN + brick_width / 2 + col * (brick_width + BRICK_GAP)
                y = self.height - BRICK_TOP_MARGIN - row * (BRICK_HEIGHT + BRICK_GAP)
                bricks.append(
                    Brick(
                        x=x,
                        y=y,
                        width=brick_width,
                        height=BRICK_HEIGHT,
                        color=BRICK_COLOR,
                    )
                )
        return bricks

    def get_obs(self) -> Tuple[float, float, float, float, float]:
        assert self.paddle is not None
        assert self.ball is not None

        w = float(self.width)
        h = float(self.height)

        ball_x = (self.ball.x / w) * 2.0 - 1.0
        ball_y = (self.ball.y / h) * 2.0 - 1.0
        ball_vx = float(self.ball.dx)
        ball_vy = float(self.ball.dy)
        paddle_x = (self.paddle.x / w) * 2.0 - 1.0

        ball_x = max(-1.0, min(1.0, ball_x))
        ball_y = max(-1.0, min(1.0, ball_y))
        ball_vx = max(-1.0, min(1.0, ball_vx))
        ball_vy = max(-1.0, min(1.0, ball_vy))
        paddle_x = max(-1.0, min(1.0, paddle_x))

        return (ball_x, ball_y, ball_vx, ball_vy, paddle_x)

    def step(self, action: int) -> Tuple[Tuple[float, float, float, float, float], float, bool, Dict]:
        if self.paddle is None or self.ball is None:
            raise RuntimeError("Env not reset()")

        if self.is_game_over:
            return self.get_obs(), 0.0, True, {"win": self.is_win, "lose": not self.is_win}

        # action: 0 stay, 1 left, 2 right
        move_dir = 0.0
        if action == 1:
            move_dir = -1.0
        elif action == 2:
            move_dir = 1.0

        self.paddle.x += move_dir * self.paddle.speed * self.fixed_dt
        self.paddle.clamp_to_screen(self.width)

        self.ball.x += self.ball.dx * self.ball.speed * self.fixed_dt
        self.ball.y += self.ball.dy * self.ball.speed * self.fixed_dt

        reflect_ball_from_walls(self.ball, self.width, self.height)

        hit_paddle = reflect_ball_from_paddle(self.ball, self.paddle)

        broke_brick = False
        hit_index = None
        for i, brick in enumerate(self.bricks):
            if reflect_ball_from_brick(self.ball, brick):
                hit_index = i
                break
        if hit_index is not None:
            self.bricks.pop(hit_index)
            broke_brick = True

        if self.ball.y + self.ball.radius < 0:
            self.is_game_over = True
            self.is_win = False

        if len(self.bricks) == 0:
            self.is_game_over = True
            self.is_win = True

        reward = 0.0
        if hit_paddle:
            reward += R_HIT_PADDLE
        if broke_brick:
            reward += R_BREAK_BRICK
        if self.is_game_over and self.is_win:
            reward += R_WIN
        if self.is_game_over and not self.is_win:
            reward += R_LOSE

        done = self.is_game_over
        info = {
            "hit_paddle": hit_paddle,
            "broke_brick": broke_brick,
            "win": self.is_game_over and self.is_win,
            "lose": self.is_game_over and (not self.is_win),
            "bricks_left": len(self.bricks),
        }
        return self.get_obs(), reward, done, info