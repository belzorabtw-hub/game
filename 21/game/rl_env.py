from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from config import R_BREAK_BRICK, R_HIT_PADDLE, R_LOSE, R_WIN
from game.game_state import GameWindow


@dataclass
class StepInfo:
    hit_paddle: bool
    broke_brick: bool
    win: bool
    lose: bool


class BreakoutEnv:
    """
    Gym-подобная среда, которая управляет уже существующей логикой игры через GameWindow.
    Визуализация остаётся в GameWindow; env предоставляет reset/step и наблюдения.
    """

    def __init__(self, window: GameWindow, fixed_dt: float = 1 / 60) -> None:
        self.window = window
        self.fixed_dt = fixed_dt

    def reset(self) -> Tuple[float, float, float, float, float]:
        self.window.setup()
        return self.get_obs()

    def get_obs(self) -> Tuple[float, float, float, float, float]:
        # obs в диапазоне [-1..1] для всех компонент
        assert self.window.paddle is not None
        assert self.window.ball is not None

        w = float(self.window.width)
        h = float(self.window.height)

        ball_x = (self.window.ball.x / w) * 2.0 - 1.0
        ball_y = (self.window.ball.y / h) * 2.0 - 1.0
        ball_vx = float(self.window.ball.dx)  # уже нормализовано
        ball_vy = float(self.window.ball.dy)  # уже нормализовано
        paddle_x = (self.window.paddle.x / w) * 2.0 - 1.0

        # страховка от минимальных численных выходов за пределы
        ball_x = max(-1.0, min(1.0, ball_x))
        ball_y = max(-1.0, min(1.0, ball_y))
        ball_vx = max(-1.0, min(1.0, ball_vx))
        ball_vy = max(-1.0, min(1.0, ball_vy))
        paddle_x = max(-1.0, min(1.0, paddle_x))

        return (ball_x, ball_y, ball_vx, ball_vy, paddle_x)

    def step(self, action: int) -> Tuple[Tuple[float, float, float, float, float], float, bool, Dict]:
        # Управление делается через action_provider в GameWindow,
        # но step доступен для внешнего кода (например, будущего обучения).
        # Здесь мы просто пробрасываем action через input_handler, аналогично GameWindow.
        if action == 1:
            self.window.input_handler.left_pressed = True
            self.window.input_handler.right_pressed = False
        elif action == 2:
            self.window.input_handler.left_pressed = False
            self.window.input_handler.right_pressed = True
        else:
            self.window.input_handler.left_pressed = False
            self.window.input_handler.right_pressed = False

        self.window.on_update(self.fixed_dt)

        info = self._build_info()
        reward = 0.0

        if info.hit_paddle:
            reward += R_HIT_PADDLE
        if info.broke_brick:
            reward += R_BREAK_BRICK
        if info.win:
            reward += R_WIN
        if info.lose:
            reward += R_LOSE

        done = info.win or info.lose
        return self.get_obs(), reward, done, {
            "hit_paddle": info.hit_paddle,
            "broke_brick": info.broke_brick,
            "win": info.win,
            "lose": info.lose,
        }

    def _build_info(self) -> StepInfo:
        hit_paddle = bool(getattr(self.window, "last_hit_paddle", False))
        broke_brick = bool(getattr(self.window, "last_broke_brick", False))
        win = bool(getattr(self.window, "is_game_over", False) and getattr(self.window, "is_win", False))
        lose = bool(getattr(self.window, "is_game_over", False) and not getattr(self.window, "is_win", False))
        return StepInfo(hit_paddle=hit_paddle, broke_brick=broke_brick, win=win, lose=lose)
