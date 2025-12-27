from __future__ import annotations

from typing import Tuple


class ScriptedPolicy:
    """
    Простой baseline: держаться под шариком.
    Действия:
      0 = stay
      1 = left
      2 = right
    """

    def __init__(self, deadzone: float = 0.03) -> None:
        self.deadzone = deadzone

    def act(self, obs: Tuple[float, float, float, float, float, float, float]) -> int:
        ball_x, _ball_y, _ball_vx, _ball_vy, paddle_x, _bricks_left, _top_brick_y = obs

        diff = ball_x - paddle_x
        if diff > self.deadzone:
            return 2
        if diff < -self.deadzone:
            return 1
        return 0
