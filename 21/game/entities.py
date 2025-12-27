from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Color = Tuple[int, int, int]


@dataclass
class Paddle:
    x: float
    y: float
    width: float
    height: float
    speed: float
    color: Color

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        return self.x + self.width / 2

    @property
    def bottom(self) -> float:
        return self.y - self.height / 2

    @property
    def top(self) -> float:
        return self.y + self.height / 2

    def clamp_to_screen(self, screen_width: float) -> None:
        half = self.width / 2
        if self.x < half:
            self.x = half
        if self.x > screen_width - half:
            self.x = screen_width - half


@dataclass
class Ball:
    x: float
    y: float
    radius: float
    dx: float
    dy: float
    speed: float
    color: Color

    def normalize_velocity(self) -> None:
        mag = (self.dx * self.dx + self.dy * self.dy) ** 0.5
        if mag <= 1e-9:
            self.dx = 0.0
            self.dy = -1.0
            return
        self.dx /= mag
        self.dy /= mag


@dataclass
class Brick:
    x: float
    y: float
    width: float
    height: float
    color: Color

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        return self.x + self.width / 2

    @property
    def bottom(self) -> float:
        return self.y - self.height / 2

    @property
    def top(self) -> float:
        return self.y + self.height / 2
