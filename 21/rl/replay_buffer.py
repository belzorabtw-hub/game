from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple


@dataclass
class Transition:
    obs: Tuple[float, float, float, float, float]
    action: int
    reward: float
    next_obs: Tuple[float, float, float, float, float]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, t: Transition) -> None:
        self.data.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.data, batch_size)
