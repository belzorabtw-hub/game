from __future__ import annotations

import os
import threading
from typing import Dict, Tuple

import torch

from rl.model import MLP


class RenderPolicy:
    """
    Политика для окна (CPU): периодически получает snapshot весов от тренера
    и выбирает action по argmax(Q).
    """

    def __init__(self, model_path: str, obs_dim: int = 5, n_actions: int = 3) -> None:
        self.model_path = model_path
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.device = torch.device("cpu")
        self.model = MLP(obs_dim, n_actions).to(self.device)
        self.model.eval()

        self._lock = threading.Lock()
        self._has_weights = False
        self._last_update_time = 0.0
        self._update_count = 0

        # АВТОЗАГРУЗКА: загружаем если файл существует
        if os.path.isfile(self.model_path):
            self.load(self.model_path)
            print(f"[LOAD] render model loaded from {self.model_path}")
        else:
            print(f"[LOAD] render model not found: {self.model_path}, using random init")

    def set_snapshot(self, state_dict: Dict[str, torch.Tensor]) -> None:
        with self._lock:
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self._has_weights = True
            self._update_count += 1
            
            if self._update_count % 10 == 0:
                print(f"[RENDER] Snapshot updated (count={self._update_count})")

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["policy"])
        self.model.eval()
        self._has_weights = True

    def act(self, obs: Tuple[float, float, float, float, float]) -> int:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(x)
            return int(torch.argmax(q, dim=1).item())