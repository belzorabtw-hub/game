from __future__ import annotations

import os
import threading
import tempfile
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Adam

from rl.model import MLP
from rl.replay_buffer import ReplayBuffer, Transition


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        model_path: str,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_capacity: int = 100_000,
        batch_size: int = 128,
        warmup_steps: int = 2_000,
        target_update_steps: int = 2_000,
    ) -> None:
        # СТРОГИЙ GPU-РЕЖИМ: запрет fallback на CPU
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA не доступна. Эта программа требует GPU для обучения. "
                "Пожалуйста, используйте GPU с поддержкой CUDA."
            )
        
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.model_path = model_path

        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_steps = target_update_steps

        # ТОЛЬКО GPU, никакого fallback
        self.device = torch.device("cuda")
        
        # Проверка что мы действительно на GPU
        if self.device.type != "cuda":
            raise RuntimeError(f"Ожидался device='cuda', но получен {self.device}")
        
        print(f"[GPU] TRAINING DEVICE: CUDA")
        print(f"[GPU] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] CUDA память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Модели создаются ТОЛЬКО на GPU
        self.policy_net = MLP(obs_dim, n_actions).to(self.device)
        self.target_net = MLP(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(buffer_capacity)

        self.global_step = 0
        self._snapshot_lock = threading.Lock()
        self._snapshot_state_dict: Dict[str, torch.Tensor] | None = None

        if os.path.isfile(self.model_path):
            self.load(self.model_path)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

    def select_action(self, obs: Tuple[float, float, float, float, float, float, float], epsilon: float) -> int:
        # Генерация случайного числа на GPU для консистентности
        if torch.rand(1, device=self.device).item() < epsilon:
            return int(torch.randint(low=0, high=self.n_actions, size=(1,), device=self.device).item())

        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(x)
            return int(torch.argmax(q, dim=1).item())

    def add_transition(self, t: Transition) -> None:
        self.buffer.add(t)

    def can_train(self) -> bool:
        return len(self.buffer) >= max(self.warmup_steps, self.batch_size)

    def train_step(self) -> float | None:
        if not self.can_train():
            return None

        batch = self.buffer.sample(self.batch_size)

        # ВСЕ тензоры создаются на GPU
        obs = torch.tensor([b.obs for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.tensor([b.next_obs for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Прямой проход на GPU
        q_values = self.policy_net(obs).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.gamma * next_q

        # Вычисление потерь на GPU
        loss = self.loss_fn(q_values, target)

        # Обратное распространение на GPU
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Клиппинг градиентов для стабильности
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()

        self.global_step += 1

        if self.global_step % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        return float(loss.item())

    def save(self, path: str | None = None) -> None:
        """Атомарное сохранение модели через временный файл"""
        p = path if path is not None else self.model_path
        
        state = {
            "policy": self.policy_net.state_dict(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
        }
        
        # Создаем временный файл для атомарной записи
        temp_dir = os.path.dirname(p) or "."
        with tempfile.NamedTemporaryFile(mode='wb', dir=temp_dir, delete=False) as tmp_file:
            temp_path = tmp_file.name
            torch.save(state, temp_path)
        
        # Атомарная замена файла
        try:
            os.replace(temp_path, p)
            print(f"[TRAIN] Model saved to {p}")
        except Exception as e:
            # Если что-то пошло не так, удаляем временный файл
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    def load(self, path: str | None = None) -> None:
        p = path if path is not None else self.model_path
        # Загружаем на GPU
        state = torch.load(p, map_location=self.device)
        self.policy_net.load_state_dict(state["policy"])
        print(f"[TRAIN] Model loaded from {p} (on GPU)")

    def update_snapshot(self) -> None:
        # Сохраняем CPU-копию весов для рендера
        cpu_sd: Dict[str, torch.Tensor] = {}
        for k, v in self.policy_net.state_dict().items():
            cpu_sd[k] = v.detach().to("cpu").clone()
        with self._snapshot_lock:
            self._snapshot_state_dict = cpu_sd

    def get_snapshot(self) -> Dict[str, torch.Tensor] | None:
        with self._snapshot_lock:
            if self._snapshot_state_dict is None:
                return None
            return dict(self._snapshot_state_dict)
