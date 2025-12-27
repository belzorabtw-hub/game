from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Tuple

import torch

from rl.headless_env import HeadlessBreakoutEnv
from rl.model import MLP


class EvaluatorThread(threading.Thread):
    """
    Поток для оценки производительности модели без обучения.
    Запускает N эпизодов с фиксированной моделью и собирает статистику.
    """
    
    def __init__(
        self,
        model_path: str,
        n_episodes: int = 100,
        log_interval: int = 10,
        max_episode_steps: int = 6000,
        obs_dim: int = 5,
        n_actions: int = 3,
    ) -> None:
        super().__init__(daemon=True)
        self.model_path = model_path
        self.n_episodes = n_episodes
        self.log_interval = log_interval
        self.max_episode_steps = max_episode_steps
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EVAL] device = {self.device}")
        
        # Загружаем модель для оценки
        self.model = MLP(obs_dim, n_actions).to(self.device)
        self.model.eval()
        self._load_model()
        
        # Создаем окружение
        self.env = HeadlessBreakoutEnv()
        
        # Статистика
        self.rewards: Deque[float] = deque(maxlen=n_episodes)
        self.episode_lengths: Deque[int] = deque(maxlen=n_episodes)
        self.wins: Deque[int] = deque(maxlen=n_episodes)
        self.bricks_broken: Deque[int] = deque(maxlen=n_episodes)
        
        self._stop_event = threading.Event()
    
    def _load_model(self) -> None:
        """Загрузка модели из файла"""
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state["policy"])
        print(f"[EVAL] Model loaded from {self.model_path}")
    
    def select_action(self, obs: Tuple[float, float, float, float, float]) -> int:
        """Выбор действия по argmax(Q) без exploration"""
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(x)
            return int(torch.argmax(q, dim=1).item())
    
    def stop(self) -> None:
        self._stop_event.set()
    
    def run(self) -> None:
        print(f"[EVAL] Starting evaluation for {self.n_episodes} episodes...")
        start_time = time.time()
        
        for episode in range(1, self.n_episodes + 1):
            if self._stop_event.is_set():
                break
            
            obs = self.env.reset()
            
            total_reward = 0.0
            steps = 0
            done = False
            last_info: Dict = {}
            bricks_broken_in_episode = 0
            
            while not done and steps < self.max_episode_steps:
                action = self.select_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                
                obs = next_obs
                total_reward += reward
                steps += 1
                last_info = info
                
                if info.get("broke_brick", False):
                    bricks_broken_in_episode += 1
            
            win = bool(last_info.get("win", False))
            
            # Сохраняем статистику
            self.rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.wins.append(1 if win else 0)
            self.bricks_broken.append(bricks_broken_in_episode)
            
            # Логирование прогресса
            if episode % self.log_interval == 0 or episode == self.n_episodes:
                win_rate = sum(self.wins) / len(self.wins) if len(self.wins) > 0 else 0.0
                avg_reward = sum(self.rewards) / len(self.rewards) if len(self.rewards) > 0 else 0.0
                avg_steps = sum(self.episode_lengths) / len(self.episode_lengths) if len(self.episode_lengths) > 0 else 0.0
                avg_bricks = sum(self.bricks_broken) / len(self.bricks_broken) if len(self.bricks_broken) > 0 else 0.0
                
                print(
                    f"[EVAL] ep={episode}/{self.n_episodes} "
                    f"win_rate={win_rate:.3f} "
                    f"avg_reward={avg_reward:.2f} "
                    f"avg_steps={avg_steps:.1f} "
                    f"avg_bricks={avg_bricks:.1f}"
                )
        
        # Финальная статистика
        elapsed_time = time.time() - start_time
        if len(self.wins) > 0:
            final_win_rate = sum(self.wins) / len(self.wins)
            final_avg_reward = sum(self.rewards) / len(self.rewards)
            final_avg_steps = sum(self.episode_lengths) / len(self.episode_lengths)
            final_avg_bricks = sum(self.bricks_broken) / len(self.bricks_broken)
            
            print("\n" + "="*60)
            print("[EVAL] FINAL RESULTS:")
            print(f"  Episodes completed: {len(self.wins)}")
            print(f"  Win rate: {final_win_rate:.3f} ({sum(self.wins)} wins)")
            print(f"  Average reward: {final_avg_reward:.2f}")
            print(f"  Average steps per episode: {final_avg_steps:.1f}")
            print(f"  Average bricks broken: {final_avg_bricks:.1f}")
            print(f"  Total time: {elapsed_time:.1f}s")
            print(f"  Time per episode: {elapsed_time/len(self.wins):.2f}s")
            print("="*60)
        
        print("[EVAL] Evaluation completed.")