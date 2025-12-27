from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Tuple

from rl.dqn_agent import DQNAgent
from rl.headless_env import HeadlessBreakoutEnv
from rl.replay_buffer import Transition
from rl.render_policy import RenderPolicy
from config import CURRICULUM_LEVELS, CURRICULUM_WINDOW, CURRICULUM_WIN_RATE_THRESHOLD


class TrainerThread(threading.Thread):
    def __init__(
        self,
        agent: DQNAgent,
        render_policy: RenderPolicy,
        max_episode_steps: int = 6000,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay_steps: int = 200_000,
        snapshot_interval_sec: float = 1.0,
        save_interval_sec: float = 10.0,
        log_window: int = 50,
    ) -> None:
        super().__init__(daemon=True)
        self.agent = agent
        self.render_policy = render_policy

        self.env = HeadlessBreakoutEnv()
        self.max_episode_steps = max_episode_steps

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps

        self.snapshot_interval_sec = snapshot_interval_sec
        self.save_interval_sec = save_interval_sec

        self.log_window = log_window
        self.recent_rewards: Deque[float] = deque(maxlen=log_window)

        self.recent_wins: Deque[int] = deque(maxlen=CURRICULUM_WINDOW)
        self.difficulty_level = 0
        self.env.set_difficulty(self.difficulty_level)
        
        # Анти-эксплойт: минимальное число эпизодов на уровне перед переходом
        self.episodes_at_current_level = 0
        self.MIN_EPISODES_PER_LEVEL = 20
        
        # Анти-эксплойт: отслеживание ранних смертей
        self.early_deaths: Deque[int] = deque(maxlen=100)
        self.EARLY_DEATH_THRESHOLD = 100  # шагов
        
        # Анти-эксплойт: отслеживание прогресса по блокам
        self.bricks_destroyed_history: Deque[int] = deque(maxlen=20)
        
        # Статистика
        self.episode_lengths: Deque[int] = deque(maxlen=100)
        self.bricks_destroyed_per_episode: Deque[int] = deque(maxlen=100)
        
        # Флаги обнаруженных эксплойтов
        self.detected_exploit_flags = {
            "early_death": 0,
            "no_progress": 0,
            "stagnation": 0
        }

        self._stop_event = threading.Event()
        
        # Для измерения скорости обучения
        self.steps_counter = 0
        self.last_steps_log_time = time.time()

    def stop(self) -> None:
        self._stop_event.set()

    def _epsilon(self, global_step: int) -> float:
        if global_step >= self.epsilon_decay_steps:
            return self.epsilon_final
        t = global_step / float(self.epsilon_decay_steps)
        return self.epsilon_start + t * (self.epsilon_final - self.epsilon_start)
    
    def _calculate_no_progress_penalty(self, bricks_destroyed: int, episode_length: int) -> float:
        """Штраф за отсутствие прогресса (анти-эксплойт)"""
        if episode_length > 500 and bricks_destroyed == 0:
            # Долго играет, но не ломает блоки
            self.detected_exploit_flags["no_progress"] += 1
            return -0.01 * (episode_length / 100)  # -0.01 за каждые 100 шагов без прогресса
        return 0.0
    
    def _calculate_early_death_penalty(self, episode_length: int, total_reward: float) -> float:
        """Дополнительный штраф за раннюю смерть (анти-эксплойт)"""
        if episode_length < self.EARLY_DEATH_THRESHOLD:
            self.early_deaths.append(1)
            self.detected_exploit_flags["early_death"] += 1
            
            # Более сильный штраф за очень раннюю смерть
            if episode_length < 50:
                return -5.0
            elif episode_length < 100:
                return -2.0
            else:
                return -1.0
        return 0.0

    def run(self) -> None:
        print(f"[TRAIN] Starting trainer thread on {self.agent.device.type.upper()}")
        
        last_snapshot_t = time.time()
        last_save_t = time.time()

        episode = 0
        while not self._stop_event.is_set():
            episode += 1
            self.episodes_at_current_level += 1
            
            obs = self.env.reset()

            total_reward = 0.0
            steps = 0
            done = False
            last_info = {}
            bricks_destroyed = 0

            episode_start_time = time.time()

            while not done and steps < self.max_episode_steps and not self._stop_event.is_set():
                eps = self._epsilon(self.agent.global_step)
                action = self.agent.select_action(obs, eps)

                next_obs, reward, done, info = self.env.step(action)
                
                # Считаем уничтоженные блоки
                if info.get("broke_brick", False):
                    bricks_destroyed += 1

                # Анти-эксплойт: добавляем штрафы за плохое поведение
                exploit_penalty = 0.0
                
                if done and steps < self.max_episode_steps:
                    # Игра закончилась (не по таймауту)
                    early_death_penalty = self._calculate_early_death_penalty(steps, total_reward)
                    exploit_penalty += early_death_penalty
                
                # Штраф за отсутствие прогресса (считается каждый шаг)
                if steps % 100 == 0 and bricks_destroyed == 0:
                    no_progress_penalty = self._calculate_no_progress_penalty(bricks_destroyed, steps)
                    exploit_penalty += no_progress_penalty
                
                # Применяем штраф
                reward += exploit_penalty

                self.agent.add_transition(
                    Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
                )

                loss = self.agent.train_step()

                obs = next_obs
                total_reward += reward
                steps += 1
                self.steps_counter += 1
                last_info = info

                now = time.time()

                if now - last_snapshot_t >= self.snapshot_interval_sec:
                    self.agent.update_snapshot()
                    snap = self.agent.get_snapshot()
                    if snap is not None:
                        self.render_policy.set_snapshot(snap)
                    last_snapshot_t = now

                if now - last_save_t >= self.save_interval_sec:
                    self.agent.save(self.agent.model_path)
                    last_save_t = now

                # Логирование скорости каждые 1000 шагов
                if self.steps_counter % 1000 == 0:
                    current_time = time.time()
                    time_diff = current_time - self.last_steps_log_time
                    if time_diff > 0:
                        steps_per_sec = 1000 / time_diff
                        device_type = self.agent.device.type.upper()
                        print(f"[TRAIN] speed = {steps_per_sec:.1f} steps/sec ({device_type})")
                        self.last_steps_log_time = current_time

            episode_time = time.time() - episode_start_time

            # Сохраняем статистику эпизода
            self.episode_lengths.append(steps)
            self.bricks_destroyed_per_episode.append(bricks_destroyed)
            self.bricks_destroyed_history.append(bricks_destroyed)
            self.recent_rewards.append(total_reward)
            
            # Вычисляем средние значения
            avg_r = sum(self.recent_rewards) / float(len(self.recent_rewards)) if len(self.recent_rewards) > 0 else 0.0
            avg_episode_length = sum(self.episode_lengths) / float(len(self.episode_lengths)) if len(self.episode_lengths) > 0 else 0.0
            avg_bricks_destroyed = sum(self.bricks_destroyed_per_episode) / float(len(self.bricks_destroyed_per_episode)) if len(self.bricks_destroyed_per_episode) > 0 else 0.0
            
            # Статистика ранних смертей
            early_death_rate = sum(self.early_deaths) / float(len(self.early_deaths)) if len(self.early_deaths) > 0 else 0.0

            bricks_left = int(last_info.get("bricks_left", -1))
            win = bool(last_info.get("win", False))
            lose = bool(last_info.get("lose", False))
            self.recent_wins.append(1 if win else 0)
            win_rate = sum(self.recent_wins) / float(len(self.recent_wins)) if len(self.recent_wins) > 0 else 0.0

            # Анти-эксплойт: минимальное число эпизодов на уровне перед переходом
            can_level_up = (
                win_rate >= CURRICULUM_WIN_RATE_THRESHOLD and 
                self.difficulty_level < (len(CURRICULUM_LEVELS) - 1) and
                self.episodes_at_current_level >= self.MIN_EPISODES_PER_LEVEL
            )
            
            if can_level_up:
                self.difficulty_level += 1
                self.env.set_difficulty(self.difficulty_level)
                self.episodes_at_current_level = 0
                print(f"[TRAIN] LEVEL UP! New difficulty: {self.difficulty_level}")

            eps = self._epsilon(self.agent.global_step)

            paddle_width, ball_speed_mult, brick_rows, brick_cols, a_min, a_max = CURRICULUM_LEVELS[self.difficulty_level]

            # Детализированное логирование с анти-эксплойт информацией
            device_type = self.agent.device.type.upper()
            print(
                f"[TRAIN] ep={episode} "
                f"reward={total_reward:.2f} "
                f"avg{self.log_window}_r={avg_r:.2f} "
                f"bricks_left={bricks_left} "
                f"win={int(win)} lose={int(lose)} "
                f"eps={eps:.3f} "
                f"step={self.agent.global_step} "
                f"level={self.difficulty_level} "
                f"ep_at_level={self.episodes_at_current_level}/{self.MIN_EPISODES_PER_LEVEL} "
                f"win_rate{CURRICULUM_WINDOW}={win_rate:.2f} "
                f"avg_len={avg_episode_length:.1f} "
                f"avg_bricks={avg_bricks_destroyed:.1f} "
                f"early_death={early_death_rate:.2f} "
                f"exploits=[ED:{self.detected_exploit_flags['early_death']} "
                f"NP:{self.detected_exploit_flags['no_progress']}] "
                f"time={episode_time:.1f}s "
                f"steps/sec={steps/episode_time:.1f} "
                f"{device_type}"
            )
            
            # Периодический вывод подробной статистики
            if episode % 100 == 0:
                print(f"[STATS] Эпизод {episode} сводка:")
                print(f"  Средняя длина эпизода: {avg_episode_length:.1f} шагов")
                print(f"  Среднее сломанных блоков: {avg_bricks_destroyed:.1f}")
                print(f"  Ранних смертей: {early_death_rate:.2%}")
                print(f"  Уровень сложности: {self.difficulty_level}")
                print(f"  Эксплойты обнаружены: {self.detected_exploit_flags}")