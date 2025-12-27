import argparse
import os
import threading
import time

import arcade
import torch

from config import WINDOW_HEIGHT, WINDOW_TITLE, WINDOW_WIDTH
from game.game_state import GameWindow
from game.rl_env import BreakoutEnv
from game.scripted_policy import ScriptedPolicy
from rl.dqn_agent import DQNAgent
from rl.trainer import TrainerThread
from rl.render_policy import RenderPolicy
from rl.evaluator import EvaluatorThread


def main() -> None:
    # Определяем устройство один раз
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] using {device.type.upper()}")
    
    if device.type == "cuda":
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("=" * 60)
        print("ПРЕДУПРЕЖДЕНИЕ: CUDA недоступна, используется CPU")
        print("Обучение будет очень медленным!")
        print("Для GPU установите: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["play", "rl", "train", "watch", "eval"], default="play")
    parser.add_argument("--model-path", default="dqn_policy.pt")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of episodes for evaluation")
    parser.add_argument("--eval-log-interval", type=int, default=10, help="Log interval for evaluation")
    args = parser.parse_args()

    window = GameWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    window.setup()

    if args.mode in ["play", "rl", "watch"]:
        window.auto_restart_visual = True

    if args.mode == "rl":
        env = BreakoutEnv(window)
        policy = ScriptedPolicy()

        def action_provider(w: GameWindow) -> int:
            obs = env.get_obs()
            return policy.act(obs)

        window.action_provider = action_provider

    if args.mode == "watch":
        render_policy = RenderPolicy(model_path=args.model_path)

        def action_provider(w: GameWindow) -> int:
            obs = BreakoutEnv(w).get_obs()
            return render_policy.act(obs)

        window.action_provider = action_provider

    if args.mode == "train":
        print(f"[TRAIN] Starting training on {device.type.upper()}")
        
        agent = DQNAgent(
            obs_dim=7,
            n_actions=3,
            model_path=args.model_path,
        )

        render_policy = RenderPolicy(model_path=args.model_path)

        trainer = TrainerThread(agent=agent, render_policy=render_policy)
        trainer.start()

        def action_provider(w: GameWindow) -> int:
            obs = BreakoutEnv(w).get_obs()
            return render_policy.act(obs)

        window.action_provider = action_provider

    if args.mode == "eval":
        render_policy = RenderPolicy(model_path=args.model_path)
        
        evaluator = EvaluatorThread(
            model_path=args.model_path,
            n_episodes=args.eval_episodes,
            log_interval=args.eval_log_interval
        )
        evaluator.start()

        def action_provider(w: GameWindow) -> int:
            obs = BreakoutEnv(w).get_obs()
            return render_policy.act(obs)

        window.action_provider = action_provider

    arcade.run()


if __name__ == "__main__":
    main()
