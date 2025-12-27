from __future__ import annotations

import math
import random
from typing import Callable, List, Optional

import arcade

from config import (
    BACKGROUND_COLOR,
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
    CRATE_TEXTURE,
    DOOR_TEXTURE,
    END_FONT_SIZE,
    FLOOR_TEXTURE,
    PADDLE_COLOR,
    PADDLE_HEIGHT,
    PADDLE_SPEED,
    PADDLE_WIDTH,
    PADDLE_Y_OFFSET,
    STARTING_AMMO,
    STARTING_HEALTH,
    STATUS_FONT_SIZE,
    TEXT_COLOR,
    TILE_SIZE,
    WALL_TEXTURE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from game.entities import Ball, Brick, Paddle
from game.input_handler import InputHandler
from game.physics import reflect_ball_from_brick, reflect_ball_from_paddle, reflect_ball_from_walls


class GameWindow(arcade.Window):
    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        auto_restart_visual: bool = False,
    ) -> None:
        super().__init__(width, height, title, update_rate=1 / 60)
        arcade.set_background_color(BACKGROUND_COLOR)

        self.input_handler = InputHandler()

        self.paddle: Optional[Paddle] = None
        self.ball: Optional[Ball] = None
        self.bricks: List[Brick] = []

        self.is_game_over = False
        self.is_win = False
        self.visual_episode = 0
        self.bricks_destroyed_in_episode = 0

        self.last_hit_paddle = False
        self.last_broke_brick = False
        self.last_reward = 0.0

        # Для auto-restart в режиме с action_provider
        self.action_provider: Optional[Callable[[GameWindow], int]] = None
        self.auto_restart_visual = auto_restart_visual
        self.steps_in_current_episode = 0
        self.total_bricks_destroyed = 0
        self.total_bricks = 0
        self.world = arcade.Scene()
        self.camera = arcade.Camera(self.width, self.height)
        self.hud_camera = arcade.Camera(self.width, self.height)
        self.health = STARTING_HEALTH
        self.ammo = STARTING_AMMO

    def setup(self) -> None:
        self.is_game_over = False
        self.is_win = False
        self.steps_in_current_episode = 0
        self.bricks_destroyed_in_episode = 0

        self.last_hit_paddle = False
        self.last_broke_brick = False
        self.last_reward = 0.0

        self.paddle = Paddle(
            x=WORLD_WIDTH / 2,
            y=PADDLE_Y_OFFSET,
            width=PADDLE_WIDTH,
            height=PADDLE_HEIGHT,
            speed=PADDLE_SPEED,
            color=PADDLE_COLOR,
        )

        # Стартовая скорость: вверх под небольшим углом
        angle = random.uniform(math.radians(35), math.radians(145))
        dx = math.cos(angle)
        dy = math.sin(angle)
        if dy <= 0:
            dy = abs(dy) + 0.2

        self.ball = Ball(
            x=WORLD_WIDTH / 2,
            y=self.paddle.top + BALL_RADIUS + 6,
            radius=BALL_RADIUS,
            dx=dx,
            dy=dy,
            speed=BALL_SPEED,
            color=BALL_COLOR,
        )
        self.ball.normalize_velocity()

        self.bricks = self._build_bricks()
        self.total_bricks = len(self.bricks)
        self._build_world()
        self.health = STARTING_HEALTH
        self.ammo = STARTING_AMMO

        if self.action_provider is not None:
            self.visual_episode += 1
            print(f"[VISUAL] episode={self.visual_episode} started")

    def _build_bricks(self) -> List[Brick]:
        bricks: List[Brick] = []

        available_width = WORLD_WIDTH - (2 * BRICK_SIDE_MARGIN)
        total_gap = (BRICK_COLS - 1) * BRICK_GAP
        brick_width = (available_width - total_gap) / BRICK_COLS

        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                x = BRICK_SIDE_MARGIN + brick_width / 2 + col * (brick_width + BRICK_GAP)
                y = WORLD_HEIGHT - BRICK_TOP_MARGIN - row * (BRICK_HEIGHT + BRICK_GAP)
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

    def _build_world(self) -> None:
        self.world = arcade.Scene()
        floor_list = arcade.SpriteList()
        wall_list = arcade.SpriteList()
        prop_list = arcade.SpriteList()

        tiles_x = math.ceil(WORLD_WIDTH / TILE_SIZE)
        tiles_y = math.ceil(WORLD_HEIGHT / TILE_SIZE)
        for col in range(tiles_x):
            for row in range(tiles_y):
                floor = arcade.Sprite(FLOOR_TEXTURE, scale=1.0)
                floor.center_x = col * TILE_SIZE + TILE_SIZE / 2
                floor.center_y = row * TILE_SIZE + TILE_SIZE / 2
                floor_list.append(floor)

        wall_positions = [
            (TILE_SIZE * 1.5, WORLD_HEIGHT * 0.75),
            (WORLD_WIDTH - TILE_SIZE * 1.5, WORLD_HEIGHT * 0.65),
            (WORLD_WIDTH * 0.5, WORLD_HEIGHT - TILE_SIZE * 1.5),
        ]
        for x, y in wall_positions:
            wall = arcade.Sprite(WALL_TEXTURE, scale=0.8)
            wall.center_x = x
            wall.center_y = y
            wall_list.append(wall)

        crate_positions = [
            (WORLD_WIDTH * 0.25, WORLD_HEIGHT * 0.35),
            (WORLD_WIDTH * 0.75, WORLD_HEIGHT * 0.4),
            (WORLD_WIDTH * 0.6, WORLD_HEIGHT * 0.2),
        ]
        for x, y in crate_positions:
            crate = arcade.Sprite(CRATE_TEXTURE, scale=0.7)
            crate.center_x = x
            crate.center_y = y
            prop_list.append(crate)

        door = arcade.Sprite(DOOR_TEXTURE, scale=0.7)
        door.center_x = WORLD_WIDTH * 0.5
        door.center_y = WORLD_HEIGHT * 0.15
        prop_list.append(door)

        self.world.add_sprite_list("floor", sprite_list=floor_list)
        self.world.add_sprite_list("walls", sprite_list=wall_list)
        self.world.add_sprite_list("props", sprite_list=prop_list)

    def on_draw(self) -> None:
        self.clear()

        assert self.paddle is not None
        assert self.ball is not None

        self.camera.use()
        self.world.draw()

        # Платформа
        arcade.draw_lrbt_rectangle_filled(
            self.paddle.left, self.paddle.right, self.paddle.bottom, self.paddle.top, self.paddle.color
        )

        # Шар
        arcade.draw_circle_filled(self.ball.x, self.ball.y, self.ball.radius, self.ball.color)

        # Блоки
        for b in self.bricks:
            arcade.draw_lrbt_rectangle_filled(b.left, b.right, b.bottom, b.top, b.color)

        self.hud_camera.use()
        # Статус
        bricks_left = len(self.bricks)
        arcade.draw_text(
            f"Блоков осталось: {bricks_left}",
            12,
            self.height - 28,
            TEXT_COLOR,
            font_size=STATUS_FONT_SIZE,
        )

        arcade.draw_text(
            f"Эпизод: {self.visual_episode}",
            12,
            self.height - 58,
            TEXT_COLOR,
            font_size=STATUS_FONT_SIZE,
        )

        arcade.draw_text(
            f"Здоровье: {self.health}",
            self.width - 12,
            self.height - 28,
            TEXT_COLOR,
            font_size=STATUS_FONT_SIZE,
            anchor_x="right",
        )

        arcade.draw_text(
            f"Патроны: {self.ammo}",
            self.width - 12,
            self.height - 58,
            TEXT_COLOR,
            font_size=STATUS_FONT_SIZE,
            anchor_x="right",
        )

        if self.is_game_over:
            msg = "ПОБЕДА!" if self.is_win else "ПОРАЖЕНИЕ"
            arcade.draw_text(
                msg,
                self.width / 2,
                self.height / 2 + 20,
                TEXT_COLOR,
                font_size=END_FONT_SIZE,
                anchor_x="center",
                anchor_y="center",
            )
            
            arcade.draw_text(
                "Нажми R чтобы перезапустить",
                self.width / 2,
                self.height / 2 - 30,
                TEXT_COLOR,
                font_size=STATUS_FONT_SIZE,
                anchor_x="center",
                anchor_y="center",
            )

    def on_update(self, delta_time: float) -> None:
        if self.paddle is None or self.ball is None:
            return

        if self.is_game_over:
            if self.auto_restart_visual:
                self.setup()
            return

        self.steps_in_current_episode += 1
        self.last_hit_paddle = False
        self.last_broke_brick = False
        self.last_reward = 0.0

        if self.action_provider is not None:
            action = self.action_provider(self)
            if action == 1:
                self.input_handler.left_pressed = True
                self.input_handler.right_pressed = False
            elif action == 2:
                self.input_handler.left_pressed = False
                self.input_handler.right_pressed = True
            else:
                self.input_handler.left_pressed = False
                self.input_handler.right_pressed = False

        # Движение платформы
        move_dir = 0.0
        if self.input_handler.left_pressed and not self.input_handler.right_pressed:
            move_dir = -1.0
        elif self.input_handler.right_pressed and not self.input_handler.left_pressed:
            move_dir = 1.0

        self.paddle.x += move_dir * self.paddle.speed * delta_time
        self.paddle.clamp_to_screen(WORLD_WIDTH)

        # Движение шара
        self.ball.x += self.ball.dx * self.ball.speed * delta_time
        self.ball.y += self.ball.dy * self.ball.speed * delta_time

        reflect_ball_from_walls(self.ball, WORLD_WIDTH, WORLD_HEIGHT)
        self.last_hit_paddle = reflect_ball_from_paddle(self.ball, self.paddle)

        # Столкновения с блоками: за кадр ломаем максимум 1 блок (стабильнее)
        hit_index = None
        for i, brick in enumerate(self.bricks):
            if reflect_ball_from_brick(self.ball, brick):
                hit_index = i
                break
        if hit_index is not None:
            self.bricks.pop(hit_index)
            self.last_broke_brick = True
            self.bricks_destroyed_in_episode += 1
            self.total_bricks_destroyed += 1

        # Условия окончания
        if self.ball.y + self.ball.radius < 0:
            self.is_game_over = True
            self.is_win = False

        if len(self.bricks) == 0:
            self.is_game_over = True
            self.is_win = True

        self._center_camera_to_player()

    def _center_camera_to_player(self) -> None:
        if self.paddle is None:
            return
        screen_center_x = self.paddle.x - self.camera.viewport_width / 2
        screen_center_y = self.paddle.y - self.camera.viewport_height / 2
        screen_center_x = max(0, min(screen_center_x, WORLD_WIDTH - self.camera.viewport_width))
        screen_center_y = max(0, min(screen_center_y, WORLD_HEIGHT - self.camera.viewport_height))
        self.camera.move_to((screen_center_x, screen_center_y), 0.1)

    def on_resize(self, width: float, height: float) -> None:
        super().on_resize(width, height)
        self.camera.resize(int(width), int(height))
        self.hud_camera.resize(int(width), int(height))

    def on_key_press(self, key: int, modifiers: int) -> None:
        if self.is_game_over and key == arcade.key.R:
            self.setup()
            return

        self.input_handler.on_key_press(key, modifiers)

    def on_key_release(self, key: int, modifiers: int) -> None:
        self.input_handler.on_key_release(key, modifiers)
