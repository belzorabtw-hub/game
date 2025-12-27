from __future__ import annotations

import arcade


class InputHandler:
    def __init__(self) -> None:
        self.left_pressed = False
        self.right_pressed = False

    def on_key_press(self, key: int, modifiers: int) -> None:
        if key in (arcade.key.LEFT, arcade.key.A):
            self.left_pressed = True
        elif key in (arcade.key.RIGHT, arcade.key.D):
            self.right_pressed = True

    def on_key_release(self, key: int, modifiers: int) -> None:
        if key in (arcade.key.LEFT, arcade.key.A):
            self.left_pressed = False
        elif key in (arcade.key.RIGHT, arcade.key.D):
            self.right_pressed = False
