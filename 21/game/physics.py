from __future__ import annotations

from typing import Optional, Tuple

from game.entities import Ball, Brick, Paddle


def clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def circle_intersects_rect(
    cx: float, cy: float, r: float,
    rx1: float, ry1: float, rx2: float, ry2: float,
) -> bool:
    # Ближайшая точка прямоугольника к центру окружности
    closest_x = clamp(cx, rx1, rx2)
    closest_y = clamp(cy, ry1, ry2)
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx * dx + dy * dy) <= (r * r)


def reflect_ball_from_walls(ball: Ball, screen_width: float, screen_height: float) -> None:
    # Лево/право
    if ball.x - ball.radius <= 0:
        ball.x = ball.radius
        ball.dx *= -1
    if ball.x + ball.radius >= screen_width:
        ball.x = screen_width - ball.radius
        ball.dx *= -1

    # Верх
    if ball.y + ball.radius >= screen_height:
        ball.y = screen_height - ball.radius
        ball.dy *= -1


def reflect_ball_from_paddle(ball: Ball, paddle: Paddle) -> bool:
    # Отражаем только если шар летит вниз и пересёк верх платформы
    if ball.dy >= 0:
        return False

    rx1 = paddle.left
    rx2 = paddle.right
    ry1 = paddle.bottom
    ry2 = paddle.top

    if not circle_intersects_rect(ball.x, ball.y, ball.radius, rx1, ry1, rx2, ry2):
        return False

    # Перемещаем шар над платформой и задаём новое направление в зависимости от точки удара
    ball.y = paddle.top + ball.radius

    hit = (ball.x - paddle.x) / (paddle.width / 2)  # [-1..1]
    hit = clamp(hit, -1.0, 1.0)

    # Базовое отражение вверх + добавка по x от позиции удара
    # Чем ближе к краю, тем сильнее горизонтальная компонента.
    ball.dx = hit
    ball.dy = 1.0
    ball.normalize_velocity()
    return True


def reflect_ball_from_brick(ball: Ball, brick: Brick) -> bool:
    rx1 = brick.left
    rx2 = brick.right
    ry1 = brick.bottom
    ry2 = brick.top

    if not circle_intersects_rect(ball.x, ball.y, ball.radius, rx1, ry1, rx2, ry2):
        return False

    # Вычисляем глубины проникновения по осям, чтобы понять, от какой стороны отражать.
    # Считаем "возможное" пересечение как если бы шар был точкой, но с расширением прямоугольника на радиус.
    expanded_left = rx1 - ball.radius
    expanded_right = rx2 + ball.radius
    expanded_bottom = ry1 - ball.radius
    expanded_top = ry2 + ball.radius

    # Если центр внутри расширенного прямоугольника - определяем ближайшую сторону.
    # Это стабильно для простого арканоида.
    if expanded_left <= ball.x <= expanded_right and expanded_bottom <= ball.y <= expanded_top:
        dist_left = abs(ball.x - expanded_left)
        dist_right = abs(expanded_right - ball.x)
        dist_bottom = abs(ball.y - expanded_bottom)
        dist_top = abs(expanded_top - ball.y)

        m = min(dist_left, dist_right, dist_bottom, dist_top)

        if m == dist_left:
            ball.x = expanded_left
            ball.dx *= -1
        elif m == dist_right:
            ball.x = expanded_right
            ball.dx *= -1
        elif m == dist_bottom:
            ball.y = expanded_bottom
            ball.dy *= -1
        else:
            ball.y = expanded_top
            ball.dy *= -1
    else:
        # Фоллбек: просто инвертируем dy
        ball.dy *= -1

    return True
