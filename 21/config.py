"""
Конфигурационные константы для игры Breakout
"""

# Окно
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Breakout RL"
BACKGROUND_COLOR = (10, 20, 30)  # Темно-синий

# Мир и текстуры
WORLD_WIDTH = 1600
WORLD_HEIGHT = 1200
TILE_SIZE = 128
FLOOR_TEXTURE = ":resources:images/tiles/grassCenter.png"
WALL_TEXTURE = ":resources:images/tiles/brickGrey.png"
CRATE_TEXTURE = ":resources:images/tiles/boxCrate_double.png"
DOOR_TEXTURE = ":resources:images/tiles/doorClosed_mid.png"

# HUD
STARTING_HEALTH = 100
STARTING_AMMO = 30

# Платформа
PADDLE_WIDTH = 120
PADDLE_HEIGHT = 20
PADDLE_SPEED = 500
PADDLE_COLOR = (100, 200, 255)
PADDLE_Y_OFFSET = 30

# Мяч
BALL_RADIUS = 10
BALL_SPEED = 400
BALL_COLOR = (255, 100, 100)

# Блоки
BRICK_ROWS = 6
BRICK_COLS = 10
BRICK_WIDTH = 70
BRICK_HEIGHT = 25
BRICK_GAP = 4
BRICK_COLOR = (100, 255, 150)
BRICK_SIDE_MARGIN = 50
BRICK_TOP_MARGIN = 60

# Текст
TEXT_COLOR = (255, 255, 255)
STATUS_FONT_SIZE = 18
END_FONT_SIZE = 48

# Награды RL
R_HIT_PADDLE = 0.05
R_BREAK_BRICK = 1.0
R_WIN = 2.0
R_LOSE = -2.0
R_TIMEOUT = -0.5

# Curriculum learning
CURRICULUM_LEVELS = [
    # (paddle_width, ball_speed_mult, brick_rows, brick_cols, angle_min, angle_max)
    (120, 1.0, 6, 10, 35, 145),
    (100, 1.2, 6, 10, 30, 150),
    (80, 1.5, 7, 10, 25, 155),
    (70, 1.8, 7, 12, 20, 160),
    (60, 2.0, 8, 12, 15, 165),
]

CURRICULUM_WINDOW = 20
CURRICULUM_WIN_RATE_THRESHOLD = 0.8

# Анти-эксплойт константы
MAX_EPISODE_STEPS = 6000  # Лимит шагов на эпизод
EARLY_DEATH_PENALTY_MULTIPLIER = 2.0  # Множитель штрафа за раннюю смерть
NO_PROGRESS_PENALTY_INTERVAL = 100  # Интервал проверки прогресса (шаги)
MIN_EPISODES_PER_LEVEL = 20  # Минимальное число эпизодов на уровне curriculum

# Рандомизация стартового угла
START_ANGLE_VARIATION_DEG = 10  # ±10 градусов вариации
MIN_START_ANGLE_DEG = 20  # Минимальный стартовый угол
MAX_START_ANGLE_DEG = 160  # Максимальный стартовый угол
