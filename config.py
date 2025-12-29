# config.py
from typing import List

# ========== 检测参数 ==========
MIN_AREA = 150
DEFAULT_GMM_VAR_THRESHOLD = 15
DEFAULT_FRAME_DIFF_THRESHOLD = 30
DEFAULT_CHANGE_THRESHOLD = 0.05  # 5%
DEFAULT_MIN_INTERVAL = 1.0      # 1秒
TARGET_HEIGHT = 480
PREVIEW_UPDATE_INTERVAL = 0.3   # 秒
GMM_PREHEAT_FRAMES = 10

# ========== UI 参数 ==========
SPEED_LEVELS: List[int] = [1, 2, 4, 8, 16, 24, 32, 64]
