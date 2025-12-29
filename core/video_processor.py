# core/video_processor.py
import cv2
import numpy as np
from typing import Optional, Tuple
from config import MIN_AREA, TARGET_HEIGHT, GMM_PREHEAT_FRAMES


class VideoProcessor:
    def __init__(self, gmm_var: int, fd_var: int, roi_mask: Optional[np.ndarray] = None):
        self.gmm_var = gmm_var
        self.fd_var = fd_var
        self.roi_mask = roi_mask
        self.reset()

    def reset(self):
        """重置内部状态"""
        self.gmm = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=self.gmm_var,
            detectShadows=False
        )
        self.prev_gray = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._preheated = False
        self._cached_roi_mask = None

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """调整分辨率 + 灰度 + ROI"""
        h, w = frame.shape[:2]
        if TARGET_HEIGHT > 0 and h > TARGET_HEIGHT:
            scale = TARGET_HEIGHT / h
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.roi_mask is not None:
            if self._cached_roi_mask is None or self._cached_roi_mask.shape[:2] != gray.shape[:2]:
                self._cached_roi_mask = cv2.resize(self.roi_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            gray = cv2.bitwise_and(gray, gray, mask=self._cached_roi_mask)

        return frame, gray

    def preheat(self, gray: np.ndarray):
        """用前 N 帧预热 GMM（不检测）"""
        if self._preheated:
            return
        for _ in range(GMM_PREHEAT_FRAMES):
            self.gmm.apply(gray)
        self._preheated = True

    def detect_change(self, gray: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        """检测变化（需在 preheat 后调用）"""
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return False, np.zeros_like(gray), 0.0

        # GMM
        gmm_mask = self.gmm.apply(gray)
        _, gmm_mask = cv2.threshold(gmm_mask, 254, 255, cv2.THRESH_BINARY)

        # 帧差
        frame_diff = cv2.absdiff(gray, self.prev_gray)
        _, diff_mask = cv2.threshold(frame_diff, self.fd_var, 255, cv2.THRESH_BINARY)

        # 融合 + 形态学
        fg_mask = cv2.bitwise_and(gmm_mask, diff_mask) 
        #fg_mask = cv2.bitwise_or(gmm_mask, diff_mask) #改用 OR 融合策略（提升灵敏度）但会能引入更多噪点 → 有必要可通过增大 MIN_AREA 或形态学来抑制
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        # 判断有效变化
        valid_change = False
        if cv2.countNonZero(fg_mask) > 0:
            _, _, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
            for i in range(1, len(stats)):
                if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
                    valid_change = True
                    break

        self.prev_gray = gray.copy()

        # 变化比例
        total = cv2.countNonZero(self._cached_roi_mask) if self._cached_roi_mask is not None else gray.size
        change_pixels = cv2.countNonZero(fg_mask)
        ratio = change_pixels / total if total > 0 else 0.0

        return valid_change, fg_mask, ratio
    
    