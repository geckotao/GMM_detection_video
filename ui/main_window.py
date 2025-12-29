# ui/main_window.py
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
import time
from datetime import datetime
from PIL import Image, ImageTk
import queue
import ctypes
from typing import Optional, Tuple, List

from config import (
    DEFAULT_CHANGE_THRESHOLD,
    DEFAULT_MIN_INTERVAL,
    TARGET_HEIGHT,
    PREVIEW_UPDATE_INTERVAL,
    SPEED_LEVELS,
    DEFAULT_GMM_VAR_THRESHOLD,
    DEFAULT_FRAME_DIFF_THRESHOLD,
    GMM_PREHEAT_FRAMES
)
from core.video_processor import VideoProcessor


class GMMVideoDetector:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.log_file_path = None
        self.ui_queue = queue.Queue()
        self.root.title("视频画面变化检测v1.3 by geckotao")
        self.dpi_scale = self.get_dpi_scale()
        self.base_font_size = 10
        self.scaled_font_size = int(self.base_font_size * self.dpi_scale)
        self.root.geometry(f"{int(1200 * self.dpi_scale)}x{int(750 * self.dpi_scale)}")
        self.root.minsize(int(1024 * self.dpi_scale), int(750 * self.dpi_scale))
        self.root.configure(bg="#f0f0f0")
        self.setup_dpi_awareness()
        self.background_mode_var = tk.BooleanVar(value=False)

        # 保存控件引用
        self.control_btn = None
        self.stop_btn = None
        self.video_listbox = None
        self.video_label = None

        # 分离控件组（关键优化）
        self.parameter_widgets = []
        self.file_widgets = []

        # UI 样式
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=("SimHei", self.scaled_font_size))
        self.style.configure("TButton", font=("SimHei", self.scaled_font_size), padding=int(5 * self.dpi_scale))
        self.style.configure("TEntry", font=("SimHei", self.scaled_font_size), padding=int(3 * self.dpi_scale))
        self.style.configure("TNotebook.Tab", font=("SimHei", self.scaled_font_size),
                             padding=(int(10 * self.dpi_scale), int(5 * self.dpi_scale)))
        self.style.configure("TLabelframe", background="#f0f0f0", borderwidth=1)
        self.style.configure("TLabelframe.Label", background="#f0f0f0",
                             font=("SimHei", self.scaled_font_size, "bold"),
                             padding=(int(5 * self.dpi_scale), int(2 * self.dpi_scale)))
        self.style.configure("Accent.TButton", font=("SimHei", self.scaled_font_size, "bold"),
                             background="#4a90e2", foreground="white")
        self.style.map("Accent.TButton", background=[("active", "#357abd"), ("pressed", "#2a5f90")])

        # 路径与日志
        self.log_dir = os.path.join(os.getcwd(), "检测日志")
        self.ensure_directory_exists(self.log_dir)
        self.log_file_path = None
        self.init_log_file()

        # 状态变量
        self.video_paths: List[str] = []
        self.current_video_index = 0
        self.processing = False
        self.paused = False
        self.roi_selected = False
        self.roi_points: List[Tuple[float, float]] = []
        self.change_threshold = DEFAULT_CHANGE_THRESHOLD
        self.save_path = os.path.join(os.getcwd(), "变化截图")
        self.ensure_directory_exists(self.save_path)
        self.last_change_time = 0
        self.min_interval = DEFAULT_MIN_INTERVAL
        self.preview_frame: Optional[np.ndarray] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.speed_levels = SPEED_LEVELS
        self.current_speed = 1
        self.gmm_var = tk.IntVar(value = DEFAULT_GMM_VAR_THRESHOLD)
        self.fd_var = tk.IntVar(value = DEFAULT_FRAME_DIFF_THRESHOLD)
        self.target_height = TARGET_HEIGHT
        self.cap: Optional[cv2.VideoCapture] = None

        # 线程安全
        self.ui_queue = queue.Queue()
        self.cap_lock = threading.Lock()

        # UI
        self.root.after(50, self.process_ui_queue)
        self.create_widgets()
        self.setup_icon()

        # 日志初始化信息
        self.log_message("程序启动 (v1.3版)")
        self.log_message(f"保存路径: {self.save_path}")
        self.log_message(f"初始处理倍速: {self.current_speed}倍")
        self.log_message(f"目标处理分辨率: 高度 ≤ {self.target_height}P")
        self.log_message(f"DPI缩放因子: {self.dpi_scale:.2f}")
        self.log_message(f"默认GMM敏感度: {DEFAULT_GMM_VAR_THRESHOLD}")
        self.log_message(f"默认帧差阈值: {DEFAULT_FRAME_DIFF_THRESHOLD}")
    # ==================== DPI 相关 ====================
    def get_dpi_scale(self) -> float:
        try:
            if os.name == 'nt':
                import platform
                win_ver = platform.version()
                if win_ver >= "10.0":
                    ctypes.windll.user32.SetProcessDPIAware()
                    dpi = ctypes.windll.user32.GetDpiForSystem()
                    return dpi / 96.0
                else:
                    ctypes.windll.user32.SetProcessDPIAware()
                    return 1.25 if self.root.winfo_screenwidth() >= 1920 else 1.0
            else:
                return 1.0
        except Exception as e:
            self.log_message(f"DPI 检测异常，使用默认缩放 1.0: {str(e)}")
            return 1.0

    def setup_dpi_awareness(self):
        try:
            if os.name == 'nt':
                import platform
                if platform.version() >= "6.2":
                    try:
                        ctypes.windll.shcore.SetProcessDpiAwareness(1)
                    except:
                        ctypes.windll.user32.SetProcessDPIAware()
                else:
                    ctypes.windll.user32.SetProcessDPIAware()
        except Exception as e:
            self.log_message(f"设置 DPI 感知失败: {str(e)}")

    # ==================== 文件与日志 ====================
    def ensure_directory_exists(self, path: str):
        if os.path.exists(path):
            return
        for attempt in range(3):
            try:
                os.makedirs(path, exist_ok=True)
                self.log_message(f"创建目录: {path}")
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.1)
                else:
                    error_msg = f"创建目录失败 {path}: {str(e)}"
                    print(error_msg)
                    self.safe_ui_call(messagebox.showerror, "错误", error_msg)

    def init_log_file(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"检测日志_{timestamp}.txt"
            self.log_file_path = os.path.join(self.log_dir, log_filename)
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                pass
            self.log_message(f"日志文件初始化成功: {self.log_file_path}")
        except Exception as e:
            error_msg = f"初始化日志文件失败: {str(e)}"
            print(error_msg)
            self.safe_ui_call(messagebox.showerror, "错误", error_msg)
            self.log_file_path = None

    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())
        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
            except Exception as e:
                print(f"写入日志失败: {str(e)}")

    # ==================== UI 相关 ====================
    def setup_icon(self):
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
            if os.path.exists(icon_path):
                icon_img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, icon_img)
                self.log_message(f"成功设置窗口图标: {icon_path}")
        except Exception as e:
            self.log_message(f"设置窗口图标失败: {str(e)}")

    def create_widgets(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        padx = int(8 * self.dpi_scale)
        pady = int(8 * self.dpi_scale)

        # 左侧面板
        left_frame = ttk.Frame(self.root, width=int(365 * self.dpi_scale))
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(padx, padx//2), pady=pady)
        left_frame.grid_propagate(False)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        control_notebook = ttk.Notebook(left_frame)
        control_notebook.grid(row=0, column=0, sticky="nsew", padx=int(5 * self.dpi_scale), pady=int(5 * self.dpi_scale))

        # === 参数设置页 ===
        settings_frame = ttk.Frame(control_notebook)
        control_notebook.add(settings_frame, text="参数设置")
        settings_canvas = tk.Canvas(settings_frame, highlightthickness=0, bg="#f0f0f0")
        settings_scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=settings_canvas.yview)
        settings_scrollable_frame = ttk.Frame(settings_canvas, style="TFrame")
        settings_scrollable_frame.bind(
            "<Configure>",
            lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
        )
        settings_canvas.create_window((0, 0), window=settings_scrollable_frame, anchor="nw")
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        settings_canvas.pack(side="left", fill="both", expand=True)
        settings_scrollbar.pack(side="right", fill="y")

        row = 0
        frame_pady = (int(5 * self.dpi_scale), int(8 * self.dpi_scale))
        inner_pad = int(10 * self.dpi_scale)

        # 视频文件管理
        file_frame = ttk.LabelFrame(settings_scrollable_frame, text="视频文件管理", padding=(inner_pad, int(8 * self.dpi_scale)))
        file_frame.grid(row=row, column=0, sticky="ew", pady=frame_pady, padx=int(5 * self.dpi_scale))
        settings_scrollable_frame.columnconfigure(0, weight=1)
        row += 1

        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=tk.X, pady=(0, int(8 * self.dpi_scale)))
        self.add_btn = ttk.Button(btn_row, text="添加视频文件", command=self.add_videos)
        self.add_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(5 * self.dpi_scale)))
        self.clear_btn = ttk.Button(btn_row, text="清空列表", command=self.clear_videos)
        self.clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        list_frame_inner = ttk.Frame(file_frame)
        list_frame_inner.pack(fill=tk.X, pady=(0, int(8 * self.dpi_scale)))
        self.video_listbox = tk.Listbox(list_frame_inner, height=int(6 * self.dpi_scale), selectmode=tk.EXTENDED, bd=1, relief=tk.SUNKEN,
                                        font=("SimHei", self.scaled_font_size))
        self.video_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll = ttk.Scrollbar(list_frame_inner, orient="vertical", command=self.video_listbox.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_listbox.configure(yscrollcommand=v_scroll.set)

        action_btn_frame = ttk.Frame(file_frame)
        action_btn_frame.pack(fill=tk.X)
        self.remove_btn = ttk.Button(action_btn_frame, text="移除", command=self.remove_video)
        self.remove_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(3 * self.dpi_scale)))
        self.preview_btn = ttk.Button(action_btn_frame, text="预览", command=self.preview_selected_video)
        self.preview_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(int(3 * self.dpi_scale), 0))

        # ROI
        roi_frame = ttk.LabelFrame(settings_scrollable_frame, text="关注区域 (ROI)", padding=(inner_pad, int(8 * self.dpi_scale)))
        roi_frame.grid(row=row, column=0, sticky="ew", pady=frame_pady, padx=int(5 * self.dpi_scale))
        row += 1
        self.roi_button = ttk.Button(roi_frame, text="选择ROI区域", command=self.select_roi)
        self.roi_button.pack(fill=tk.X, pady=(0, int(5 * self.dpi_scale)))
        self.roi_status_label = ttk.Label(roi_frame, text="未选取关注区域", foreground="gray")
        self.roi_status_label.pack(anchor=tk.W, pady=(0, int(2 * self.dpi_scale)))
        ttk.Label(roi_frame, text="操作: 左键加点，右键删点，点【确认选择】选取关注区域。",
                  font=("TkDefaultFont", max(8, int(8 * self.dpi_scale))), foreground="#666666").pack(anchor=tk.W)

        # 检测参数
        param_frame = ttk.LabelFrame(settings_scrollable_frame, text="检测参数", padding=(inner_pad, int(8 * self.dpi_scale)))
        param_frame.grid(row=row, column=0, sticky="ew", pady=frame_pady, padx=int(5 * self.dpi_scale))
        row += 1

        # GMM 敏感度
        ttk.Label(param_frame, text="GMM 敏感度 (varThreshold)").pack(anchor=tk.W, pady=(0, int(2 * self.dpi_scale)))
        gmm_f = ttk.Frame(param_frame)
        gmm_f.pack(fill=tk.X, pady=(0, int(8 * self.dpi_scale)))
        self.gmm_var = tk.IntVar(value=DEFAULT_GMM_VAR_THRESHOLD)
        self.gmm_scale = ttk.Scale(gmm_f, from_=10, to=50, variable=self.gmm_var, command=self.update_gmm_threshold)
        self.gmm_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(5 * self.dpi_scale)))
        self.gmm_label = ttk.Label(gmm_f, text=str(DEFAULT_GMM_VAR_THRESHOLD), width=int(6 * self.dpi_scale))
        self.gmm_label.pack(side=tk.RIGHT)

        # 帧差阈值
        ttk.Label(param_frame, text="帧间差分阈值").pack(anchor=tk.W, pady=(0, int(2 * self.dpi_scale)))
        fd_f = ttk.Frame(param_frame)
        fd_f.pack(fill=tk.X, pady=(0, int(8 * self.dpi_scale)))
        self.fd_var = tk.IntVar(value=DEFAULT_FRAME_DIFF_THRESHOLD)
        self.fd_scale = ttk.Scale(fd_f, from_=5, to=60, variable=self.fd_var, command=self.update_frame_diff_threshold)
        self.fd_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(5 * self.dpi_scale)))
        self.fd_label = ttk.Label(fd_f, text=str(DEFAULT_FRAME_DIFF_THRESHOLD), width=int(6 * self.dpi_scale))
        self.fd_label.pack(side=tk.RIGHT)

        # 画面变化阈值
        ttk.Label(param_frame, text="画面变化阈值 (%)").pack(anchor=tk.W, pady=(0, int(2 * self.dpi_scale)))
        thresh_f = ttk.Frame(param_frame)
        thresh_f.pack(fill=tk.X, pady=(0, int(8 * self.dpi_scale)))
        self.threshold_entry_var = tk.StringVar(value=f"{self.change_threshold * 100:.1f}")
        self.threshold_scale = ttk.Scale(thresh_f, from_=0.1, to=50, value=self.change_threshold*100, command=self.update_threshold)
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(5 * self.dpi_scale)))
        self.threshold_label = ttk.Label(thresh_f, text=f"{self.change_threshold*100:.1f}%", width=int(6 * self.dpi_scale))
        self.threshold_label.pack(side=tk.RIGHT)
        self.threshold_entry = ttk.Entry(
            thresh_f,
            textvariable=self.threshold_entry_var,
            width=int(8 * self.dpi_scale),
            font=("SimHei", self.scaled_font_size)
        )
        self.threshold_entry.pack(side=tk.RIGHT, padx=(int(5 * self.dpi_scale), 0))
        self.threshold_entry.bind("<Return>", self.validate_threshold_input)

        # 截图最小间隔
        ttk.Label(param_frame, text="截图最小间隔 (秒)").pack(anchor=tk.W, pady=(0, int(2 * self.dpi_scale)))
        intv_f = ttk.Frame(param_frame)
        intv_f.pack(fill=tk.X, pady=(0, int(8 * self.dpi_scale)))
        self.interval_entry_var = tk.StringVar(value=f"{self.min_interval:.1f}")
        self.interval_scale = ttk.Scale(intv_f, from_=0.1, to=10, value=self.min_interval, command=self.update_interval)
        self.interval_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(5 * self.dpi_scale)))
        self.interval_label = ttk.Label(intv_f, text=f"{self.min_interval:.1f}s", width=int(6 * self.dpi_scale))
        self.interval_label.pack(side=tk.RIGHT)
        self.interval_entry = ttk.Entry(
            intv_f,
            textvariable=self.interval_entry_var,
            width=int(8 * self.dpi_scale),
            font=("SimHei", self.scaled_font_size)
        )
        self.interval_entry.pack(side=tk.RIGHT, padx=(int(5 * self.dpi_scale), 0))
        self.interval_entry.bind("<Return>", self.validate_interval_input)

        # 截图保存
        save_frame = ttk.LabelFrame(settings_scrollable_frame, text="截图保存", padding=(inner_pad, int(8 * self.dpi_scale)))
        save_frame.grid(row=row, column=0, sticky="ew", pady=frame_pady, padx=int(5 * self.dpi_scale))
        row += 1
        ttk.Label(save_frame, text="保存路径:").pack(anchor=tk.W, pady=(0, int(2 * self.dpi_scale)))
        path_f1 = ttk.Frame(save_frame)
        path_f1.pack(fill=tk.X, pady=(0, int(5 * self.dpi_scale)))
        self.save_path_var = tk.StringVar(value=self.save_path)
        self.save_path_entry = ttk.Entry(
            path_f1,
            textvariable=self.save_path_var,
            font=("SimHei", self.scaled_font_size)
        )
        self.save_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(3 * self.dpi_scale)))
        self.save_path_btn = ttk.Button(path_f1, text="更改", command=self.change_save_path, width=int(6 * self.dpi_scale))
        self.save_path_btn.pack(side=tk.RIGHT)

        # 占位行
        ttk.Label(settings_scrollable_frame, text="").grid(row=row, column=0, pady=int(10 * self.dpi_scale))

        # === 处理控制页 ===
        control_frame_nb = ttk.Frame(control_notebook)
        control_notebook.add(control_frame_nb, text="处理控制")
        control_frame_nb.grid_rowconfigure(4, weight=1)
        control_frame_nb.grid_columnconfigure(0, weight=1)

        status_progress_frame = ttk.LabelFrame(control_frame_nb, text="状态与进度", padding=(inner_pad, int(8 * self.dpi_scale)))
        status_progress_frame.grid(row=0, column=0, sticky="ew", padx=int(5 * self.dpi_scale), pady=(int(5 * self.dpi_scale), int(5 * self.dpi_scale)))
        self.status_var = tk.StringVar(value=f"就绪 | 当前倍速: {self.current_speed}倍")
        status_label = ttk.Label(status_progress_frame, textvariable=self.status_var,
                                 font=("SimHei", self.scaled_font_size, "bold"), foreground="#27ae60")
        status_label.pack(anchor=tk.W, pady=(0, int(5 * self.dpi_scale)))
        self.progress_label = ttk.Label(status_progress_frame, text="等待开始处理...", font=("SimHei", self.scaled_font_size))
        self.progress_label.pack(anchor=tk.W, pady=(0, int(5 * self.dpi_scale)))

        # 总进度 + 实时百分比
        progress_row = ttk.Frame(status_progress_frame)
        progress_row.pack(fill=tk.X, pady=(0, int(2 * self.dpi_scale)))
        ttk.Label(progress_row, text="总进度:").pack(side=tk.LEFT)
        self.total_percent_label = ttk.Label(progress_row, text="0.0%", font=("SimHei", self.scaled_font_size, "bold"), foreground="#0bd300")
        self.total_percent_label.pack(side=tk.LEFT, padx=int(5 * self.dpi_scale))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=int(5 * self.dpi_scale))

        speed_frame = ttk.LabelFrame(control_frame_nb, text="处理速度控制", padding=(inner_pad, int(8 * self.dpi_scale)))
        speed_frame.grid(row=1, column=0, sticky="ew", padx=int(5 * self.dpi_scale), pady=int(5 * self.dpi_scale))
        speed_info_frame = ttk.Frame(speed_frame)
        speed_info_frame.pack(fill=tk.X, pady=(0, int(5 * self.dpi_scale)))
        self.speed_label = ttk.Label(speed_info_frame, text=f"当前: {self.current_speed}x",
                                     font=("SimHei", self.scaled_font_size, "bold"))
        self.speed_label.pack(side=tk.LEFT)

        speed_btn_frame = ttk.Frame(speed_frame)
        speed_btn_frame.pack(fill=tk.X)
        cols = 4
        self.speed_buttons = []
        for i, s in enumerate(self.speed_levels):
            btn = ttk.Button(speed_btn_frame, text=f"{s}x", command=lambda s=s: self.set_speed(s), width=int(5 * self.dpi_scale))
            btn.grid(row=i//cols, column=i%cols, padx=int(2 * self.dpi_scale), pady=int(2 * self.dpi_scale), sticky="ew")
            self.speed_buttons.append(btn)
        for i in range(cols):
            speed_btn_frame.grid_columnconfigure(i, weight=1)

        control_buttons_frame = ttk.LabelFrame(control_frame_nb, text="控制命令", padding=(inner_pad, int(8 * self.dpi_scale)))
        control_buttons_frame.grid(row=2, column=0, sticky="ew", padx=int(5 * self.dpi_scale), pady=int(5 * self.dpi_scale))
        btn_frame = ttk.Frame(control_buttons_frame)
        btn_frame.pack(fill=tk.X, pady=int(5 * self.dpi_scale))
        self.control_btn = ttk.Button(btn_frame, text="开始处理", command=self.on_control_click)
        self.control_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=int(2 * self.dpi_scale))
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop_processing)
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=int(2 * self.dpi_scale))

        # === 后台模式选项 ===
        bg_mode_frame = ttk.LabelFrame(control_frame_nb, text="处理选项", padding=(inner_pad, int(8 * self.dpi_scale)))
        bg_mode_frame.grid(row=3, column=0, sticky="ew", padx=int(5 * self.dpi_scale), pady=int(5 * self.dpi_scale))
        ttk.Checkbutton(
            bg_mode_frame,
            text="后台模式（关闭预览以加速）",
            variable=self.background_mode_var,
            command=self.toggle_background_mode
        ).pack(anchor=tk.W, padx=int(5 * self.dpi_scale))
        self.style.configure(
            "TCheckbutton",
            font=("SimHei", self.scaled_font_size),
            background="#f0f0f0"
        )

        # === 使用帮助页 ===
        help_frame = ttk.Frame(control_notebook)
        control_notebook.add(help_frame, text="使用帮助")
        help_text_widget = tk.Text(
            help_frame,
            wrap=tk.WORD,
            font=("SimHei", self.scaled_font_size),
            bg="#f9f9f9",
            relief=tk.FLAT,
            padx=int(12 * self.dpi_scale),
            pady=int(12 * self.dpi_scale)
        )
        help_scrollbar = ttk.Scrollbar(help_frame, orient="vertical", command=help_text_widget.yview)
        help_text_widget.configure(yscrollcommand=help_scrollbar.set)
        help_content = """视频画面变化检测工具 v1.3
【操作步骤】
① 添加视频
- 点击【参数设置】→【添加视频文件】
- 可多选，文件将显示在列表中
- 支持从列表中移除或清空
② 预览与设置 ROI（可选但推荐）
- 选中视频后点击【预览】
- 点击【选择ROI区域】，在弹出窗口中：
• 左键：添加顶点
• 右键：删除最后一个点
• R 键：重置所有点
• 点【确认选择】：确认 ROI区域（需 ≥3 个点）
- ROI 将以绿色多边形显示在预览画面
③ 配置参数（【参数设置】标签页）
- 【画面变化阈值】：默认 5%。值越小越敏感。
- 【截图最小间隔】：默认 1 秒，避免连续截图。
- 【GMM 敏感度】：默认 25。值越大，越不敏感（可减少“车走后地面误报”）。
- 【帧间差分阈值】：默认 30。值越大，越忽略微小运动（可抑制噪点）。
④ 设置保存路径
- 默认路径：程序目录下的“变化截图”文件夹
- 点击【更改】可自定义路径（需有写入权限）
⑤ 开始处理
- 切换到【处理控制】标签页
- 选择处理倍速（1x 最精准，高倍速会跳帧加速但可能漏检）
- 点击【开始处理】
- 支持【暂停】/【停止】操作
- 支持打开/关闭视频预览
⑥ 查看结果
- 截图自动保存为：`视频名_frame_帧号_时间戳.jpg`
- 日志文件位于“检测日志”文件夹，记录所有操作与错误
如有问题，请查看日志文件或联系开发者（geckotao@hotmail.com）。
"""
        help_text_widget.insert("1.0", help_content)
        help_text_widget.config(state=tk.DISABLED)

        def _on_mousewheel(event):
            if os.name == 'nt':
                help_text_widget.yview_scroll(-int(event.delta / 120), "units")
            else:
                help_text_widget.yview_scroll(-int(event.delta), "units")
        help_text_widget.bind("<MouseWheel>", _on_mousewheel)
        help_text_widget.bind("<Button-4>", lambda e: help_text_widget.yview_scroll(-1, "units"))
        help_text_widget.bind("<Button-5>", lambda e: help_text_widget.yview_scroll(1, "units"))
        help_text_widget.bind("<Button-1>", lambda e: help_text_widget.focus_set())
        help_text_widget.pack(side="left", fill="both", expand=True)
        help_scrollbar.pack(side="right", fill="y")

        # === 右侧预览区 ===
        right_frame = ttk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(padx//2, padx), pady=pady)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        display_frame = ttk.LabelFrame(right_frame, text="视频预览", padding=int(8 * self.dpi_scale))
        display_frame.grid(row=0, column=0, sticky="nsew")
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)

        preview_container = ttk.Frame(display_frame, borderwidth=1, relief=tk.SUNKEN)
        preview_container.grid(row=0, column=0, sticky="nsew", padx=int(2 * self.dpi_scale), pady=int(2 * self.dpi_scale))
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)

        self.video_label = ttk.Label(preview_container, background="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        preview_container.bind("<Configure>", self.on_preview_resize)

        info_bar = ttk.Frame(right_frame, height=int(30 * self.dpi_scale))
        info_bar.grid(row=1, column=0, sticky="ew", pady=(int(5 * self.dpi_scale), 0))
        info_bar.grid_columnconfigure(0, weight=1)
        self.info_label = ttk.Label(info_bar, text="就绪 - 请添加视频文件并开始处理", font=("SimHei", max(9, int(9 * self.dpi_scale))))
        self.info_label.pack(side=tk.LEFT, padx=int(5 * self.dpi_scale))

        # 分离控件组
        self.parameter_widgets = [
            self.roi_button,
            self.gmm_scale, self.gmm_label,
            self.fd_scale, self.fd_label,
            self.threshold_scale, self.threshold_entry, self.threshold_label,
            self.interval_scale, self.interval_entry, self.interval_label,
            self.save_path_entry, self.save_path_btn
        ]
        self.file_widgets = [
            self.add_btn, self.clear_btn, self.remove_btn, self.preview_btn,
            self.video_listbox
        ]

    # ==================== UI 逻辑 ====================
    def process_ui_queue(self):
        while not self.ui_queue.empty():
            try:
                task = self.ui_queue.get_nowait()
                task()
                self.ui_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                self.log_message(f"处理UI任务时出错: {str(e)}")
        if self.root.winfo_exists():
            self.root.after(50, self.process_ui_queue)

    def safe_ui_call(self, func, *args, **kwargs):
        self.ui_queue.put(lambda: func(*args, **kwargs))

    def on_preview_resize(self, event=None):
        if not event or event.width < 50 or event.height < 50:
            return
        frame_to_display = getattr(self, '_last_displayed_frame', None)
        if frame_to_display is not None:
            rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
            self.display_frame(rgb)

    def display_frame(self, frame: np.ndarray):
        if frame is None:
            self.video_label.config(image='', text="无视频预览", foreground="gray")
            return
        self._last_displayed_frame = frame.copy()
        container = self.video_label.master
        container.update_idletasks()
        dw, dh = container.winfo_width(), container.winfo_height()
        if dw < 100 or dh < 100:
            return
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return
        scale = min(dw / w, dh / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if self.roi_selected and self.roi_points:
            orig_h, orig_w = self.preview_frame.shape[:2]
            sx, sy = new_w / orig_w, new_h / orig_h
            pts = np.array([[int(x * sx), int(y * sy)] for x, y in self.roi_points], dtype=np.int32)
            cv2.polylines(resized, [pts], True, (0, 255, 0), thickness=2)
        img = Image.fromarray(resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=imgtk, text="")
        self.video_label.image = imgtk

    def _disable_non_control_widgets(self):
        for w in self.parameter_widgets + self.file_widgets:
            try:
                w.config(state='disabled')
            except:
                pass
        self.video_listbox.config(state=tk.DISABLED)

    def _enable_non_control_widgets(self):
        for w in self.parameter_widgets:
            try:
                w.config(state='normal')
            except:
                pass
        for w in self.file_widgets:
            try:
                w.config(state='disabled')
            except:
                pass
        self.video_listbox.config(state=tk.DISABLED)

    # ==================== 优化后的控制逻辑 ====================
    def on_control_click(self):
        if self.processing and self.paused:
            # ✅ 简化：不再手动重置状态或重建 ROI
            self.paused = False
            self.control_btn.config(text="暂停", style="TButton")
            self._disable_non_control_widgets()
            self.stop_btn.config(state="normal")
            self.progress_label.config(text="继续处理中...")
            self.status_var.set(f"处理中 | 倍速{self.current_speed}x")
            self.log_message("继续视频处理")
            self.info_label.config(text="继续视频处理")
        elif self.processing and not self.paused:
            self.paused = True
            self.control_btn.config(text="继续处理")
            self._enable_non_control_widgets()
            self.stop_btn.config(state="normal")
            self.progress_label.config(text="已暂停，可修改参数后继续")
            self.status_var.set("已暂停")
            self.log_message("暂停视频处理")
            self.info_label.config(text="视频处理已暂停")
        else:
            self._start_new_processing()

    def _start_new_processing(self):
        if self.cap is not None:
            with self.cap_lock:
                try:
                    if self.cap.isOpened():
                        self.cap.release()
                except Exception as e:
                    self.log_message(f"启动前清理 cap 出错: {e}")
            self.cap = None

        if not self.video_paths:
            messagebox.showwarning("警告", "请先添加视频文件")
            return

        try:
            test_file = os.path.join(self.save_path, f"test_{int(time.time())}.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            messagebox.showerror("路径错误", f"保存路径不可写:\n{self.save_path}\n{str(e)}")
            self.log_message(f"保存路径不可写: {str(e)}")
            return

        self.processing = True
        self.paused = False
        self.current_video_index = 0
        self.control_btn.config(text="暂停", style="TButton")
        self.stop_btn.config(state="normal")
        self._disable_non_control_widgets()
        self.status_var.set(f"处理中 | 倍速{self.current_speed}x")
        self.progress_label.config(text="开始处理...")
        self.log_message("开始处理视频列表")
        self.info_label.config(text="开始处理视频列表")
        threading.Thread(target=self.process_videos, daemon=True).start()

    # ==================== 参数回调 ====================
    def validate_threshold_input(self, event=None):
        try:
            v = float(self.threshold_entry_var.get())
            if 0.1 <= v <= 50:
                self.change_threshold = v / 100
                self.threshold_scale.set(v)
                self.threshold_label.config(text=f"{v:.1f}%")
                self.log_message(f"变化阈值设置为 {v:.1f}%")
            else:
                raise ValueError
        except ValueError:
            messagebox.showwarning("输入错误", "请输入 0.1 ~ 50 之间的有效数字")
            self.threshold_entry_var.set(f"{self.change_threshold*100:.1f}")

    def validate_interval_input(self, event=None):
        try:
            v = float(self.interval_entry_var.get())
            if 0.1 <= v <= 10:
                self.min_interval = v
                self.interval_scale.set(v)
                self.interval_label.config(text=f"{v:.1f}s")
                self.log_message(f"截图最小间隔设置为 {v:.1f}秒")
            else:
                raise ValueError
        except ValueError:
            messagebox.showwarning("输入错误", "请输入 0.1 ~ 10 之间的有效数字")
            self.interval_entry_var.set(f"{self.min_interval:.1f}")

    def update_threshold(self, value):
        v = float(value)
        self.change_threshold = v / 100
        self.threshold_label.config(text=f"{v:.1f}%")
        self.threshold_entry_var.set(f"{v:.1f}")
        self.log_message(f"变化阈值设置为 {v:.1f}%")

    def update_interval(self, value):
        v = float(value)
        self.min_interval = v
        self.interval_label.config(text=f"{v:.1f}秒")
        self.interval_entry_var.set(f"{v:.1f}")
        self.log_message(f"截图最小间隔设置为 {v:.1f}秒")

    def update_gmm_threshold(self, value):
        v = int(float(value))
        self.gmm_var.set(v)
        self.gmm_label.config(text=str(v))
        self.log_message(f"GMM 敏感度设置为 {v}")

    def update_frame_diff_threshold(self, value):
        v = int(float(value))
        self.fd_var.set(v)
        self.fd_label.config(text=str(v))
        self.log_message(f"帧差阈值设置为 {v}")

    # ==================== ROI 相关 ====================
    def select_roi(self):
        if self.roi_selected:
            self.cancel_roi()
            return
        if self.processing and not self.paused:
            messagebox.showinfo("提示", "请先暂停或停止视频处理，再选择ROI区域")
            return
        if not self.video_paths:
            messagebox.showwarning("警告", "请先添加视频文件")
            return
        if self.preview_frame is None:
            self.video_listbox.selection_clear(0, tk.END)
            self.video_listbox.selection_set(0)
            self.preview_selected_video()
        if self.preview_frame is None:
            messagebox.showwarning("警告", "无法加载视频预览，请检查文件有效性")
            return

        roi_window = tk.Toplevel(self.root)
        roi_window.title("选择关注区域 (ROI)")
        roi_window.geometry("1024x700")
        roi_window.transient(self.root)
        roi_window.grab_set()

        canvas = tk.Canvas(roi_window, bg="black")
        canvas.pack(fill=tk.BOTH, expand=True)

        pil_img = Image.fromarray(cv2.cvtColor(self.preview_frame, cv2.COLOR_BGR2RGB))
        self.roi_original_size = pil_img.size
        self.roi_scale_factor = 1.0
        self.roi_image = pil_img
        self.roi_photo = None

        roi_points: List[Tuple[float, float]] = []
        lines, circles = [], []

        def draw_roi():
            for item in lines + circles:
                canvas.delete(item)
            lines.clear(); circles.clear()
            if len(roi_points) == 0: return
            scaled = [(x * self.roi_scale_factor, y * self.roi_scale_factor) for x, y in roi_points]
            for i, (x, y) in enumerate(scaled):
                circles.append(canvas.create_oval(x-3, y-3, x+3, y+3, fill="red"))
                if i > 0:
                    lines.append(canvas.create_line(scaled[i-1][0], scaled[i-1][1], x, y, fill="lime", width=2))
            if len(roi_points) > 2:
                lines.append(canvas.create_line(scaled[-1][0], scaled[-1][1], scaled[0][0], scaled[0][1], fill="lime", width=2))

        def update_image():
            new_size = (int(self.roi_original_size[0] * self.roi_scale_factor),
                        int(self.roi_original_size[1] * self.roi_scale_factor))
            resized_img = self.roi_image.resize(new_size, Image.LANCZOS)
            self.roi_photo = ImageTk.PhotoImage(resized_img)
            canvas.config(scrollregion=(0, 0, new_size[0], new_size[1]))
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=self.roi_photo)

            # === 绘制 10x10 虚线参考网格 ===
            grid_rows, grid_cols = 10, 10
            w, h = new_size
            for i in range(1, grid_cols):
                x = int(w * i / grid_cols)
                canvas.create_line(x, 0, x, h, dash=(4, 4), fill="white", width=1)
            for j in range(1, grid_rows):
                y = int(h * j / grid_rows)
                canvas.create_line(0, y, w, y, dash=(4, 4), fill="white", width=1)

            draw_roi()

        def fit_to_window():
            win_w, win_h = roi_window.winfo_width(), roi_window.winfo_height()
            if win_w <= 1 or win_h <= 1:
                roi_window.after(100, fit_to_window)
                return
            img_w, img_h = self.roi_original_size
            self.roi_scale_factor = min((win_w-20)/img_w, (win_h-20)/img_h, 1.0)
            update_image()

        def on_click(e):
            x = canvas.canvasx(e.x) / self.roi_scale_factor
            y = canvas.canvasy(e.y) / self.roi_scale_factor
            roi_points.append((x, y))
            draw_roi()

        def on_right_click(e):
            if roi_points: roi_points.pop(); draw_roi()

        def on_key(e):
            if e.char.lower() == 'r':
                roi_points.clear(); draw_roi()

        def on_wheel(e):
            x, y = canvas.canvasx(e.x), canvas.canvasy(e.y)
            orig_x, orig_y = x / self.roi_scale_factor, y / self.roi_scale_factor
            if e.delta > 0: self.roi_scale_factor *= 1.2
            else: self.roi_scale_factor /= 1.2
            self.roi_scale_factor = max(0.1, min(self.roi_scale_factor, 5.0))
            update_image()
            dx = orig_x * self.roi_scale_factor - x
            dy = orig_y * self.roi_scale_factor - y
            canvas.xview_scroll(int(dx), "units")
            canvas.yview_scroll(int(dy), "units")

        canvas.bind("<Button-1>", on_click)
        canvas.bind("<Button-3>", on_right_click)
        roi_window.bind("<Key>", on_key)
        canvas.bind("<MouseWheel>", on_wheel)
        canvas.focus_set()

        roi_window.update_idletasks()
        fit_to_window()

        btn_frame = ttk.Frame(roi_window)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        def confirm():
            if len(roi_points) >= 3:
                self.roi_points = roi_points
                self.roi_selected = True
                self.roi_button.config(text="取消选取", command=self.cancel_roi)
                self.create_roi_mask()
                roi_window.destroy()
                self.log_message(f"已选择ROI区域，{len(roi_points)}个顶点")
                self.info_label.config(text=f"已选择ROI区域，{len(roi_points)}个顶点")
            else:
                messagebox.showerror("错误", "至少需要3个点才能构成多边形")

        def cancel():
            roi_window.destroy()

        ttk.Button(btn_frame, text="确认选择", command=confirm).pack(side=tk.RIGHT, padx=(5,0))
        ttk.Button(btn_frame, text="取消", command=cancel).pack(side=tk.RIGHT)

    def cancel_roi(self):
        self.roi_selected = False
        self.roi_points = []
        self.roi_mask = None
        self.roi_button.config(text="选择ROI区域", command=self.select_roi)
        self.roi_status_label.config(text="未选取关注区域", foreground="gray")
        if self.preview_frame is not None:
            self.display_frame(cv2.cvtColor(self.preview_frame, cv2.COLOR_BGR2RGB))
        self.log_message("已取消ROI区域选择")
        self.info_label.config(text="已取消ROI区域选择", foreground="gray")

    def create_roi_mask(self, target_size: Optional[Tuple[int, int]] = None):
        if not self.roi_selected or not self.roi_points or self.preview_frame is None:
            self.roi_mask = None
            return
        try:
            h, w = self.preview_frame.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            pts = np.array([[int(x), int(y)] for x, y in self.roi_points], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            self.roi_mask = mask
            area_ratio = cv2.countNonZero(mask) / mask.size * 100
            self.roi_status_label.config(text=f"已选取ROI区域 ({area_ratio:.1f}% 画面)", foreground="red")
            self.log_message(f"ROI 面积占比: {area_ratio:.1f}%")
        except Exception as e:
            self.log_message(f"创建ROI掩码失败: {str(e)}")
            messagebox.showerror("错误", f"创建ROI掩码失败: {str(e)}")
            self.roi_mask = None

    # ==================== 视频处理（优化后）====================
    def safe_video_capture(self, video_path: str) -> Tuple[Optional[cv2.VideoCapture], Optional[str]]:
        if not os.path.exists(video_path):
            return None, f"视频文件不存在: {video_path}"
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                return cap, None
        except Exception:
            pass
        if os.name == 'nt':
            try:
                long_path = "\\\\?\\" + os.path.abspath(video_path)
                cap = cv2.VideoCapture(long_path)
                if cap.isOpened():
                    return cap, None
            except Exception:
                pass
        return None, f"无法打开视频: {os.path.basename(video_path)}"

    def process_videos(self):
        try:
            current_index = self.current_video_index
            while current_index < len(self.video_paths):
                if not self.processing:
                    break
                video_path = self.video_paths[current_index]
                self.safe_ui_call(self.progress_label.config, text=f"正在处理第 {current_index + 1} 个视频")
                self.log_message(f"开始处理视频: {video_path}")

                if self.cap is not None:
                    with self.cap_lock:
                        if self.cap.isOpened():
                            self.cap.release()
                    self.cap = None

                self.cap, err = self.safe_video_capture(video_path)
                if err:
                    self.safe_ui_call(messagebox.showerror, "错误", err)
                    self.log_message(err)
                    if self.processing:
                        current_index += 1
                        self.current_video_index = current_index
                    continue

                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

                with self.cap_lock:
                    ret, first_frame = self.cap.read()
                if not ret:
                    self.log_message(f"无法读取首帧: {video_path}")
                    if self.processing:
                        current_index += 1
                        self.current_video_index = current_index
                    continue

                if self.target_height > 0 and first_frame.shape[0] > self.target_height:
                    scale = self.target_height / first_frame.shape[0]
                    new_w = int(first_frame.shape[1] * scale)
                    first_frame = cv2.resize(first_frame, (new_w, self.target_height), interpolation=cv2.INTER_AREA)

                roi_for_processor = self.roi_mask if self.roi_selected else None
                processor = VideoProcessor(
                    gmm_var=self.gmm_var.get(),
                    fd_var=self.fd_var.get(),
                    roi_mask=roi_for_processor
                )
                # === GMM 预热（不检测）===
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # 预热阶段：读取前 N 帧，逐帧喂给 GMM
                for i in range(GMM_PREHEAT_FRAMES):
                    with self.cap_lock:
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                    # 预处理（resize + ROI）
                    _, gray = processor.preprocess_frame(frame)
                    # 直接喂给 GMM 背景模型（每帧一次）
                    processor.gmm.apply(gray)

                # 预热结束，从第 GMM_PREHEAT_FRAMES 帧开始正式检测
                frame_id = GMM_PREHEAT_FRAMES

                last_saved_time = 0

                while frame_id < total_frames:
                    if not self.processing:
                        break
                    if self.paused:
                        time.sleep(0.05)
                        continue

                    # ✅ 优化跳帧
                    if self.current_speed > 1:
                        next_frame = min(frame_id + self.current_speed, total_frames - 1)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                        with self.cap_lock:
                            ret, frame = self.cap.read()
                        frame_id = next_frame + 1
                    else:
                        with self.cap_lock:
                            ret, frame = self.cap.read()
                        frame_id += 1

                    if not ret:
                        break
                    # ===== 预处理 =====
                    _, gray = processor.preprocess_frame(frame)
                    # ===== 检测变化（传入学习率）=====
                    valid_change, fg_mask, change_ratio = processor.detect_change(gray)
                    # ===== 判断是否触发截图 =====
                    now = time.time()
                    if valid_change and (now - last_saved_time) > self.min_interval:
                        last_saved_time = now
                        video_basename = os.path.splitext(os.path.basename(video_path))[0]
                        self.save_screenshot(frame.copy(), video_basename, frame_id)

                    if not self.background_mode_var.get():
                        now_time = time.time()
                        if not hasattr(self, '_last_preview_update_time'):
                            self._last_preview_update_time = now_time
                        if now_time - self._last_preview_update_time >= PREVIEW_UPDATE_INTERVAL:
                            # 确保 frame 是 resize 后的，且与 fg_mask 同高宽
                            h, w = fg_mask.shape
                            if frame.shape[:2] != (h, w):
                                # 安全 resize frame 到 fg_mask 尺寸（理论上不应发生，但防御性编程）
                                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                            color_mask = np.zeros_like(frame)
                            color_mask[:, :, 2] = fg_mask
                            marked = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)
                            if self.roi_selected and self.roi_points:
                                h, w = frame.shape[:2]
                                if self.preview_frame is not None:
                                    orig_h, orig_w = self.preview_frame.shape[:2]
                                    sx, sy = w / orig_w, h / orig_h
                                    pts = np.array([[int(x * sx), int(y * sy)] for x, y in self.roi_points], dtype=np.int32)
                                    cv2.polylines(marked, [pts], True, (0, 255, 0), 2)
                            cv2.putText(
                                marked,
                                f"Change: {change_ratio*100:.1f}% | Speed: {self.current_speed}x",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1
                            )
                            rgb_marked = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
                            self.safe_ui_call(self.display_frame, rgb_marked)
                            self._last_preview_update_time = now_time

                    overall_progress = ((current_index + (frame_id / total_frames)) / len(self.video_paths)) * 100
                    self.safe_ui_call(self.total_percent_label.config, text=f"{overall_progress:.1f}%")
                    self.safe_ui_call(self.progress_var.set, overall_progress)
                    self.safe_ui_call(
                        self.progress_label.config,
                        text=f"第{current_index + 1}个[{frame_id}/{total_frames}]，共{len(self.video_paths)}个视频"
                    )

                if self.processing:
                    current_index += 1
                    self.current_video_index = current_index

            if self.processing:
                self.safe_ui_call(self.progress_label.config, text="所有视频处理完毕")
                self.safe_ui_call(self.status_var.set, "处理完成")
                self.safe_ui_call(self.progress_var.set, 100)
                self.safe_ui_call(self._stop_cleanup)
                self.log_message(f"所有 {len(self.video_paths)} 个视频处理完成")
                self.safe_ui_call(messagebox.showinfo, "完成", "所有视频处理已完成")

        except Exception as e:
            self.processing = False
            import traceback
            self.log_message(f"处理出错: {e}\n{traceback.format_exc()}")
            self.safe_ui_call(messagebox.showerror, "错误", str(e))
        finally:
            if self.cap is not None:
                with self.cap_lock:
                    if self.cap.isOpened():
                        self.cap.release()
                self.cap = None

    # ==================== UI 交互 ====================
    def toggle_background_mode(self):
        mode = "开启" if self.background_mode_var.get() else "关闭"
        self.log_message(f"后台模式已{mode}")
        self.info_label.config(text=f"后台模式已{mode} - 预览{'禁用' if self.background_mode_var.get() else '启用'}")

    def preview_selected_video(self):
        if not self.video_paths:
            messagebox.showwarning("警告", "请先添加视频文件")
            return
        selected = self.video_listbox.curselection()
        if not selected:
            messagebox.showwarning("警告", "请先选择一个视频文件")
            return
        if self.processing:
            self.stop_processing()
            time.sleep(0.5)

        video_path = self.video_paths[selected[0]]

        if self.cap is not None:
            with self.cap_lock:
                if self.cap.isOpened():
                    self.cap.release()
            self.cap = None

        self.cap, err = self.safe_video_capture(video_path)
        if err:
            messagebox.showerror("错误", err)
            self.log_message(err)
            self.cap = None
            return

        with self.cap_lock:
            ret, frame = self.cap.read()
        if ret:
            self.preview_frame = frame.copy()
            self.display_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.progress_label.config(text=f"预览: {os.path.basename(video_path)}")
            self.info_label.config(text=f"预览中: {os.path.basename(video_path)}")
            self.log_message(f"预览视频: {os.path.basename(video_path)}")
        else:
            messagebox.showerror("错误", f"无法读取视频帧: {os.path.basename(video_path)}")
            with self.cap_lock:
                if self.cap.isOpened():
                    self.cap.release()
            self.cap = None

    def add_videos(self):
        paths = filedialog.askopenfilenames(title="选择视频文件", filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv;*.flv")])
        if paths:
            added = 0
            for p in paths:
                if p not in self.video_paths:
                    self.video_paths.append(p)
                    self.video_listbox.insert(tk.END, os.path.basename(p))
                    added += 1
            self.log_message(f"添加了 {added} 个视频文件")
            self.info_label.config(text=f"已添加 {added} 个视频文件")

    def remove_video(self):
        selected = self.video_listbox.curselection()
        if selected:
            for i in sorted(selected, reverse=True):
                del self.video_paths[i]
                self.video_listbox.delete(i)
            self.log_message(f"移除了 {len(selected)} 个视频文件")
            self.info_label.config(text=f"已移除 {len(selected)} 个视频文件")

    def clear_videos(self):
        count = len(self.video_paths)
        self.video_paths.clear()
        self.video_listbox.delete(0, tk.END)
        self.current_video_index = 0
        self.cancel_roi()
        if self.cap is not None:
            with self.cap_lock:
                if self.cap.isOpened():
                    self.cap.release()
            self.cap = None
        self.log_message(f"清空了 {count} 个视频文件")
        self.info_label.config(text=f"已清空所有视频文件 ({count} 个)")

    def change_save_path(self):
        new_path = filedialog.askdirectory(title="选择截图保存路径")
        if new_path:
            try:
                test_file = os.path.join(new_path, f"test_{int(time.time())}.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                self.save_path = new_path
                self.save_path_var.set(new_path)
                self.ensure_directory_exists(new_path)
                self.log_message(f"保存路径更改至: {new_path}")
                self.info_label.config(text=f"保存路径: {os.path.basename(new_path)}")
            except Exception as e:
                messagebox.showwarning("路径警告", f"所选路径不可写:\n{str(e)}")
                self.log_message(f"路径不可写: {str(e)}")

    def set_speed(self, speed: int):
        if speed in self.speed_levels:
            self.current_speed = speed
            self.speed_label.config(text=f"当前: {speed}x")
            self.status_var.set(f"{self.status_var.get().split('|')[0].strip()} | 倍速: {speed}x")
            self.log_message(f"处理倍速设置为 {speed}倍")
            self.info_label.config(text=f"处理倍速: {speed}倍")

    def _stop_cleanup(self):
        self.processing = False
        self.paused = False
        self.current_video_index = 0

        for w in self.parameter_widgets + self.file_widgets:
            try:
                w.config(state='normal')
            except:
                pass
        self.video_listbox.config(state=tk.NORMAL)

        self.control_btn.config(text="开始处理", style="TButton", state="normal")
        self.stop_btn.config(state="disabled")
        self.control_btn.focus_set()

        self.status_var.set("已停止")
        self.progress_label.config(text="处理已停止")
        self.progress_var.set(0)
        self.info_label.config(text="处理已停止")
        self.root.after(50, lambda: self.control_btn.config(text="开始处理", style="TButton"))

    def stop_processing(self):
        if not (self.processing or self.paused):
            return
        self.processing = False
        self.paused = False
        self.log_message("处理已停止")
        self.root.after(100, self._stop_cleanup)

    def save_screenshot(self, frame: np.ndarray, video_basename: str, frame_num: int) -> Optional[str]:
        try:
            safe_name = "".join(c for c in video_basename if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}_frame_{frame_num}.jpg"
            full_path = os.path.join(self.save_path, filename)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(full_path, quality=95)
            self.log_message(f"截图已保存: {full_path}")
            self.info_label.config(text=f"已保存截图: {filename}")
            return full_path
        except Exception as e:
            self.log_message(f"保存截图失败: {str(e)}")
            messagebox.showerror("保存错误", f"保存截图失败:\n{str(e)}")
            self.info_label.config(text="截图保存失败")
            return None
        
