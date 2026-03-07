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
import traceback

# ==================== 常量定义 ====================
MIN_AREA = 100
DEFAULT_GMM_VAR_THRESHOLD = 25
DEFAULT_FRAME_DIFF_THRESHOLD = 30
DEFAULT_CHANGE_THRESHOLD = 0.05
DEFAULT_MIN_INTERVAL = 1.0
PREVIEW_UPDATE_INTERVAL = 0.3
GMM_PREHEAT_FRAMES = 10

# ==================== 专业配色方案（协调统一） ====================
COLORS = {
    'bg_main': '#f5f7fa',
    'bg_card': '#ffffff',
    'bg_input': '#ffffff',
    'primary': '#1890ff',
    'primary_hover': '#40a9ff',
    'primary_active': '#096dd9',
    'success': '#52c41a',
    'warning': '#faad14',
    'danger': '#ff4d4f',
    'info': '#1890ff',
    'text_main': '#262626',
    'text_sec': '#595959',
    'text_disable': '#bfbfbf',
    'border': '#d9d9d9',
    'border_light': '#e8e8e8',
    'divider': '#f0f0f0',
}

# ==================== 视频处理逻辑类 ====================
class VideoProcessor:
    def __init__(self, gmm_var: int, fd_var: int, roi_mask: Optional[np.ndarray] = None):
        self.gmm_var = gmm_var
        self.fd_var = fd_var
        self.roi_mask = roi_mask
        self.gmm = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=self.gmm_var, detectShadows=False
        )
        self.prev_gray = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def preprocess_frame(self, frame: np.ndarray, target_height: int) -> Tuple[np.ndarray, np.ndarray]:
        if frame.shape[0] > target_height:
            scale = target_height / frame.shape[0]
            new_w = int(frame.shape[1] * scale)
            frame = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.roi_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=self.roi_mask)
        return frame, gray
    
    def detect_change(self, gray: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            for _ in range(GMM_PREHEAT_FRAMES):
                self.gmm.apply(gray)
            return False, np.zeros_like(gray), 0.0
        
        gmm_mask = self.gmm.apply(gray)
        _, gmm_mask = cv2.threshold(gmm_mask, 254, 255, cv2.THRESH_BINARY)
        frame_diff = cv2.absdiff(gray, self.prev_gray)
        _, diff_mask = cv2.threshold(frame_diff, self.fd_var, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_and(gmm_mask, diff_mask)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        valid_change = False
        if cv2.countNonZero(fg_mask) > 0:
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
                    valid_change = True
                    break
        
        self.prev_gray = gray.copy()
        total_pixels = cv2.countNonZero(self.roi_mask) if self.roi_mask is not None else gray.size
        change_pixels = cv2.countNonZero(fg_mask)
        change_ratio = change_pixels / total_pixels if total_pixels > 0 else 0.0
        return valid_change, fg_mask, change_ratio

# ==================== 主界面类 ====================
class GMMVideoDetector:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("视频画面变化检测 by geckotao")
        
        # 提前初始化必要属性
        self.log_file_path = None
        self.log_file_handle = None
        self._log_file_ready = False
        self.ui_queue = queue.Queue()
        self.cap_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.cap = None
        self._proc_prev_gray = None
        
        # 简化 DPI 检测
        self.dpi_scale = self.get_dpi_scale()
        self.scaled_font_size = int(10 * self.dpi_scale)
        
        # 窗口配置（立即显示）
        self.root.geometry(f"{int(1280 * self.dpi_scale)}x{int(800 * self.dpi_scale)}")
        self.root.minsize(int(1100 * self.dpi_scale), int(750 * self.dpi_scale))
        self.root.configure(bg=COLORS['bg_main'])
        self.root.update_idletasks() 
        
        # 状态变量初始化
        self.background_mode_var = tk.BooleanVar(value=False)
        self.video_paths: List[str] = []
        self.current_video_index = 0
        self.processing = False
        self.paused = False
        self.roi_selected = False
        self.roi_points: List[Tuple[float, float]] = []
        self.change_threshold = DEFAULT_CHANGE_THRESHOLD
        self.save_path = os.path.join(os.getcwd(), "变化截图")
        self.min_interval = DEFAULT_MIN_INTERVAL
        self.target_height_var = tk.IntVar(value=480)
        self.preview_frame: Optional[np.ndarray] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.processing_roi_mask: Optional[np.ndarray] = None
        self.speed_levels = [1, 2, 4, 8, 16, 24, 32, 64]
        self.current_speed = 1
        
        # 线程控制
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # UI 控件引用
        self.control_btn = None
        self.stop_btn = None
        self.video_listbox = None
        self.video_label = None
        self.parameter_widgets = []
        self.file_widgets = []
        self.speed_buttons = []
        
        # 先构建 UI（用户可立即交互）
        self._setup_styles()
        self.create_widgets()
        
        # 后台初始化日志和图标（不阻塞启动）
        self.root.after(100, self._init_background_tasks)
        
        # 启动 UI 队列
        self.root.after(50, self.process_ui_queue)
    
    def _init_background_tasks(self):
        """后台初始化非关键任务"""
        self.log_dir = os.path.join(os.getcwd(), "检测日志")
        self.ensure_directory_exists(self.log_dir)
        self.ensure_directory_exists(self.save_path)
        self.init_log_file()
        self.setup_icon()
        self._log_file_ready = True
        self.log_message("程序启动 (v1.3 版)")
        self.log_message(f"保存路径：{self.save_path}")
    
    def log_message(self, message: str):
        """优化日志写入"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())
        if getattr(self, '_log_file_ready', False):
            log_file_handle = getattr(self, 'log_file_handle', None)
            if log_file_handle:
                try:
                    log_file_handle.write(log_entry)
                    log_file_handle.flush()
                except:
                    pass
    
    def get_dpi_scale(self) -> float:
        """简化 DPI 检测"""
        try:
            if os.name == 'nt':
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                    dpi = ctypes.windll.user32.GetDpiForSystem()
                    return dpi / 96.0
                except:
                    pass
            return 1.0
        except:
            return 1.0
    
    # ==================== 资源管理 ====================
    def _release_capture(self):
        with self.cap_lock:
            if self.cap is not None:
                try:
                    if self.cap.isOpened():
                        self.cap.release()
                except Exception as e:
                    self.log_message(f"释放视频 capture 失败：{e}")
                finally:
                    self.cap = None
    
    def _close_log_file(self):
        if self.log_file_handle:
            try:
                self.log_file_handle.flush()
                self.log_file_handle.close()
            except:
                pass
            self.log_file_handle = None
    
    # ==================== DPI 相关 ====================
    def get_dpi_scale(self) -> float:
        try:
            if os.name == 'nt':
                import platform
                win_ver = platform.version()
                if win_ver >= "10.0":
                    try:
                        ctypes.windll.user32.SetProcessDPIAware()
                        dpi = ctypes.windll.user32.GetDpiForSystem()
                        return dpi / 96.0
                    except:
                        pass
                else:
                    try:
                        ctypes.windll.user32.SetProcessDPIAware()
                        return 1.25 if self.root.winfo_screenwidth() >= 1920 else 1.0
                    except:
                        pass
            return 1.0
        except:
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
        except:
            pass
    
    # ==================== 样式配置（配色协调核心） ====================
    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except:
            pass
        f = self.scaled_font_size
        
        # 框架
        style.configure("TFrame", background=COLORS['bg_main'])
        style.configure("Card.TFrame", background=COLORS['bg_card'])
        
        # 标签（统一文字颜色）
        style.configure("TLabel",
                       background=COLORS['bg_main'],
                       font=('Microsoft YaHei UI', f),
                       foreground=COLORS['text_main'])
        style.configure("Heading.TLabel",
                       background=COLORS['bg_main'],
                       font=('Microsoft YaHei UI', f, 'bold'),
                       foreground=COLORS['primary'])
        style.configure("Status.TLabel",
                       background=COLORS['bg_main'],
                       font=('Consolas', f),
                       foreground=COLORS['success'])
        
        # 按钮（统一蓝色系）
        style.configure("TButton",
                       font=('Microsoft YaHei UI', f),
                       padding=(int(12 * self.dpi_scale), int(6 * self.dpi_scale)))
        style.map("TButton",
                 background=[("active", COLORS['primary_hover']),
                            ("pressed", COLORS['primary_active'])],
                 foreground=[("active", "white"), ("pressed", "white")])
        
        # 强调按钮
        style.configure("Accent.TButton",
                       font=('Microsoft YaHei UI', f, 'bold'),
                       padding=(int(18 * self.dpi_scale), int(8 * self.dpi_scale)))
        style.map("Accent.TButton",
                 background=[("active", COLORS['primary_active']),
                            ("pressed", "#0050b3")],
                 foreground=[("active", "white"), ("pressed", "white")])
        
        # 危险按钮（柔和红）
        style.configure("Danger.TButton",
                       font=('Microsoft YaHei UI', f, 'bold'),
                       padding=(int(18 * self.dpi_scale), int(8 * self.dpi_scale)))
        style.map("Danger.TButton",
                 background=[("active", "#d9363e"),
                            ("pressed", "#a8071a")],
                 foreground=[("active", "white"), ("pressed", "white")])
        
        # 输入框
        style.configure("TEntry",
                       font=('Microsoft YaHei UI', f),
                       padding=int(6 * self.dpi_scale),
                       fieldbackground=COLORS['bg_input'])
        
        # 标签框（卡片式，统一边框色）
        style.configure("TLabelframe",
                       background=COLORS['bg_main'],
                       borderwidth=1,
                       relief="flat")
        style.configure("TLabelframe.Label",
                       background=COLORS['bg_main'],
                       font=('Microsoft YaHei UI', f, 'bold'),
                       foreground=COLORS['primary'],
                       padding=(int(12 * self.dpi_scale), int(4 * self.dpi_scale)))
        
        # 进度条（统一蓝色）
        style.configure("Horizontal.TProgressbar",
                       thickness=int(20 * self.dpi_scale),
                       background=COLORS['primary'],
                       troughcolor=COLORS['border_light'],
                       borderwidth=0)
        
        # 笔记本书签
        style.configure("TNotebook",
                       background=COLORS['bg_main'],
                       borderwidth=0)
        style.configure("TNotebook.Tab",
                       font=('Microsoft YaHei UI', f, 'bold'),
                       padding=(int(24 * self.dpi_scale), int(10 * self.dpi_scale)),
                       background=COLORS['border_light'],
                       foreground=COLORS['text_sec'])
        style.map("TNotebook.Tab",
                 background=[("selected", COLORS['bg_card'])],
                 foreground=[("selected", COLORS['primary'])])
        
        # 复选框
        style.configure("TCheckbutton",
                       font=('Microsoft YaHei UI', f),
                       background=COLORS['bg_main'],
                       foreground=COLORS['text_main'])
        
        # 列表框
        style.configure("TListbox",
                       font=('Microsoft YaHei UI', f),
                       background=COLORS['bg_card'],
                       foreground=COLORS['text_main'],
                       selectbackground=COLORS['primary'],
                       selectforeground="white")
        
        self.style = style
    
    # ==================== 文件与日志 ====================
    def ensure_directory_exists(self, path: str):
        if os.path.exists(path):
            return
        for attempt in range(3):
            try:
                os.makedirs(path, exist_ok=True)
                self.log_message(f"创建目录：{path}")
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
            self.log_file_handle = open(self.log_file_path, 'a', encoding='utf-8')
            self.log_message(f"日志文件初始化成功：{self.log_file_path}")
        except Exception as e:
            error_msg = f"初始化日志文件失败：{str(e)}"
            print(error_msg)
            self.safe_ui_call(messagebox.showerror, "错误", error_msg)
            self.log_file_path = None
    
    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())
        log_file_handle = getattr(self, 'log_file_handle', None)
        log_file_path = getattr(self, 'log_file_path', None)
        with self.log_lock:
            if log_file_handle:
                try:
                    log_file_handle.write(log_entry)
                    log_file_handle.flush()
                except Exception as e:
                    print(f"写入日志失败：{str(e)}")
            elif log_file_path:
                try:
                    with open(log_file_path, 'a', encoding='utf-8') as f:
                        f.write(log_entry)
                except:
                    pass
    
    def __del__(self):
        self._close_log_file()
        self._release_capture()
    
    # ==================== UI 相关 ====================
    def setup_icon(self):
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
            if os.path.exists(icon_path):
                icon_img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, icon_img)
                self.log_message(f"成功设置窗口图标")
        except Exception as e:
            self.log_message(f"设置窗口图标失败：{str(e)}")
    
    def create_widgets(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        padx = int(12 * self.dpi_scale)
        pady = int(12 * self.dpi_scale)
        
        # 左侧面板
        left_frame = ttk.Frame(self.root, width=int(420 * self.dpi_scale))
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(padx, padx//2), pady=pady)
        left_frame.grid_propagate(False)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        control_notebook = ttk.Notebook(left_frame)
        control_notebook.grid(row=0, column=0, sticky="nsew",
                             padx=int(5 * self.dpi_scale),
                             pady=int(5 * self.dpi_scale))
        self._create_settings_tab(control_notebook)
        self._create_control_tab(control_notebook)
        self._create_help_tab(control_notebook)
        
        # 右侧预览区
        self._create_preview_area(padx, pady)
        
        # 底部状态栏（统一浅色）
        self._create_status_bar()
        
        # 分离控件组
        self.parameter_widgets = [
            self.roi_button,
            self.gmm_scale, self.gmm_label,
            self.fd_scale, self.fd_label,
            self.threshold_scale, self.threshold_entry, self.threshold_label,
            self.interval_scale, self.interval_entry, self.interval_label,
            self.target_height_scale, self.target_height_entry, self.target_height_label,
            self.save_path_entry, self.save_path_btn
        ]
        self.file_widgets = [
            self.add_btn, self.clear_btn, self.remove_btn, self.preview_btn,
            self.video_listbox
        ]
    
    def _create_status_bar(self):
        """创建底部状态栏（浅色统一风格）"""
        status_bar = tk.Frame(self.root, height=int(35 * self.dpi_scale),
                             bg=COLORS['bg_card'], relief=tk.FLAT,
                             highlightthickness=1, highlightbackground=COLORS['border_light'])
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        status_bar.grid_propagate(False)
        
        # 左侧状态
        self.status_bar_label = tk.Label(status_bar, text="就绪",
                                        bg=COLORS['bg_card'],
                                        fg=COLORS['success'],
                                        font=('Microsoft YaHei UI', self.scaled_font_size),
                                        anchor=tk.W)
        self.status_bar_label.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                  padx=int(15 * self.dpi_scale))
        
        # 右侧版本
        version_label = tk.Label(status_bar, text="v1.3",
                                bg=COLORS['bg_card'],
                                fg=COLORS['text_sec'],
                                font=('Microsoft YaHei UI', self.scaled_font_size),
                                anchor=tk.E)
        version_label.pack(side=tk.RIGHT, padx=int(15 * self.dpi_scale))
    
    def _create_settings_tab(self, notebook):
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="参数设置")
        
        settings_canvas = tk.Canvas(settings_frame, highlightthickness=0, bg=COLORS['bg_main'])
        settings_scrollbar = ttk.Scrollbar(settings_frame, orient="vertical",
                                          command=settings_canvas.yview)
        settings_scrollable_frame = ttk.Frame(settings_canvas, style="TFrame")
        settings_scrollable_frame.bind("<Configure>",
                                      lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")))
        settings_canvas.create_window((0, 0), window=settings_scrollable_frame, anchor="nw")
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        settings_canvas.pack(side="left", fill="both", expand=True)
        settings_scrollbar.pack(side="right", fill="y")
        
        row = 0
        frame_pady = (int(8 * self.dpi_scale), int(12 * self.dpi_scale))
        inner_pad = int(15 * self.dpi_scale)
        
        self._create_file_management_frame(settings_scrollable_frame, row, frame_pady, inner_pad)
        row += 1
        self._create_roi_frame(settings_scrollable_frame, row, frame_pady, inner_pad)
        row += 1
        self._create_param_frame(settings_scrollable_frame, row, frame_pady, inner_pad)
        row += 1
        self._create_save_frame(settings_scrollable_frame, row, frame_pady, inner_pad)
        row += 1
        
        ttk.Label(settings_scrollable_frame, text="").grid(row=row, column=0,
                                                          pady=int(15 * self.dpi_scale))
    
    def _create_file_management_frame(self, parent, row, pady, inner_pad):
        file_frame = ttk.LabelFrame(parent, text="视频文件管理",
                                   padding=(inner_pad, int(10 * self.dpi_scale)))
        file_frame.grid(row=row, column=0, sticky="ew", pady=pady,
                       padx=int(8 * self.dpi_scale))
        parent.columnconfigure(0, weight=1)
        
        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=tk.X, pady=(0, int(10 * self.dpi_scale)))
        self.add_btn = ttk.Button(btn_row, text="添加视频文件", command=self.add_videos)
        self.add_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                         padx=(0, int(6 * self.dpi_scale)))
        self.clear_btn = ttk.Button(btn_row, text="清空列表", command=self.clear_videos)
        self.clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        list_frame_inner = ttk.Frame(file_frame)
        list_frame_inner.pack(fill=tk.X, pady=(0, int(10 * self.dpi_scale)))
        self.video_listbox = tk.Listbox(list_frame_inner, height=int(6 * self.dpi_scale),
                                       selectmode=tk.EXTENDED, bd=1, relief=tk.SOLID,
                                       font=('Microsoft YaHei UI', self.scaled_font_size),
                                       bg=COLORS['bg_card'],
                                       selectbackground=COLORS['primary'],
                                       selectforeground="white")
        self.video_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll = ttk.Scrollbar(list_frame_inner, orient="vertical",
                                command=self.video_listbox.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_listbox.configure(yscrollcommand=v_scroll.set)
        
        action_btn_frame = ttk.Frame(file_frame)
        action_btn_frame.pack(fill=tk.X)
        self.remove_btn = ttk.Button(action_btn_frame, text="移除", command=self.remove_video)
        self.remove_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                            padx=(0, int(4 * self.dpi_scale)))
        self.preview_btn = ttk.Button(action_btn_frame, text="预览",
                                     command=self.preview_selected_video)
        self.preview_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                             padx=(int(4 * self.dpi_scale), 0))
    
    def _create_roi_frame(self, parent, row, pady, inner_pad):
        roi_frame = ttk.LabelFrame(parent, text="关注区域 (ROI)",
                                  padding=(inner_pad, int(10 * self.dpi_scale)))
        roi_frame.grid(row=row, column=0, sticky="ew", pady=pady,
                      padx=int(8 * self.dpi_scale))
        
        self.roi_button = ttk.Button(roi_frame, text="选择 ROI 区域", command=self.select_roi)
        self.roi_button.pack(fill=tk.X, pady=(0, int(6 * self.dpi_scale)))
        
        self.roi_status_label = ttk.Label(roi_frame, text="未选取关注区域",
                                         foreground=COLORS['text_sec'])
        self.roi_status_label.pack(anchor=tk.W, pady=(0, int(3 * self.dpi_scale)))
        
        ttk.Label(roi_frame,
                 text="操作：左键加点，右键删点，点【确认选择】选取关注区域。",
                 font=('Microsoft YaHei UI', max(8, int(8 * self.dpi_scale))),
                 foreground=COLORS['text_sec']).pack(anchor=tk.W)
    
    def _create_param_frame(self, parent, row, pady, inner_pad):
        param_frame = ttk.LabelFrame(parent, text="检测参数",
                                    padding=(inner_pad, int(10 * self.dpi_scale)))
        param_frame.grid(row=row, column=0, sticky="ew", pady=pady,
                        padx=int(8 * self.dpi_scale))
        
        ttk.Label(param_frame, text="GMM 敏感度 (varThreshold)",
                 font=('Microsoft YaHei UI', self.scaled_font_size, 'bold')).pack(anchor=tk.W,
                 pady=(0, int(3 * self.dpi_scale)))
        gmm_f = ttk.Frame(param_frame)
        gmm_f.pack(fill=tk.X, pady=(0, int(10 * self.dpi_scale)))
        self.gmm_var = tk.IntVar(value=25)
        self.gmm_scale = ttk.Scale(gmm_f, from_=10, to=50, variable=self.gmm_var,
                                  command=self.update_gmm_threshold)
        self.gmm_scale.pack(side=tk.LEFT, fill=tk.X, expand=True,
                           padx=(0, int(6 * self.dpi_scale)))
        self.gmm_label = ttk.Label(gmm_f, text="25", width=int(6 * self.dpi_scale),
                                  font=('Consolas', self.scaled_font_size))
        self.gmm_label.pack(side=tk.RIGHT)
        
        ttk.Label(param_frame, text="帧间差分阈值",
                 font=('Microsoft YaHei UI', self.scaled_font_size, 'bold')).pack(anchor=tk.W,
                 pady=(0, int(3 * self.dpi_scale)))
        fd_f = ttk.Frame(param_frame)
        fd_f.pack(fill=tk.X, pady=(0, int(10 * self.dpi_scale)))
        self.fd_var = tk.IntVar(value=30)
        self.fd_scale = ttk.Scale(fd_f, from_=5, to=60, variable=self.fd_var,
                                 command=self.update_frame_diff_threshold)
        self.fd_scale.pack(side=tk.LEFT, fill=tk.X, expand=True,
                          padx=(0, int(6 * self.dpi_scale)))
        self.fd_label = ttk.Label(fd_f, text="30", width=int(6 * self.dpi_scale),
                                 font=('Consolas', self.scaled_font_size))
        self.fd_label.pack(side=tk.RIGHT)
        
        ttk.Label(param_frame, text="画面变化阈值 (%)",
                 font=('Microsoft YaHei UI', self.scaled_font_size, 'bold')).pack(anchor=tk.W,
                 pady=(0, int(3 * self.dpi_scale)))
        thresh_f = ttk.Frame(param_frame)
        thresh_f.pack(fill=tk.X, pady=(0, int(10 * self.dpi_scale)))
        self.threshold_entry_var = tk.StringVar(value=f"{self.change_threshold * 100:.1f}")
        self.threshold_scale = ttk.Scale(thresh_f, from_=0.1, to=50,
                                        value=self.change_threshold*100,
                                        command=self.update_threshold)
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=(0, int(6 * self.dpi_scale)))
        self.threshold_label = ttk.Label(thresh_f,
                                        text=f"{self.change_threshold*100:.1f}%",
                                        width=int(6 * self.dpi_scale),
                                        font=('Consolas', self.scaled_font_size))
        self.threshold_label.pack(side=tk.RIGHT)
        self.threshold_entry = ttk.Entry(thresh_f, textvariable=self.threshold_entry_var,
                                        width=int(8 * self.dpi_scale),
                                        font=('Microsoft YaHei UI', self.scaled_font_size))
        self.threshold_entry.pack(side=tk.RIGHT, padx=(int(6 * self.dpi_scale), 0))
        self.threshold_entry.bind("<Return>", self.validate_threshold_input)
        
        ttk.Label(param_frame, text="截图最小间隔 (秒)",
                 font=('Microsoft YaHei UI', self.scaled_font_size, 'bold')).pack(anchor=tk.W,
                 pady=(0, int(3 * self.dpi_scale)))
        intv_f = ttk.Frame(param_frame)
        intv_f.pack(fill=tk.X, pady=(0, int(10 * self.dpi_scale)))
        self.interval_entry_var = tk.StringVar(value=f"{self.min_interval:.1f}")
        self.interval_scale = ttk.Scale(intv_f, from_=0.1, to=10, value=self.min_interval,
                                       command=self.update_interval)
        self.interval_scale.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                padx=(0, int(6 * self.dpi_scale)))
        self.interval_label = ttk.Label(intv_f, text=f"{self.min_interval:.1f}s",
                                       width=int(6 * self.dpi_scale),
                                       font=('Consolas', self.scaled_font_size))
        self.interval_label.pack(side=tk.RIGHT)
        self.interval_entry = ttk.Entry(intv_f, textvariable=self.interval_entry_var,
                                       width=int(8 * self.dpi_scale),
                                       font=('Microsoft YaHei UI', self.scaled_font_size))
        self.interval_entry.pack(side=tk.RIGHT, padx=(int(6 * self.dpi_scale), 0))
        self.interval_entry.bind("<Return>", self.validate_interval_input)

        ttk.Label(param_frame, text="压缩视频处理分辨率",
                 font=('Microsoft YaHei UI', self.scaled_font_size, 'bold')).pack(anchor=tk.W,
                 pady=(0, int(3 * self.dpi_scale)))
        h_f = ttk.Frame(param_frame)
        h_f.pack(fill=tk.X, pady=(0, int(10 * self.dpi_scale)))
        self.target_height_scale = ttk.Scale(h_f, from_=240, to=1080, variable=self.target_height_var,
                                            command=self.update_target_height)
        self.target_height_scale.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                     padx=(0, int(6 * self.dpi_scale)))
        self.target_height_label = ttk.Label(h_f, text="480", width=int(6 * self.dpi_scale),
                                            font=('Consolas', self.scaled_font_size))
        self.target_height_label.pack(side=tk.RIGHT)
        self.target_height_entry = ttk.Entry(h_f, textvariable=self.target_height_var,
                                            width=int(8 * self.dpi_scale),
                                            font=('Microsoft YaHei UI', self.scaled_font_size))
        self.target_height_entry.pack(side=tk.RIGHT, padx=(int(6 * self.dpi_scale), 0))
        self.target_height_entry.bind("<Return>", self.validate_target_height_input)
    
    def _create_save_frame(self, parent, row, pady, inner_pad):
        save_frame = ttk.LabelFrame(parent, text="截图保存",
                                   padding=(inner_pad, int(10 * self.dpi_scale)))
        save_frame.grid(row=row, column=0, sticky="ew", pady=pady,
                       padx=int(8 * self.dpi_scale))
        
        ttk.Label(save_frame, text="保存路径:",
                 font=('Microsoft YaHei UI', self.scaled_font_size, 'bold')).pack(anchor=tk.W,
                 pady=(0, int(3 * self.dpi_scale)))
        path_f1 = ttk.Frame(save_frame)
        path_f1.pack(fill=tk.X, pady=(0, int(6 * self.dpi_scale)))
        self.save_path_var = tk.StringVar(value=self.save_path)
        self.save_path_entry = ttk.Entry(path_f1, textvariable=self.save_path_var,
                                        font=('Microsoft YaHei UI', self.scaled_font_size))
        self.save_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=(0, int(4 * self.dpi_scale)))
        self.save_path_btn = ttk.Button(path_f1, text="更改", command=self.change_save_path,
                                       width=int(6 * self.dpi_scale))
        self.save_path_btn.pack(side=tk.RIGHT)
    
    def _create_control_tab(self, notebook):
        control_frame_nb = ttk.Frame(notebook)
        notebook.add(control_frame_nb, text="处理控制")
        control_frame_nb.grid_rowconfigure(4, weight=1)
        control_frame_nb.grid_columnconfigure(0, weight=1)
        inner_pad = int(15 * self.dpi_scale)
        
        status_progress_frame = ttk.LabelFrame(control_frame_nb, text="状态与进度",
                                              padding=(inner_pad, int(10 * self.dpi_scale)))
        status_progress_frame.grid(row=0, column=0, sticky="ew",
                                  padx=int(8 * self.dpi_scale),
                                  pady=(int(8 * self.dpi_scale), int(8 * self.dpi_scale)))
        self.status_var = tk.StringVar(value=f"就绪 | 当前倍速：{self.current_speed}倍")
        status_label = ttk.Label(status_progress_frame, textvariable=self.status_var,
                                font=('Microsoft YaHei UI', self.scaled_font_size, 'bold'),
                                foreground=COLORS['success'])
        status_label.pack(anchor=tk.W, pady=(0, int(6 * self.dpi_scale)))
        
        self.progress_label = ttk.Label(status_progress_frame, text="等待开始处理...",
                                       font=('Microsoft YaHei UI', self.scaled_font_size))
        self.progress_label.pack(anchor=tk.W, pady=(0, int(6 * self.dpi_scale)))
        
        progress_row = ttk.Frame(status_progress_frame)
        progress_row.pack(fill=tk.X, pady=(0, int(3 * self.dpi_scale)))
        ttk.Label(progress_row, text="总进度:",
                 font=('Microsoft YaHei UI', self.scaled_font_size, 'bold')).pack(side=tk.LEFT)
        self.total_percent_label = ttk.Label(progress_row, text="0.0%",
                                            font=('Consolas', self.scaled_font_size, 'bold'),
                                            foreground=COLORS['success'])
        self.total_percent_label.pack(side=tk.LEFT, padx=int(6 * self.dpi_scale))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_progress_frame, variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=int(6 * self.dpi_scale))
        
        speed_frame = ttk.LabelFrame(control_frame_nb, text="处理速度控制",
                                    padding=(inner_pad, int(10 * self.dpi_scale)))
        speed_frame.grid(row=1, column=0, sticky="ew", padx=int(8 * self.dpi_scale),
                        pady=int(8 * self.dpi_scale))
        speed_info_frame = ttk.Frame(speed_frame)
        speed_info_frame.pack(fill=tk.X, pady=(0, int(6 * self.dpi_scale)))
        self.speed_label = ttk.Label(speed_info_frame,
                                    text=f"当前：{self.current_speed}x",
                                    font=('Microsoft YaHei UI', self.scaled_font_size, 'bold'))
        self.speed_label.pack(side=tk.LEFT)
        
        speed_btn_frame = ttk.Frame(speed_frame)
        speed_btn_frame.pack(fill=tk.X)
        cols = 4
        for i, s in enumerate(self.speed_levels):
            btn = ttk.Button(speed_btn_frame, text=f"{s}x",
                            command=lambda s=s: self.set_speed(s),
                            width=int(5 * self.dpi_scale))
            btn.grid(row=i//cols, column=i%cols, padx=int(3 * self.dpi_scale),
                    pady=int(3 * self.dpi_scale), sticky="ew")
            self.speed_buttons.append(btn)
        for i in range(cols):
            speed_btn_frame.grid_columnconfigure(i, weight=1)
        
        control_buttons_frame = ttk.LabelFrame(control_frame_nb, text="控制命令",
                                              padding=(inner_pad, int(10 * self.dpi_scale)))
        control_buttons_frame.grid(row=2, column=0, sticky="ew",
                                  padx=int(8 * self.dpi_scale),
                                  pady=int(8 * self.dpi_scale))
        btn_frame = ttk.Frame(control_buttons_frame)
        btn_frame.pack(fill=tk.X, pady=int(6 * self.dpi_scale))
        self.control_btn = ttk.Button(btn_frame, text="开始处理",
                                     command=self.on_control_click, style="Accent.TButton")
        self.control_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                             padx=int(3 * self.dpi_scale))
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop_processing,
                                  style="Danger.TButton")
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                          padx=int(3 * self.dpi_scale))
        
        bg_mode_frame = ttk.LabelFrame(control_frame_nb, text="处理选项",
                                      padding=(inner_pad, int(10 * self.dpi_scale)))
        bg_mode_frame.grid(row=3, column=0, sticky="ew", padx=int(8 * self.dpi_scale),
                          pady=int(8 * self.dpi_scale))
        ttk.Checkbutton(bg_mode_frame, text="后台模式（关闭预览以加速）",
                       variable=self.background_mode_var,
                       command=self.toggle_background_mode).pack(anchor=tk.W,
                       padx=int(6 * self.dpi_scale))
    
    def _create_help_tab(self, notebook):
        help_frame = ttk.Frame(notebook)
        notebook.add(help_frame, text="使用帮助")
        
        help_text_widget = tk.Text(help_frame, wrap=tk.WORD,
                                  font=('Microsoft YaHei UI', self.scaled_font_size),
                                  bg="#fafafa", relief=tk.FLAT,
                                  padx=int(15 * self.dpi_scale),
                                  pady=int(15 * self.dpi_scale))
        help_scrollbar = ttk.Scrollbar(help_frame, orient="vertical",
                                      command=help_text_widget.yview)
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
      • 点【确认选择】：确认关注（ROI）区域（需 ≥3 个点）
  - ROI 将以绿色多边形显示在预览画面

③ 配置参数（【参数设置】标签页）
  - 【画面变化阈值】：默认 5%。值越小越敏感。
  - 【截图最小间隔】：默认 1 秒，避免连续截图。
  - 【GMM 敏感度】：默认 25。
     值越大，越不敏感（可减少“车走后地面误报”）。
  - 【帧间差分阈值】：默认 30。
     值越大，越忽略微小运动（可抑制噪点）。
  - 【压缩视频处理分辨率】：默认 480 。
     值越小处理越快（480就是按比例压缩至480P处理）。

④ 设置保存路径
  - 默认路径：程序目录下的“变化截图”文件夹
  - 点击【更改】可自定义路径（需有写入权限）

⑤ 开始处理
  - 切换到【处理控制】标签页
  - 选择处理倍速（1x 最精准，高倍速会跳帧加速但可能漏检）
  - 点击【开始处理】
  - 支持【暂停】/【停止】操作

⑥ 查看结果
  - 截图自动保存为：`视频名_frame_帧号.jpg`
  - 日志文件位于“检测日志”文件夹，记录所有操作与错误

如有问题，请联系开发者：geckotao@hotmail.com
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
    
    def _create_preview_area(self, padx, pady):
        right_frame = ttk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky="nsew",
                        padx=(padx//2, padx), pady=pady)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        display_frame = ttk.LabelFrame(right_frame, text="视频预览",
                                      padding=int(10 * self.dpi_scale))
        display_frame.grid(row=0, column=0, sticky="nsew")
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        
        preview_container = tk.Frame(display_frame, borderwidth=1, relief=tk.SOLID,
                                    bg=COLORS['border_light'])
        preview_container.grid(row=0, column=0, sticky="nsew",
                              padx=int(3 * self.dpi_scale),
                              pady=int(3 * self.dpi_scale))
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)
        
        self.video_label = tk.Label(preview_container, background="#000000")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        preview_container.bind("<Configure>", self.on_preview_resize)
        
        info_bar = tk.Frame(right_frame, height=int(35 * self.dpi_scale),
                           bg=COLORS['bg_card'])
        info_bar.grid(row=1, column=0, sticky="ew",
                     pady=(int(6 * self.dpi_scale), 0))
        info_bar.grid_columnconfigure(0, weight=1)
        self.info_label = tk.Label(info_bar,
                                  text="就绪 - 请添加视频文件并开始处理",
                                  font=('Microsoft YaHei UI',
                                       max(9, int(9 * self.dpi_scale))),
                                  bg=COLORS['bg_card'],
                                  fg=COLORS['text_main'])
        self.info_label.pack(side=tk.LEFT, padx=int(6 * self.dpi_scale))
    
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
                self.log_message(f"处理 UI 任务时出错：{str(e)}")
        
        if self.root.winfo_exists():
            self.root.after(50, self.process_ui_queue)
    
    def safe_ui_call(self, func, *args, **kwargs):
        self.ui_queue.put(lambda: func(*args, **kwargs))
    
    def on_preview_resize(self, event=None):
        if not event or event.width < 50 or event.height < 50:
            return
        if hasattr(self, '_last_preview_size') and (event.width, event.height) == self._last_preview_size:
            return
        self._last_preview_size = (event.width, event.height)
        frame_to_display = getattr(self, '_last_displayed_frame', None)
        if frame_to_display is not None:
            rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
            self.display_frame(rgb)
    
    def display_frame(self, frame: np.ndarray):
        if frame is None:
            self.video_label.config(image='', text="无视频预览", fg=COLORS['text_disable'])
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
    
    def on_control_click(self):
        if self.processing and self.paused:
            self.progress_label.config(text="正在检查参数和 ROI 有无更改，请稍候...")
            self.root.update_idletasks()
            self._pause_event.clear()
            self._reload_roi_and_params()
            self.paused = False
            self.control_btn.config(text="暂停", style="TButton")
            self._disable_non_control_widgets()
            self.stop_btn.config(state="normal")
            self.progress_label.config(text="继续处理中（已重载参数和 ROI）...")
            self.status_var.set(f"处理中")
            self.log_message("继续视频处理（已重载参数和 ROI）")
            self.info_label.config(text="继续视频处理")
        elif self.processing and not self.paused:
            self.paused = True
            self._pause_event.set()
            self.control_btn.config(text="继续处理")
            self._enable_non_control_widgets()
            self.stop_btn.config(state="normal")
            self.progress_label.config(text="已暂停，可修改参数后继续")
            self.status_var.set(f"已暂停")
            self.log_message("暂停视频处理")
            self.info_label.config(text="视频处理已暂停")
        else:
            self._start_new_processing()
    
    def _reload_roi_and_params(self):
        self._proc_prev_gray = None
        if self.roi_selected:
            frame_width, frame_height = None, None
            if self.cap and self.cap.isOpened():
                pos_before = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                with self.cap_lock:
                    ret, temp_frame = self.cap.read()
                    if ret and temp_frame is not None:
                        target_h = self.target_height_var.get()
                        if target_h > 0 and temp_frame.shape[0] > target_h:
                            scale = target_h / temp_frame.shape[0]
                            new_w = int(temp_frame.shape[1] * scale)
                            new_h = target_h
                            temp_frame = cv2.resize(temp_frame, (new_w, new_h),
                                                   interpolation=cv2.INTER_AREA)
                        frame_height, frame_width = temp_frame.shape[:2]
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos_before)
            else:
                if self.preview_frame is not None:
                    h, w = self.preview_frame.shape[:2]
                    target_h = self.target_height_var.get()
                    if target_h > 0 and h > target_h:
                        scale = target_h / h
                        frame_width = int(w * scale)
                        frame_height = target_h
                    else:
                        frame_width, frame_height = w, h
            
            if frame_width and frame_height:
                try:
                    self.create_roi_mask(target_size=(frame_width, frame_height))
                except Exception as e:
                    self.log_message(f"重建 ROI 失败：{e}")
                    self.processing_roi_mask = None
                    self.safe_ui_call(messagebox.showwarning, "警告",
                                     "ROI 重建失败，将使用全画面检测")
            else:
                self.processing_roi_mask = None
        else:
            self.processing_roi_mask = None
    
    def _start_new_processing(self):
        self._release_capture()
        if not self.video_paths:
            messagebox.showwarning("警告", "请先添加视频文件")
            return
        
        try:
            test_file = os.path.join(self.save_path, f"test_{int(time.time())}.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            messagebox.showerror("路径错误",
                                f"保存路径不可写:\n{self.save_path}\n{str(e)}")
            self.log_message(f"保存路径不可写：{str(e)}")
            return
        
        self.processing = True
        self.paused = False
        self._stop_event.clear()
        self._pause_event.clear()
        self.current_video_index = 0
        self._proc_prev_gray = None
        
        self.control_btn.config(text="暂停", style="Accent.TButton")
        self.stop_btn.config(state="normal")
        self._disable_non_control_widgets()
        self.status_var.set(f"处理中")
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

    def validate_target_height_input(self, event=None):
        try:
            v = int(self.target_height_var.get())
            if 240 <= v <= 2160:
                self.target_height_scale.set(v)
                self.target_height_label.config(text=str(v))
                self.log_message(f"处理分辨率高度设置为 {v}")
            else:
                raise ValueError
        except ValueError:
            messagebox.showwarning("输入错误", "请输入 240 ~ 2160 之间的有效整数")
            self.target_height_var.set(480)
            self.target_height_label.config(text="480")
    
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

    def update_target_height(self, value):
        v = int(float(value))
        self.target_height_label.config(text=str(v))
        # 同步 Entry 显示
        self.target_height_var.set(v)
        self.log_message(f"处理分辨率高度设置为 {v}")
    
    # ==================== ROI 相关 ====================
    def select_roi(self):
        if self.roi_selected:
            self.cancel_roi()
            return
        
        if self.processing and not self.paused:
            messagebox.showinfo("提示", "请先暂停或停止视频处理，再选择 ROI 区域")
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
        roi_window.configure(bg=COLORS['bg_main'])
        
        canvas = tk.Canvas(roi_window, bg="#000000", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        pil_img = Image.fromarray(cv2.cvtColor(self.preview_frame,
                                               cv2.COLOR_BGR2RGB))
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
            scaled = [(x * self.roi_scale_factor, y * self.roi_scale_factor)
                     for x, y in roi_points]
            for i, (x, y) in enumerate(scaled):
                circles.append(canvas.create_oval(x-4, y-4, x+4, y+4, fill="#ff4d4f", outline="white"))
                if i > 0:
                    lines.append(canvas.create_line(scaled[i-1][0], scaled[i-1][1],
                                                   x, y, fill="#52c41a", width=3))
            if len(roi_points) > 2:
                lines.append(canvas.create_line(scaled[-1][0], scaled[-1][1],
                                               scaled[0][0], scaled[0][1],
                                               fill="#52c41a", width=3))
        
        def update_image():
            new_size = (int(self.roi_original_size[0] * self.roi_scale_factor),
                       int(self.roi_original_size[1] * self.roi_scale_factor))
            resized_img = self.roi_image.resize(new_size, Image.LANCZOS)
            self.roi_photo = ImageTk.PhotoImage(resized_img)
            canvas.config(scrollregion=(0, 0, new_size[0], new_size[1]))
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=self.roi_photo)
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
        
        btn_frame = tk.Frame(roi_window, bg=COLORS['bg_card'])
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        def confirm():
            if len(roi_points) >= 3:
                self.roi_points = roi_points
                self.roi_selected = True
                self.roi_button.config(text="取消选取", command=self.cancel_roi)
                self.create_roi_mask()
                roi_window.destroy()
                self.log_message(f"已选择 ROI 区域，{len(roi_points)}个顶点")
                self.info_label.config(text=f"已选择 ROI 区域，{len(roi_points)}个顶点")
            else:
                messagebox.showerror("错误", "至少需要 3 个点才能构成多边形")
        
        def cancel():
            roi_window.destroy()
        
        tk.Button(btn_frame, text="确认选择", command=confirm,
                 bg=COLORS['primary'], fg="white",
                 font=('Microsoft YaHei UI', self.scaled_font_size, 'bold'),
                 padx=int(15 * self.dpi_scale), pady=int(5 * self.dpi_scale),
                 relief=tk.FLAT, cursor="hand2").pack(side=tk.RIGHT, padx=(5,0))
        tk.Button(btn_frame, text="取消", command=cancel,
                 bg=COLORS['border'], fg=COLORS['text_main'],
                 font=('Microsoft YaHei UI', self.scaled_font_size),
                 padx=int(15 * self.dpi_scale), pady=int(5 * self.dpi_scale),
                 relief=tk.FLAT, cursor="hand2").pack(side=tk.RIGHT)
    
    def cancel_roi(self):
        self.roi_selected = False
        self.roi_points = []
        self.roi_mask = None
        self.processing_roi_mask = None
        self.roi_button.config(text="选择 ROI 区域", command=self.select_roi)
        self.roi_status_label.config(text="未选取关注区域", foreground=COLORS['text_sec'])
        if self.preview_frame is not None:
            self.display_frame(cv2.cvtColor(self.preview_frame, cv2.COLOR_BGR2RGB))
        self.log_message("已取消 ROI 区域选择")
        self.info_label.config(text="已取消 ROI 区域选择", foreground=COLORS['text_sec'])
    
    def create_roi_mask(self, target_size: Optional[Tuple[int, int]] = None):
        if not self.roi_selected or not self.roi_points or self.preview_frame is None:
            self.roi_mask = None
            self.processing_roi_mask = None
            return
        try:
            h, w = self.preview_frame.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            pts = np.array([[int(x), int(y)] for x, y in self.roi_points], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            self.roi_mask = mask
            if target_size is not None:
                self.processing_roi_mask = cv2.resize(mask, target_size,
                                                     interpolation=cv2.INTER_NEAREST)
            else:
                self.processing_roi_mask = mask
            area_ratio = cv2.countNonZero(mask) / mask.size * 100
            self.roi_status_label.config(text=f"已选取 ROI 区域 ({area_ratio:.1f}% 画面)",
                                        foreground=COLORS['danger'])
            self.log_message(f"ROI 面积占比：{area_ratio:.1f}%")
        except Exception as e:
            self.log_message(f"创建 ROI 掩码失败：{str(e)}")
            messagebox.showerror("错误", f"创建 ROI 掩码失败：{str(e)}")
            self.roi_mask = None
            self.processing_roi_mask = None
    
    # ==================== 视频处理 ====================
    def safe_video_capture(self, video_path: str) -> Tuple[Optional[cv2.VideoCapture],
                                                           Optional[str]]:
        if not os.path.exists(video_path):
            return None, f"视频文件不存在：{video_path}"
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
        return None, f"无法打开视频：{os.path.basename(video_path)}"
    
    def process_videos(self):
        try:
            current_index = self.current_video_index
            while current_index < len(self.video_paths):
                if self._stop_event.is_set():
                    break
                
                video_path = self.video_paths[current_index]
                self.safe_ui_call(self.progress_label.config,
                                 text=f"正在处理第 {current_index + 1} 个视频")
                self.safe_ui_call(self.status_var.set, "处理中")
                self.log_message(f"开始处理视频：{video_path}")
                
                self._release_capture()
                self.cap, err = self.safe_video_capture(video_path)
                if err:
                    self.safe_ui_call(messagebox.showerror, "错误", err)
                    self.log_message(err)
                    if not self._stop_event.is_set():
                        current_index += 1
                        self.current_video_index = current_index
                    continue
                
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    total_frames = 1
                
                with self.cap_lock:
                    ret, first_frame = self.cap.read()
                    if not ret:
                        self.log_message(f"无法读取首帧：{video_path}")
                        if not self._stop_event.is_set():
                            current_index += 1
                            self.current_video_index = current_index
                        continue
                    
                    frame_height, frame_width = first_frame.shape[:2]
                    target_h = self.target_height_var.get()
                    if target_h > 0 and frame_height > target_h:
                        scale = target_h / frame_height
                        new_w = int(frame_width * scale)
                        new_h = target_h
                        first_frame = cv2.resize(first_frame, (new_w, new_h),
                                                interpolation=cv2.INTER_AREA)
                        frame_height, frame_width = first_frame.shape[:2]
                    
                    self._current_process_size = (frame_width, frame_height)
                    if self.roi_selected:
                        self.create_roi_mask(target_size=(frame_width, frame_height))
                    else:
                        self.processing_roi_mask = None
                    
                    processor = VideoProcessor(
                        gmm_var=self.gmm_var.get(),
                        fd_var=self.fd_var.get(),
                        roi_mask=self.processing_roi_mask
                    )
                    _, gray_first = processor.preprocess_frame(first_frame, self.target_height_var.get())
                    _ = processor.detect_change(gray_first)
                
                with self.cap_lock:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                frame_id = 0
                last_saved_time = 0
                
                while frame_id < total_frames:
                    if self._stop_event.is_set():
                        break
                    if self.paused:
                        self._pause_event.wait(timeout=0.05)
                        continue
                    
                    if self.current_speed > 1:
                        for _ in range(self.current_speed - 1):
                            with self.cap_lock:
                                ret, _ = self.cap.read()
                                if not ret: break
                            if self._stop_event.is_set(): break
                    
                    if self._stop_event.is_set():
                        break
                    
                    with self.cap_lock:
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                    
                    original_frame, gray = processor.preprocess_frame(frame, self.target_height_var.get())
                    valid_change, fg_mask, change_ratio = processor.detect_change(gray)
                    
                    now = time.time()
                    if valid_change and (now - last_saved_time) > self.min_interval:
                        last_saved_time = now
                        video_basename = os.path.splitext(os.path.basename(video_path))[0]
                        self.save_screenshot(original_frame.copy(), video_basename, frame_id)
                    
                    if not self.background_mode_var.get():
                        now_time = time.time()
                        if not hasattr(self, '_last_preview_update_time'):
                            self._last_preview_update_time = now_time
                        if now_time - self._last_preview_update_time >= PREVIEW_UPDATE_INTERVAL:
                            color_mask = np.zeros_like(original_frame)
                            color_mask[:, :, 2] = fg_mask
                            marked = cv2.addWeighted(original_frame, 1, color_mask, 0.5, 0)
                            if self.roi_selected and self.roi_points:
                                pts = np.array(self.roi_points, np.int32)
                                cv2.polylines(marked, [pts], True, (0, 255, 0), 2)
                            cv2.putText(marked,
                                       f"Change: {change_ratio * 100:.1f}% | Speed: {self.current_speed}x",
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (0, 0, 255), 1)
                            rgb_marked = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
                            self.safe_ui_call(self.display_frame, rgb_marked)
                            self._last_preview_update_time = now_time
                    
                    overall_progress = ((current_index + (frame_id + 1) / total_frames)
                                       / len(self.video_paths)) * 100
                    self.safe_ui_call(self.total_percent_label.config,
                                     text=f"{overall_progress:.1f}%")
                    self.safe_ui_call(self.progress_var.set, overall_progress)
                    self.safe_ui_call(self.progress_label.config,
                                     text=f"以{self.current_speed}x 处理第{current_index + 1}个 [{frame_id + 1}/{total_frames}]|共{len(self.video_paths)}个视频")
                    
                    frame_id += self.current_speed
                
                self._release_capture()
                if not self._stop_event.is_set():
                    current_index += 1
                    self.current_video_index = current_index
                else:
                    break
            
            if not self._stop_event.is_set():
                self.safe_ui_call(self.progress_label.config, text="所有视频处理完毕")
                self.safe_ui_call(self.status_var.set, "处理完成")
                self.safe_ui_call(self.progress_var.set, 100)
                self.safe_ui_call(self._stop_cleanup)
                self.log_message(f"所有 {len(self.video_paths)} 个视频处理完成")
                self.safe_ui_call(messagebox.showinfo, "完成", "所有视频处理已完成")
        
        except Exception as e:
            self.processing = False
            error_detail = traceback.format_exc()
            error_msg = f"处理出错：{str(e)}\n详细信息：{error_detail}"
            self.log_message(error_msg)
            self.safe_ui_call(messagebox.showerror, "错误", error_msg)
        finally:
            self._release_capture()
    
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
        self._release_capture()
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
                target_h = self.target_height_var.get()
                if target_h > 0 and frame.shape[0] > target_h:
                    scale = target_h / frame.shape[0]
                    new_w = int(frame.shape[1] * scale)
                    frame = cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)
                
                self.display_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.progress_label.config(text=f"预览：{os.path.basename(video_path)}")
                self.info_label.config(text=f"预览中：{os.path.basename(video_path)}")
                self.log_message(f"预览视频：{os.path.basename(video_path)}")
            else:
                messagebox.showerror("错误",
                                    f"无法读取视频帧：{os.path.basename(video_path)}")
        
        self._release_capture()
    
    def add_videos(self):
        paths = filedialog.askopenfilenames(title="选择视频文件",
                                           filetypes=[("视频文件",
                                                      "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.mpeg *.mpg *.m1v *.m2v *.vob *.ts *.m2ts *.mts")])
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
        self._release_capture()
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
                self.log_message(f"保存路径更改至：{new_path}")
                self.info_label.config(text=f"保存路径：{os.path.basename(new_path)}")
            except Exception as e:
                messagebox.showwarning("路径警告", f"所选路径不可写:\n{str(e)}")
                self.log_message(f"路径不可写：{str(e)}")
    
    def set_speed(self, speed: int):
        if speed in self.speed_levels:
            self.current_speed = speed
            self.speed_label.config(text=f"当前：{speed}x")
            self.status_var.set(f"{self.status_var.get().split('|')[0].strip()} | 倍速：{speed}x")
            self.log_message(f"处理倍速设置为 {speed}倍")
            self.info_label.config(text=f"处理倍速：{speed}倍")
    
    def _stop_cleanup(self):
        self.processing = False
        self.paused = False
        self._stop_event.set()
        self._pause_event.set()
        self.current_video_index = 0
        self._proc_prev_gray = None
        
        for w in self.parameter_widgets + self.file_widgets:
            try:
                w.config(state='normal')
            except:
                pass
        self.video_listbox.config(state=tk.NORMAL)
        
        self.control_btn.config(text="开始处理", style="Accent.TButton",
                               state="normal")
        self.stop_btn.config(state="disabled")
        self.control_btn.focus_set()
        
        self.status_var.set(f"已停止")
        self.progress_label.config(text="处理已停止")
        self.progress_var.set(0)
        self.info_label.config(text="处理已停止")
        
        self.root.after(50, lambda: self.control_btn.config(text="开始处理",
                                                           style="Accent.TButton"))
    
    def stop_processing(self):
        if not (self.processing or self.paused):
            return
        self._stop_event.set()
        self.log_message("处理已停止")
        self.root.after(100, self._stop_cleanup)
    
    def save_screenshot(self, frame: np.ndarray, video_basename: str,
                       frame_num: int) -> Optional[str]:
        try:
            safe_name = "".join(c for c in video_basename
                               if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}_frame_{frame_num}.jpg"
            full_path = os.path.join(self.save_path, filename)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(full_path, quality=95)
            self.log_message(f"截图已保存：{full_path}")
            self.info_label.config(text=f"已保存截图：{filename}")
            return full_path
        except Exception as e:
            self.log_message(f"保存截图失败：{str(e)}")
            messagebox.showerror("保存错误", f"保存截图失败:\n{str(e)}")
            self.info_label.config(text="截图保存失败")
            return None


if __name__ == "__main__":
    root = tk.Tk()
    app = GMMVideoDetector(root)
    try:
        root.mainloop()
    finally:
        app._close_log_file()
        app._release_capture()
