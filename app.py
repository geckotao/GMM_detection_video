import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
import time
from datetime import datetime
from PIL import Image, ImageTk
import tempfile
import queue
import ctypes

class GMMVideoDetector:
    def ensure_directory_exists(self, path):
        if os.path.exists(path):
            return
        for attempt in range(3):
            try:
                os.makedirs(path, exist_ok=True)
                if hasattr(self, 'log_file_path'):
                    self.log_message(f"创建目录: {path}")
                else:
                    print(f"创建目录: {path}")
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.1)
                else:
                    error_msg = f"创建目录失败 {path}: {str(e)}"
                    print(error_msg)
                    messagebox.showerror("错误", error_msg)
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
            messagebox.showerror("错误", error_msg)
            self.log_file_path = None

    def log_message(self, message):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        print(log_entry.strip())

        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
            except Exception as e:
                print(f"写入日志失败: {str(e)}")

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

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(50, self.process_ui_queue)

    def safe_ui_call(self, func, *args, **kwargs):
        self.ui_queue.put(lambda: func(*args, **kwargs))

    def __init__(self, root):
        self.root = root
        self.root.title("视频画面变化检测 by geckotao")
        self.dpi_scale = self.get_dpi_scale()
        self.base_font_size = 10 
        self.scaled_font_size = int(self.base_font_size * self.dpi_scale)
        self.root.geometry(f"{int(1200 * self.dpi_scale)}x{int(700 * self.dpi_scale)}")
        self.root.minsize(int(1024 * self.dpi_scale), int(700 * self.dpi_scale))
        self.root.configure(bg="#f0f0f0")
        self.setup_dpi_awareness()
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", 
                            background="#f0f0f0", 
                            font=("SimHei", self.scaled_font_size))
        self.style.configure("TButton", 
                            font=("SimHei", self.scaled_font_size), 
                            padding=int(5 * self.dpi_scale))
        self.style.configure("TEntry", 
                            font=("SimHei", self.scaled_font_size), 
                            padding=int(3 * self.dpi_scale)) 
        self.style.map("TEntry",
                    font=[("focus", ("SimHei", self.scaled_font_size)),
                            ("disabled", ("SimHei", self.scaled_font_size))])
        self.style.configure("TNotebook.Tab", 
                            font=("SimHei", self.scaled_font_size),
                            padding=(int(10 * self.dpi_scale), int(5 * self.dpi_scale)))
        self.style.configure("TLabelframe", 
                            background="#f0f0f0", 
                            borderwidth=1)
        self.style.configure("TLabelframe.Label", 
                            background="#f0f0f0", 
                            font=("SimHei", self.scaled_font_size, "bold"), 
                            padding=(int(5 * self.dpi_scale), int(2 * self.dpi_scale)))
        self.style.configure("Accent.TButton", 
                            font=("SimHei", self.scaled_font_size, "bold"), 
                            background="#4a90e2", 
                            foreground="white")
        self.style.map("Accent.TButton", 
                    background=[("active", "#357abd"), ("pressed", "#2a5f90")])
        
        self.log_dir = os.path.join(os.getcwd(), "检测日志")
        self.ensure_directory_exists(self.log_dir)
        self.log_file = None
        self.init_log_file()
        
        self.video_paths = []
        self.current_video_index = 0
        self.processing = False
        self.paused = False
        self.roi_selected = False
        self.roi_points = []
        self.gmm = None
        self.change_threshold = 0.05
        self.save_path = os.path.join(os.getcwd(), "变化截图")
        self.backup_save_path = os.path.join(tempfile.gettempdir(), "变化截图备份")
        self.cap = None
        self.last_change_time = 0
        self.min_interval = 1
        self.preview_frame = None
        self.roi_mask = None
        self.speed_levels = [1, 2, 4, 8, 16, 24, 32, 64]
        self.current_speed = 1
        self.speed_lock = threading.Lock()

        self.target_height = 480

        self.ui_queue = queue.Queue()
        self.root.after(50, self.process_ui_queue)

        self.ensure_directory_exists(self.save_path)
        self.ensure_directory_exists(self.backup_save_path)

        try:
            icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
            if os.path.exists(icon_path):
                icon_img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, icon_img)
                self.log_message(f"成功设置窗口图标: {icon_path}")
        except Exception as e:
            self.log_message(f"设置窗口图标失败: {str(e)}")

        self.create_widgets()
        self.log_message("程序启动")
        self.log_message(f"主保存路径: {self.save_path}")
        self.log_message(f"备份保存路径: {self.backup_save_path}")
        self.log_message(f"初始处理倍速: {self.current_speed}倍")
        self.log_message(f"目标处理分辨率: 高度 ≤ {self.target_height}P")
        self.log_message(f"DPI缩放因子: {self.dpi_scale:.2f}")

    def get_dpi_scale(self):
        try:
            if os.name == 'nt':

                user32 = ctypes.windll.user32
                user32.SetProcessDPIAware()
                dpi = user32.GetDpiForSystem()
                return dpi / 96.0  
            else:

                screen_width = self.root.winfo_screenwidth()
                if screen_width >= 3840:  
                    return 2.0
                elif screen_width >= 2560:  
                    return 1.5
                else: 
                    return 1.0
        except:
            return 1.0

    def setup_dpi_awareness(self):
        try:
            if os.name == 'nt':
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except:
            pass

    def create_widgets(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        padx = int(8 * self.dpi_scale)
        pady = int(8 * self.dpi_scale)
        
        left_frame = ttk.Frame(self.root, width=int(365 * self.dpi_scale))
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(padx, padx//2), pady=pady)
        left_frame.grid_propagate(False)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        control_notebook = ttk.Notebook(left_frame)
        control_notebook.grid(row=0, column=0, sticky="nsew", padx=int(5 * self.dpi_scale), pady=int(5 * self.dpi_scale))

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

        file_frame = ttk.LabelFrame(settings_scrollable_frame, text="视频文件管理", padding=(inner_pad, int(8 * self.dpi_scale)))
        file_frame.grid(row=row, column=0, sticky="ew", pady=frame_pady, padx=int(5 * self.dpi_scale))
        settings_scrollable_frame.columnconfigure(0, weight=1)
        row += 1

        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=tk.X, pady=(0, int(8 * self.dpi_scale)))
        ttk.Button(btn_row, text="添加视频文件", command=self.add_videos).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(5 * self.dpi_scale)))
        ttk.Button(btn_row, text="清空列表", command=self.clear_videos).pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        ttk.Button(action_btn_frame, text="移除", command=self.remove_video).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(3 * self.dpi_scale)))
        ttk.Button(action_btn_frame, text="预览", command=self.preview_selected_video).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(int(3 * self.dpi_scale), 0))

        roi_frame = ttk.LabelFrame(settings_scrollable_frame, text="关注区域 (ROI)", padding=(inner_pad, int(8 * self.dpi_scale)))
        roi_frame.grid(row=row, column=0, sticky="ew", pady=frame_pady, padx=int(5 * self.dpi_scale))
        row += 1
        self.roi_button = ttk.Button(roi_frame, text="选择ROI区域", command=self.select_roi)
        self.roi_button.pack(fill=tk.X, pady=(0, int(5 * self.dpi_scale)))
        self.roi_status_label = ttk.Label(roi_frame, text="未选取关注区域", foreground="gray")
        self.roi_status_label.pack(anchor=tk.W, pady=(0, int(2 * self.dpi_scale)))
        ttk.Label(roi_frame, text="操作: 左键加点，右键删点，R重置，滚轮缩放，关闭窗口确认。", 
                 font=("TkDefaultFont", max(8, int(8 * self.dpi_scale))), foreground="#666666").pack(anchor=tk.W)

        param_frame = ttk.LabelFrame(settings_scrollable_frame, text="检测参数", padding=(inner_pad, int(8 * self.dpi_scale)))
        param_frame.grid(row=row, column=0, sticky="ew", pady=frame_pady, padx=int(5 * self.dpi_scale))
        row += 1

        ttk.Label(param_frame, text="画面变化阈值 (%)").pack(anchor=tk.W, pady=(0, int(2 * self.dpi_scale)))
        thresh_f = ttk.Frame(param_frame)
        thresh_f.pack(fill=tk.X, pady=(0, int(8 * self.dpi_scale)))
        self.threshold_scale = ttk.Scale(thresh_f, from_=0.1, to=50, value=self.change_threshold*100, command=self.update_threshold)
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(5 * self.dpi_scale)))
        self.threshold_label = ttk.Label(thresh_f, text=f"{self.change_threshold*100:.1f}%", width=int(6 * self.dpi_scale))
        self.threshold_label.pack(side=tk.RIGHT)
        self.threshold_entry_var = tk.StringVar(value=f"{self.change_threshold*100:.1f}")
        self.threshold_entry = ttk.Entry(
            thresh_f, 
            textvariable=self.threshold_entry_var, 
            width=int(8 * self.dpi_scale),
            font=("SimHei", self.scaled_font_size) 
        )
        self.threshold_entry.pack(side=tk.RIGHT, padx=(int(5 * self.dpi_scale), 0))
        self.threshold_entry.bind("<Return>", self.validate_threshold_input)

        ttk.Label(param_frame, text="截图最小间隔 (秒)").pack(anchor=tk.W, pady=(0, int(2 * self.dpi_scale)))
        intv_f = ttk.Frame(param_frame)
        intv_f.pack(fill=tk.X, pady=(0, int(8 * self.dpi_scale)))
        self.interval_scale = ttk.Scale(intv_f, from_=0.1, to=10, value=self.min_interval, command=self.update_interval)
        self.interval_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(5 * self.dpi_scale)))
        self.interval_label = ttk.Label(intv_f, text=f"{self.min_interval:.1f}s", width=int(6 * self.dpi_scale))
        self.interval_label.pack(side=tk.RIGHT)
        self.interval_entry_var = tk.StringVar(value=f"{self.min_interval:.1f}")
        self.interval_entry = ttk.Entry(
            intv_f, 
            textvariable=self.interval_entry_var, 
            width=int(8 * self.dpi_scale),
            font=("SimHei", self.scaled_font_size)
        )        
        self.interval_entry.pack(side=tk.RIGHT, padx=(int(5 * self.dpi_scale), 0))
        self.interval_entry.bind("<Return>", self.validate_interval_input)

        save_frame = ttk.LabelFrame(settings_scrollable_frame, text="截图保存", padding=(inner_pad, int(8 * self.dpi_scale)))
        save_frame.grid(row=row, column=0, sticky="ew", pady=frame_pady, padx=int(5 * self.dpi_scale))
        row += 1
        ttk.Label(save_frame, text="主保存路径:").pack(anchor=tk.W, pady=(0, int(2 * self.dpi_scale)))
        path_f1 = ttk.Frame(save_frame)
        path_f1.pack(fill=tk.X, pady=(0, int(5 * self.dpi_scale)))
        self.save_path_var = tk.StringVar(value=self.save_path)
        ttk.Entry(
            path_f1, 
            textvariable=self.save_path_var,
            font=("SimHei", self.scaled_font_size) 
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, int(3 * self.dpi_scale)))

        ttk.Button(path_f1, text="更改", command=self.change_save_path, width=int(6 * self.dpi_scale)).pack(side=tk.RIGHT)
        ttk.Label(save_frame, text="备份路径:").pack(anchor=tk.W, pady=(int(5 * self.dpi_scale), int(2 * self.dpi_scale)))
        self.backup_path_label = ttk.Label(save_frame, text=self.backup_save_path, 
                                          font=("TkDefaultFont", max(8, int(8 * self.dpi_scale))), foreground="#666666", 
                                          wraplength=int(250 * self.dpi_scale), justify=tk.LEFT)
        self.backup_path_label.pack(anchor=tk.W, pady=(0, int(5 * self.dpi_scale)))

        ttk.Label(settings_scrollable_frame, text="").grid(row=row, column=0, pady=int(10 * self.dpi_scale))

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
        ttk.Label(status_progress_frame, text="整体进度:").pack(anchor=tk.W)
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
        for i, s in enumerate(self.speed_levels):
            btn = ttk.Button(speed_btn_frame, text=f"{s}x", command=lambda s=s: self.set_speed(s), width=int(5 * self.dpi_scale))
            btn.grid(row=i//cols, column=i%cols, padx=int(2 * self.dpi_scale), pady=int(2 * self.dpi_scale), sticky="ew")
        for i in range(cols):
            speed_btn_frame.grid_columnconfigure(i, weight=1)

        control_buttons_frame = ttk.LabelFrame(control_frame_nb, text="控制命令", padding=(inner_pad, int(8 * self.dpi_scale)))
        control_buttons_frame.grid(row=2, column=0, sticky="ew", padx=int(5 * self.dpi_scale), pady=int(5 * self.dpi_scale))
        btn_frame = ttk.Frame(control_buttons_frame)
        btn_frame.pack(fill=tk.X, pady=int(5 * self.dpi_scale))
        ttk.Button(btn_frame, text="开始处理", command=self.start_processing).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=int(2 * self.dpi_scale))
        ttk.Button(btn_frame, text="暂停", command=self.pause_processing).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=int(2 * self.dpi_scale))
        ttk.Button(btn_frame, text="停止", command=self.stop_processing).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=int(2 * self.dpi_scale))

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
        help_content = """
【软件使用说明】
本程序使用GMM（Gaussian Mixture Model)高斯混合模型算法检测画面变化率。

1. 【添加视频文件】
点击“添加视频文件”按钮，选择一个或多个视频（支持 MP4、AVI、MOV、MKV、FLV）。
视频将显示在列表中，可多选后点击“移除”或“预览”。

2. 【预览与选择关注区域（ROI）】
选中视频后点击“预览”，可查看第一帧画面。
点击“选择ROI区域”可在预览帧上绘制多边形区域：
• 左键点击添加顶点
• 右键点击删除最后一个顶点
• 按 R 键重置所有点
• 关闭窗口即确认选择（需 ≥3 个点）
ROI 用于限定检测范围，排除干扰提升效率。

3. 【设置检测参数】
**画面变化阈值**：当画面变化比例超过此值时触发截图（默认 5%）。
**截图最小间隔**：两次截图之间至少间隔指定秒数，避免截图过多。

4. 【设置保存路径】
默认截图保存在程序目录下的“变化截图”文件夹。
可点击“更改”指定其他路径；若主路径不可写，将自动尝试备份路径。

5. 【开始处理】
点击“开始处理”按钮，程序将逐个分析视频。
支持暂停/继续、停止操作。
可通过“处理速度控制”调整分析倍速（丢帧方式），加快处理。

6. 【结果查看】
检测到变化大于设定值时，会自动保存原始帧截图。
截图文件名包含视频名、帧号和时间戳，便于追溯。
所有操作和错误信息会记录在“检测日志”文件夹中。

7. 【注意事项】
建议根据视频画面设测试参数。
高倍速处理是用丢帧方式处理，要注意丢帧太多会漏掉快速移动目标，同时为保证性能会降低预览更新频率。
若截图保存失败，请检查磁盘权限或更换保存路径。
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

    def on_preview_resize(self, event=None):
        if event.width < 50 or event.height < 50:
            return
        if hasattr(self, '_last_preview_size'):
            if (event.width, event.height) == self._last_preview_size:
                return
        self._last_preview_size = (event.width, event.height)

        if hasattr(self, '_last_displayed_frame') and self._last_displayed_frame is not None:
            self.display_frame(self._last_displayed_frame.copy())
        elif self.preview_frame is not None:
            rgb_frame = cv2.cvtColor(self.preview_frame, cv2.COLOR_BGR2RGB)
            self.display_frame(rgb_frame)

    def display_frame(self, frame):

        self._last_displayed_frame = frame.copy()

        container = self.video_label.master 
        container.update_idletasks()
        display_width = container.winfo_width()
        display_height = container.winfo_height()

        if display_width < 100 or display_height < 100:
            return

        frame_height, frame_width = frame.shape[:2]
        if frame_width == 0 or frame_height == 0:
            return

        scale = min(display_width / frame_width, display_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if self.roi_selected and self.roi_points and self.preview_frame is not None:
            orig_h, orig_w = self.preview_frame.shape[:2]
            scale_x = new_width / orig_w
            scale_y = new_height / orig_h
            scaled_pts = np.array([
                [int(x * scale_x), int(y * scale_y)] for (x, y) in self.roi_points
            ], dtype=np.int32)
            cv2.polylines(resized_frame, [scaled_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        img = Image.fromarray(resized_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

    def validate_threshold_input(self, event=None):
        try:
            value = float(self.threshold_entry_var.get())
            if 0.1 <= value <= 50:
                self.change_threshold = value / 100
                self.threshold_scale.set(value)
                self.threshold_label.config(text=f"{value:.1f}%")
                self.log_message(f"变化阈值设置为 {value:.1f}%")
            else:
                raise ValueError("超出范围")
        except ValueError:
            messagebox.showwarning("输入错误", "请输入 0.1 ~ 50 之间的有效数字")
            self.threshold_entry_var.set(f"{self.change_threshold*100:.1f}")

    def validate_interval_input(self, event=None):
        try:
            value = float(self.interval_entry_var.get())
            if 0.1 <= value <= 10:
                self.min_interval = value
                self.interval_scale.set(value)
                self.interval_label.config(text=f"{value:.1f}s")
                self.log_message(f"截图最小间隔设置为 {value:.1f}秒")
            else:
                raise ValueError("超出范围")
        except ValueError:
            messagebox.showwarning("输入错误", "请输入 0.1 ~ 10 之间的有效数字")
            self.interval_entry_var.set(f"{self.min_interval:.1f}")

    def update_threshold(self, value):
        self.change_threshold = float(value) / 100
        self.threshold_label.config(text=f"{float(value):.1f}%")
        self.threshold_entry_var.set(f"{float(value):.1f}")
        self.log_message(f"变化阈值设置为 {float(value):.1f}%")

    def update_interval(self, value):
        self.min_interval = float(value)
        self.interval_label.config(text=f"{self.min_interval:.1f}秒")
        self.interval_entry_var.set(f"{self.min_interval:.1f}")
        self.log_message(f"截图最小间隔设置为 {self.min_interval:.1f}秒")

    def select_roi(self):
        if self.roi_selected:
            self.cancel_roi()
            return
        if self.processing:
            messagebox.showinfo("提示", "请先停止视频处理，再选择ROI区域")
            return
        if not self.video_paths:
            messagebox.showwarning("警告", "请先添加视频文件")
            return
        if self.preview_frame is None:
            selected_indices = self.video_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("警告", "请先选择并预览一个视频文件")
                return
            self.preview_selected_video()
            if self.preview_frame is None:
                return
        roi_window = tk.Toplevel(self.root)
        roi_window.title("选择关注区域 (ROI) - 左键加点，右键删点，按 R 重置，滚轮缩放，关闭窗口确认选择")
        roi_window.geometry("1024x700")
        roi_window.transient(self.root)
        roi_window.grab_set()
        canvas_frame = ttk.Frame(roi_window)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=0)
        v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        pil_img = Image.fromarray(cv2.cvtColor(self.preview_frame, cv2.COLOR_BGR2RGB))
        self.roi_original_size = pil_img.size
        self.roi_scale_factor = 1.0
        self.roi_image = pil_img
        self.roi_photo = None
        roi_points = []
        lines = []
        circles = []

        def draw_roi():
            for item in lines + circles:
                canvas.delete(item)
            lines.clear()
            circles.clear()
            if len(roi_points) < 1:
                return
            scaled = [(x * self.roi_scale_factor, y * self.roi_scale_factor) for x, y in roi_points]
            for i, (x, y) in enumerate(scaled):
                c = canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", outline="red")
                circles.append(c)
                if i > 0:
                    l = canvas.create_line(scaled[i-1][0], scaled[i-1][1], x, y, fill="lime", width=2)
                    lines.append(l)
            if len(roi_points) > 2:
                l = canvas.create_line(scaled[-1][0], scaled[-1][1], scaled[0][0], scaled[0][1], fill="lime", width=2)
                lines.append(l)

        def update_image():
            new_width = int(self.roi_original_size[0] * self.roi_scale_factor)
            new_height = int(self.roi_original_size[1] * self.roi_scale_factor)
            resized_img = self.roi_image.resize((new_width, new_height), Image.LANCZOS)
            self.roi_photo = ImageTk.PhotoImage(resized_img)
            canvas.config(scrollregion=(0, 0, new_width, new_height))
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=self.roi_photo)
            draw_roi()

        def fit_to_window():
            win_w = roi_window.winfo_width()
            win_h = roi_window.winfo_height()
            if win_w <= 1 or win_h <= 1:
                roi_window.after(100, fit_to_window)
                return
            img_w, img_h = self.roi_original_size
            scale_w = (win_w - 20) / img_w
            scale_h = (win_h - 20) / img_h
            self.roi_scale_factor = min(scale_w, scale_h, 1.0)
            update_image()

        def on_click(event):
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            orig_x = x / self.roi_scale_factor
            orig_y = y / self.roi_scale_factor
            roi_points.append((orig_x, orig_y))
            draw_roi()

        def on_right_click(event):
            if roi_points:
                roi_points.pop()
                draw_roi()

        def on_key(event):
            if event.char.lower() == 'r':
                roi_points.clear()
                draw_roi()

        def on_mousewheel(event):
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            orig_x = x / self.roi_scale_factor
            orig_y = y / self.roi_scale_factor
            if event.delta > 0 or event.num == 4:
                self.roi_scale_factor *= 1.2
            elif event.delta < 0 or event.num == 5:
                self.roi_scale_factor /= 1.2
            self.roi_scale_factor = max(0.1, min(self.roi_scale_factor, 5.0))
            update_image()
            new_x = orig_x * self.roi_scale_factor
            new_y = orig_y * self.roi_scale_factor
            dx = new_x - x
            dy = new_y - y
            canvas.xview_scroll(int(dx), "units")
            canvas.yview_scroll(int(dy), "units")

        canvas.bind("<Button-1>", on_click)
        canvas.bind("<Button-3>", on_right_click)
        roi_window.bind("<Key>", on_key)
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Button-4>", on_mousewheel)
        canvas.bind("<Button-5>", on_mousewheel)
        canvas.focus_set()
        roi_window.update_idletasks()
        fit_to_window()

        def on_close():
            if len(roi_points) >= 3:
                self.roi_points = roi_points + [roi_points[0]]
                self.roi_selected = True
                self.roi_button.config(text="取消选取", command=self.cancel_roi)
                self.roi_status_label.config(text=f"已选取关注区域，{len(roi_points)}个顶点")
                self.create_roi_mask()
                if self.preview_frame is not None:
                    preview_with_roi = self.preview_frame.copy()
                    pts = np.array(self.roi_points, np.int32)
                    cv2.polylines(preview_with_roi, [pts], True, (0, 255, 0), 2)
                    rgb_frame = cv2.cvtColor(preview_with_roi, cv2.COLOR_BGR2RGB)
                    self.display_frame(rgb_frame)
                self.log_message(f"已选择ROI区域，{len(roi_points)}个顶点")
                self.info_label.config(text=f"已选择ROI区域，{len(roi_points)}个顶点")
            else:
                if len(roi_points) > 0:
                    messagebox.showerror("错误", "至少需要3个点才能构成多边形")
            roi_window.destroy()

        roi_window.protocol("WM_DELETE_WINDOW", on_close)

    def cancel_roi(self):
        self.roi_selected = False
        self.roi_points = []
        self.roi_mask = None
        self.roi_button.config(text="选择ROI区域", command=self.select_roi)
        self.roi_status_label.config(text="未选取关注区域")
        if self.preview_frame is not None:
            rgb_frame = cv2.cvtColor(self.preview_frame, cv2.COLOR_BGR2RGB)
            self.display_frame(rgb_frame)
        self.log_message("已取消ROI区域选择")
        self.info_label.config(text="已取消ROI区域选择")

    def create_roi_mask(self):
        if not self.roi_selected or not self.roi_points or self.preview_frame is None:
            return
        try:
            height, width = self.preview_frame.shape[:2]
            mask = np.zeros((height, width), np.uint8)
            pts = np.array([[int(x), int(y)] for x, y in self.roi_points[:-1]], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            self.roi_mask = mask
            self.log_message(f"已创建ROI掩码，大小: {width}x{height}")
        except Exception as e:
            error_msg = f"创建ROI掩码时出错: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("错误", error_msg)
            self.roi_mask = None

    def process_videos(self):
        try:
            while self.current_video_index < len(self.video_paths) and self.processing:
                video_path = self.video_paths[self.current_video_index]
                video_name = os.path.basename(video_path)
                self.safe_ui_call(self.progress_label.config, text=f"正在处理: {video_name}")
                self.safe_ui_call(self.status_var.set, f"处理中: {video_name} | 当前倍速: {self.current_speed}倍")
                self.log_message(f"开始处理视频: {video_name}")
                if self.cap is not None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(video_path)
                if not self.cap.isOpened():
                    error_msg = f"无法打开视频: {video_name}"
                    self.safe_ui_call(messagebox.showerror, "错误", error_msg)
                    self.log_message(error_msg)
                    self.current_video_index += 1
                    continue

                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

                ret, first_frame = self.cap.read()
                if not ret:
                    self.log_message(f"无法读取首帧: {video_name}")
                    self.current_video_index += 1
                    continue

                if self.target_height > 0 and first_frame.shape[0] > self.target_height:
                    scale = self.target_height / first_frame.shape[0]
                    new_w = int(first_frame.shape[1] * scale)
                    new_h = self.target_height
                    first_frame = cv2.resize(first_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                frame_height, frame_width = first_frame.shape[:2]

                gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                self.gmm = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)

                if self.roi_selected and self.roi_mask is not None:
                    roi_mask_resized = cv2.resize(self.roi_mask, (frame_width, frame_height))
                    gray_roi = cv2.bitwise_and(gray, gray, mask=roi_mask_resized)
                    self.gmm.apply(gray_roi)
                else:
                    self.gmm.apply(gray)

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_id = 0
                last_saved_time = 0

                while frame_id < total_frames and self.processing:
                    if self.paused:
                        time.sleep(0.1)
                        continue

                    if self.current_speed > 1:
                        target_frame = int(frame_id + self.current_speed)
                        if target_frame >= total_frames:
                            break
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        frame_id = target_frame
                    else:
                        frame_id += 1

                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    if self.target_height > 0 and frame.shape[0] > self.target_height:
                        scale = self.target_height / frame.shape[0]
                        new_w = int(frame.shape[1] * scale)
                        new_h = self.target_height
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    current_roi_mask = None
                    if self.roi_selected and self.roi_mask is not None:
                        current_roi_mask = cv2.resize(self.roi_mask, (frame_width, frame_height))
                        gray = cv2.bitwise_and(gray, gray, mask=current_roi_mask)

                    fg_mask = self.gmm.apply(gray)
                    _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
                    total_pixels = cv2.countNonZero(current_roi_mask) if self.roi_selected else gray.size
                    change_pixels = cv2.countNonZero(fg_mask)
                    change_ratio = change_pixels / total_pixels if total_pixels > 0 else 0

                    now = time.time()
                    if change_ratio > self.change_threshold and (now - last_saved_time) > self.min_interval:
                        last_saved_time = now
                        video_basename = os.path.splitext(video_name)[0]
                        saved_path = self.save_screenshot(frame.copy(), video_basename, frame_id)
                        if saved_path:
                            self.safe_ui_call(self.status_var.set,
                                f"已保存截图: {os.path.basename(saved_path)} | 当前倍速: {self.current_speed}倍")

                    now_time = time.time()
                    if not hasattr(self, '_last_preview_update_time'):
                        self._last_preview_update_time = now_time

                    if now_time - self._last_preview_update_time >= 0.3:
                        color_mask = np.zeros_like(frame)
                        color_mask[:, :, 2] = fg_mask
                        marked = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)
                        if self.roi_selected and self.roi_points:
                            pts = np.array(self.roi_points, np.int32)
                            cv2.polylines(marked, [pts], True, (0, 255, 0), 2)
                        cv2.putText(marked, f"Change: {change_ratio*100:.1f}% | Speed: {self.current_speed}x",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        rgb_marked = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
                        self.safe_ui_call(self.display_frame, rgb_marked)
                        self._last_preview_update_time = now_time


                    overall_progress = ((self.current_video_index + frame_id / total_frames) / len(self.video_paths)) * 100
                    self.safe_ui_call(self.progress_var.set, overall_progress)
                    self.safe_ui_call(self.progress_label.config,
                        text=f"处理: {video_name} ({frame_id}/{total_frames}) | Speed: {self.current_speed}x")

                    if self.current_speed == 1:
                        time.sleep(1 / fps)

                self.cap.release()
                self.current_video_index += 1

            self.processing = False
            self.safe_ui_call(self.progress_label.config, text="所有视频处理完毕")
            self.safe_ui_call(self.status_var.set, f"处理完成 | 当前倍速: {self.current_speed}倍")
            self.safe_ui_call(self.progress_var.set, 100)
            if os.listdir(self.backup_save_path):
                self.safe_ui_call(messagebox.showinfo, "备份提示", f"部分截图在:\n{self.backup_save_path}")
            if self.current_video_index >= len(self.video_paths):
                self.log_message(f"所有 {len(self.video_paths)} 个视频处理完成")
                self.safe_ui_call(messagebox.showinfo, "完成", "所有视频处理已完成")
        except Exception as e:
            self.processing = False

            import traceback
            error_detail = traceback.format_exc()  
            error_msg = f"处理出错: {str(e)}\n详细信息: {error_detail}"
            self.log_message(error_msg)
            self.safe_ui_call(messagebox.showerror, "错误", error_msg)

            release_error = ""
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    if self.cap.isOpened():
                        self.cap.release()
                        self.log_message("异常处理中释放了cap资源")
                except Exception as release_err:
                    release_error = f"；释放资源时也出错: {str(release_err)}"
                    self.log_message(release_error)
                finally:
                    self.cap = None
 

    def preview_selected_video(self):
        if not self.video_paths:
            messagebox.showwarning("警告", "请先添加视频文件")
            return
        selected_indices = self.video_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("警告", "请先选择一个视频文件")
            return
        if self.processing:
            self.stop_processing()
            time.sleep(0.5)
        video_index = selected_indices[0]
        video_path = self.video_paths[video_index]
        video_name = os.path.basename(video_path)
        self.log_message(f"预览视频: {video_name}")
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            error_msg = f"无法打开视频文件: {video_name}"
            messagebox.showerror("错误", error_msg)
            self.log_message(error_msg)
            self.cap = None
            return
        ret, frame = self.cap.read()
        if ret:
            self.preview_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_frame(rgb_frame)
            self.progress_label.config(text=f"预览: {video_name} (可选择ROI区域)")
            self.info_label.config(text=f"预览中: {video_name}")
            self.log_message(f"成功预览视频: {video_name}")
        else:
            error_msg = f"无法读取视频帧: {video_name}"
            messagebox.showerror("错误", error_msg)
            self.log_message(error_msg)
            self.cap.release()
            self.cap = None

    def add_videos(self):
        file_paths = filedialog.askopenfilenames(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv;*.flv")]
        )
        if file_paths:
            added_count = 0
            for path in file_paths:
                if path not in self.video_paths:
                    self.video_paths.append(path)
                    self.video_listbox.insert(tk.END, os.path.basename(path))
                    added_count += 1
            self.log_message(f"添加了 {added_count} 个视频文件")
            self.info_label.config(text=f"已添加 {added_count} 个视频文件")

    def remove_video(self):
        selected_indices = self.video_listbox.curselection()
        if selected_indices:
            removed_count = 0
            for i in sorted(selected_indices, reverse=True):
                del self.video_paths[i]
                self.video_listbox.delete(i)
                removed_count += 1
            self.log_message(f"移除了 {removed_count} 个视频文件")
            self.info_label.config(text=f"已移除 {removed_count} 个视频文件")

    def clear_videos(self):
        video_count = len(self.video_paths)
        self.video_paths.clear()
        self.video_listbox.delete(0, tk.END)
        self.current_video_index = 0
        self.cancel_roi()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.log_message(f"清空了所有视频文件，共 {video_count} 个")
        self.info_label.config(text=f"已清空所有视频文件 ({video_count} 个)")

    def change_save_path(self):
        new_path = filedialog.askdirectory(title="选择截图保存路径")
        if new_path:
            try:
                test_filename = f"test_write_permission_{datetime.now().strftime('%Y%m%d%H%M%S')}.tmp"
                test_file = os.path.join(new_path, test_filename)
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                self.save_path = new_path
                self.save_path_var.set(new_path)
                self.ensure_directory_exists(self.save_path)
                self.log_message(f"截图保存路径更改至: {new_path}")
                self.info_label.config(text=f"截图保存路径已更改: {os.path.basename(new_path)}")
            except Exception as e:
                error_msg = f"所选路径不可写: {str(e)}"
                messagebox.showwarning("路径警告", error_msg)
                self.log_message(error_msg)

    def set_speed(self, speed):
        if speed in self.speed_levels:
            with self.speed_lock:
                self.current_speed = speed
            self.speed_label.config(text=f"当前: {self.current_speed}倍")
            self.status_var.set(f"{self.status_var.get().split('|')[0].strip()} | 当前倍速: {self.current_speed}倍")
            self.log_message(f"处理倍速设置为 {self.current_speed}倍")
            self.info_label.config(text=f"处理倍速已设置为 {self.current_speed}倍")

    def start_processing(self):
        if not self.video_paths:
            messagebox.showwarning("警告", "请先添加视频文件")
            return
        if self.processing:
            if self.paused:
                self.paused = False
                self.progress_label.config(text="继续处理...")
                self.status_var.set(f"处理中 | 当前倍速: {self.current_speed}倍")
                self.log_message("继续视频处理")
                self.info_label.config(text="继续视频处理")
            return

        try:
            test_filename = f"test_screenshot_permission_{datetime.now().strftime('%Y%m%d%H%M%S')}.tmp"
            test_file = os.path.join(self.save_path, test_filename)
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            self.log_message(f"主保存路径可写性测试通过: {self.save_path}")
        except Exception as e:
            error_msg = f"主保存路径不可写: {str(e)}\n将尝试使用备份路径: {self.backup_save_path}"
            messagebox.showwarning("路径警告", error_msg)
            self.log_message(error_msg)
            try:
                test_file = os.path.join(self.backup_save_path, test_filename)
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                self.log_message(f"备份保存路径可写性测试通过: {self.backup_save_path}")
            except Exception as e2:
                error_msg2 = f"备份保存路径也不可写: {str(e2)}\n程序无法保存截图，可能需要以管理员身份运行或更换保存路径。"
                messagebox.showerror("路径错误", error_msg2)
                self.log_message(error_msg2)
                return

        self.processing = True
        self.paused = False
        self.current_video_index = 0
        self.status_var.set(f"处理中 | 当前倍速: {self.current_speed}倍")
        self.log_message("开始处理视频列表")
        self.info_label.config(text="开始处理视频列表")
        threading.Thread(target=self.process_videos, daemon=True).start()

    def pause_processing(self):
        if self.processing and not self.paused:
            self.paused = True
            self.progress_label.config(text="已暂停，点击开始继续处理")
            self.status_var.set(f"已暂停 | 当前倍速: {self.current_speed}倍")
            self.log_message("暂停视频处理")
            self.info_label.config(text="视频处理已暂停")

    def stop_processing(self):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.cap is not None:
            try:
                self.cap.release()
                self.log_message(f"[{timestamp}] self.cap 已释放")
            except Exception as e:
                self.log_message(f"[{timestamp}] 释放 cap 出错: {str(e)}")
            finally:
                self.cap = None  

 
        self.log_message("已停止处理视频")
        self.safe_ui_call(self.status_var.set, f"已停止 | 当前倍速: {self.current_speed}倍")
        self.safe_ui_call(self.progress_label.config, text="处理已停止")
        
 
        self.gmm = None
        
        self.safe_ui_call(self.status_var.set, f"已停止 | 当前倍速: {self.current_speed}倍")
        self.safe_ui_call(self.progress_label.config, text="处理已停止")
        self.safe_ui_call(self.progress_var.set, 0)
        self.safe_ui_call(self.info_label.config, text="处理已停止")
        
        self.current_video_index = 0
        self.log_message("处理已停止")

    def save_screenshot(self, marked_frame, video_basename, frame_num):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_name = f"{video_basename}_frame_{frame_num}_{timestamp}.jpg"
            main_path = os.path.join(self.save_path, screenshot_name)
            success = False
            try:
                success = cv2.imwrite(main_path, marked_frame)
                if success:
                    self.log_message(f"使用OpenCV成功保存截图到主路径: {main_path}")
                    self.info_label.config(text=f"已保存变化截图: {os.path.basename(main_path)}")
                    return main_path
            except Exception as e:
                self.log_message(f"OpenCV保存到主路径失败: {str(e)}")
            try:
                rgb_img = cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                pil_img.save(main_path)
                success = True
                self.log_message(f"使用PIL成功保存截图到主路径: {main_path}")
                self.info_label.config(text=f"已保存变化截图: {os.path.basename(main_path)}")
                return main_path
            except Exception as e:
                self.log_message(f"PIL保存到主路径失败: {str(e)}")

            self.log_message(f"主路径保存失败，尝试备份路径: {self.backup_save_path}")
            backup_path = os.path.join(self.backup_save_path, screenshot_name)
            try:
                cv2.imwrite(backup_path, marked_frame)
                success = True
                self.log_message(f"使用OpenCV成功保存截图到备份路径: {backup_path}")
                self.info_label.config(text=f"已保存变化截图到备份路径: {os.path.basename(backup_path)}")
                return backup_path
            except Exception as e:
                self.log_message(f"OpenCV保存到备份路径失败: {str(e)}")
            try:
                rgb_img = cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                pil_img.save(backup_path)
                success = True
                self.log_message(f"使用PIL成功保存截图到备份路径: {backup_path}")
                self.info_label.config(text=f"已保存变化截图到备份路径: {os.path.basename(backup_path)}")
                return backup_path
            except Exception as e:
                self.log_message(f"PIL保存到备份路径失败: {str(e)}")

            if not success:
                error_msg = "无法保存截图到任何路径，请检查权限设置。"
                messagebox.showerror("保存错误", error_msg)
                self.log_message(error_msg)
                self.info_label.config(text="截图保存失败，请检查路径权限")
                return None
        except Exception as e:
            error_msg = f"保存截图时发生错误: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("错误", error_msg)
            self.info_label.config(text="截图保存失败")
            return None

if __name__ == "__main__":
    root = tk.Tk()
    app = GMMVideoDetector(root)
    root.mainloop()