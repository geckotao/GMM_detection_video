# main.py
import sys
import os

# 将当前目录加入路径（便于直接运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import GMMVideoDetector
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = GMMVideoDetector(root)
    root.mainloop()
    