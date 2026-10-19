import sys
import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import cv2
import mediapipe as mp
import threading
import time
import customtkinter as ctk
from PIL import Image, ImageTk
from logic import OpticalAssistantEngine

ctk.set_appearance_mode("Dark")

class ModernOpticalAssistantGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Smart AI Optical Assistant")
        self.geometry("980x680")
        self.configure(fg_color="#0F111A")
        self.resizable(False, False)
        self.iconbitmap(resource_path("icon.ico"))

        self.engine = OpticalAssistantEngine(model_path=resource_path('face_landmarker.task'))
        self.is_running = False
        self.cap = None
        self.thread = None

        self.prev_frame_time = 0
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, fg_color="#1A1D24", corner_radius=20, border_width=1, border_color="#2A2E3D")
        self.sidebar.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="👓 OPTICAL AI", 
            font=ctk.CTkFont(family="Consolas", size=22, weight="bold"),
            text_color="#00E5FF"
        )
        self.logo_label.pack(padx=20, pady=(25, 2))

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar, 
            text="Virtual Try-On & Styling", 
            font=ctk.CTkFont(size=11),
            text_color="#6C7289"
        )
        self.subtitle_label.pack(padx=20, pady=(0, 20))

        # زر تشغيل الكاميرا
        self.btn_toggle = ctk.CTkButton(
            self.sidebar, 
            text="START ENGINE", 
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#00E5FF", 
            text_color="#0F111A",
            hover_color="#00B0FF",
            height=45,
            corner_radius=12,
            command=self.toggle_engine
        )
        self.btn_toggle.pack(padx=20, pady=10, fill="x")

        # زر بدء المعايرة الذكية
        self.btn_calibrate = ctk.CTkButton(
            self.sidebar, 
            text="START CALIBRATION [C]", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#3B82F6", 
            text_color="#FFFFFF",
            hover_color="#2563EB",
            height=40,
            corner_radius=12,
            command=self.trigger_calibration
        )
        self.btn_calibrate.pack(padx=20, pady=5, fill="x")

        # بطاقة الحالة المباشرة
        self.status_card = ctk.CTkFrame(self.sidebar, fg_color="#12141C", corner_radius=14, border_width=1, border_color="#252936")
        self.status_card.pack(padx=20, pady=15, fill="x")

        self.status_title = ctk.CTkLabel(self.status_card, text="FACE SHAPE STATUS", font=ctk.CTkFont(size=10, weight="bold"), text_color="#6C7289")
        self.status_title.pack(padx=15, pady=(10, 2), anchor="w")

        self.status_text = ctk.CTkLabel(
            self.status_card, 
            text="OFFLINE", 
            font=ctk.CTkFont(family="Consolas", size=16, weight="bold"),
            text_color="#FF5252"
        )
        self.status_text.pack(padx=15, pady=(0, 10), anchor="w")

        # --- دليل الاستخدام والتحكم ---
        self.guide_frame = ctk.CTkFrame(self.sidebar, fg_color="#12141C", corner_radius=14)
        self.guide_frame.pack(padx=20, pady=(5, 20), fill="both", expand=True)

        guide_text = (
            "💡 QUICK GUIDE\n\n"
            "1️⃣ Click START ENGINE\n"
            "2️⃣ Click CALIBRATION\n"
            "   Follow 5 head movements\n"
            "3️⃣ Click items in video catalog\n"
            "   Or use mouse to switch frames\n\n"
            "⭐ Best match is marked green!"
        )
        
        self.guide_label = ctk.CTkLabel(
            self.guide_frame, 
            text=guide_text, 
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#A0A5B5",
            justify="left"
        )
        self.guide_label.pack(padx=12, pady=12, anchor="nw")

        # --- لوحة الكاميرا الرئيسية (Viewport) ---
        self.video_frame = ctk.CTkFrame(self, fg_color="#1A1D24", corner_radius=20, border_width=1, border_color="#2A2E3D")
        self.video_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.video_label = ctk.CTkLabel(
            self.video_frame, 
            text="📷 CAMERA OFFLINE\n\nClick START ENGINE to initialize virtual try-on", 
            font=ctk.CTkFont(size=14),
            text_color="#52586B"
        )
        self.video_label.pack(expand=True, fill="both", padx=15, pady=15)
        
        # ربط حدث الكليك بالماوس داخل إطار الفيديو لاختيار النظارات من الكتالوج السفلي
        self.video_label.bind("<Button-1>", self.on_video_click)

    def on_video_click(self, event):
        if not self.is_running:
            return
        # إحداثيات الكليك بالنسبة للـ widget
        click_x, click_y = event.x, event.y
        for idx, (x1, y1, x2, y2) in enumerate(self.engine.thumbnail_boxes):
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                if idx < len(self.engine.active_glasses_list):
                    self.engine.current_glasses_index = idx
                    print(f"Selected glasses via GUI click: {self.engine.active_glasses_list[idx]}")

    def trigger_calibration(self):
        if self.is_running:
            self.engine.is_calibrating = True
            self.engine.calibration_step = 0
            self.engine.collected_ratios = []
            self.engine.final_face_shape = "🔄 Scanning..."
            print("Starting Precise Calibration...")

    def toggle_engine(self):
        if not self.is_running:
            self.is_running = True
            self.btn_toggle.configure(text="STOP ENGINE", fg_color="#FF3366", hover_color="#D81B60", text_color="#FFFFFF")
            self.status_text.configure(text="INITIALIZING...", text_color="#00E5FF")
            
            self.thread = threading.Thread(target=self.run_engine_loop, daemon=True)
            self.thread.start()
        else:
            self.is_running = False
            self.btn_toggle.configure(text="START ENGINE", fg_color="#00E5FF", hover_color="#00B0FF", text_color="#0F111A")
            self.status_text.configure(text="OFFLINE", text_color="#FF5252")
            self.video_label.configure(image="", text="📷 CAMERA OFFLINE\n\nClick START ENGINE to initialize virtual try-on")

    def update_gui_state(self, img_tk, status_msg):
        if self.is_running:
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk
            self.status_text.configure(text=status_msg, text_color="#00E5FF")

    def run_engine_loop(self):
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.engine.detector.model_asset_path if hasattr(self.engine.detector, 'model_asset_path') else 'face_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            with FaceLandmarker.create_from_options(options) as recognizer:
                while self.is_running and self.cap.isOpened():
                    success, frame = self.cap.read()
                    if not success or frame is None:
                        continue

                    frame = cv2.flip(frame, 1)

                    # حساب الـ FPS
                    new_frame_time = time.time()
                    fps = int(1 / (new_frame_time - self.prev_frame_time)) if self.prev_frame_time > 0 else 0
                    self.prev_frame_time = new_frame_time

                    processed_frame, status_msg = self.engine.process_frame(frame, recognizer)

                    # عرض الـ FPS
                    cv2.putText(processed_frame, f"FPS: {fps}", (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 2, cv2.LINE_AA)

                    # تحويل الإطار للعرض في CustomTkinter
                    rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_image)
                    img_tk = ImageTk.PhotoImage(image=img)

                    self.after(0, self.update_gui_state, img_tk, status_msg)
        except Exception as e:
            print(f"Error in Vision Loop: {e}")
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()

if __name__ == "__main__":
    app = ModernOpticalAssistantGUI()
    app.mainloop()