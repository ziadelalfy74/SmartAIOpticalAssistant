import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class OpticalAssistantEngine:
    def __init__(self, model_path='face_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.load_glasses()
        
        self.is_calibrating = False
        self.calibration_step = 0
        self.collected_ratios = []
        self.final_face_shape = "⏳ READY"
        self.active_glasses_list = []
        self.current_glasses_index = 0
        
        self.center_nose_y = 0
        self.center_nose_x = 0
        self.thumbnail_boxes = []

        self.compatibility_rules = {
            "Square 🟧": ["round","aviator", "cateye",  "square"],
            "Round 🔵": ["square", "cateye", "aviator", "round"],
            "Oval 🟢": ["aviator", "square", "round", "cateye"],
            "Rectangle / Diamond 🔶": ["round", "cateye", "square", "aviator"]
        }
        
        self.start_time = time.time()

    def load_glasses(self):
        try:
            self.glasses_images = {
                "square": cv2.imread('square.png', cv2.IMREAD_UNCHANGED),
                "round": cv2.imread('round.png', cv2.IMREAD_UNCHANGED),
                "aviator": cv2.imread('aviator.png', cv2.IMREAD_UNCHANGED),
                "cateye": cv2.imread('cateye.png', cv2.IMREAD_UNCHANGED)
            }
        except Exception as e:
            print(f"Error loading glasses images: {e}")
            self.glasses_images = {}

    def calculate_face_metrics(self, landmarks, image_width, image_height):
        left_cheek = np.array([landmarks[234].x * image_width, landmarks[234].y * image_height])
        right_cheek = np.array([landmarks[454].x * image_width, landmarks[454].y * image_height])
        face_width = np.linalg.norm(left_cheek - right_cheek)

        chin = np.array([landmarks[152].x * image_width, landmarks[152].y * image_height])
        forehead = np.array([landmarks[10].x * image_width, landmarks[10].y * image_height])
        face_height = np.linalg.norm(chin - forehead)

        left_jaw = np.array([landmarks[172].x * image_width, landmarks[172].y * image_height])
        right_jaw = np.array([landmarks[397].x * image_width, landmarks[397].y * image_height])
        jaw_width = np.linalg.norm(left_jaw - right_jaw)

        ratio = face_height / face_width
        jaw_ratio = jaw_width / face_width
        return ratio, jaw_ratio

    def get_prescaled_dims(self, glasses_img, desired_width):
        if glasses_img is None: return 0, 0
        h, w = glasses_img.shape[:2]
        aspect_ratio = h / w
        desired_height = int(desired_width * aspect_ratio)
        return desired_width, desired_height

    def overlay_image(self, bg, overlay, x, y, w, h, ang=0):
        if overlay is None or w == 0 or h == 0: return bg
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
        if ang != 0:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
            overlay = cv2.warpAffine(overlay, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        h_ov, w_ov, _ = overlay.shape
        x1, y1 = int(x - w_ov / 2), int(y - h_ov / 2)
        x2, y2 = x1 + w_ov, y1 + h_ov
        if x1 < 0 or y1 < 0 or x2 > bg.shape[1] or y2 > bg.shape[0]: return bg
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
            alpha[alpha < 0.1] = 0.0
            for c in range(3):
                bg[y1:y2, x1:x2, c] = (alpha * overlay[:, :, c] + (1 - alpha) * bg[y1:y2, x1:x2, c])
        return bg

    def process_frame(self, frame, recognizer):
        image_height, image_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        detection_result = recognizer.detect_for_video(mp_image, timestamp_ms)

        status_text = self.final_face_shape
        calibration_msg = ""

        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                nose_tip = face_landmarks[1]
                nx, ny = int(nose_tip.x * image_width), int(nose_tip.y * image_height)

                if self.is_calibrating:
                    ratio, jaw_ratio = self.calculate_face_metrics(face_landmarks, image_width, image_height)
                    
                    if self.calibration_step == 0:
                        self.center_nose_x = nx
                        self.center_nose_y = ny
                        self.collected_ratios.append((ratio, jaw_ratio))
                        calibration_msg = "Step 1/5: 😊 Keep Still & Look Forward"
                        self.calibration_step = 1

                    elif self.calibration_step == 1:
                        calibration_msg = "Step 2/5: ⬆️ Look UP please!"
                        self.collected_ratios.append((ratio, jaw_ratio))
                        if ny < self.center_nose_y - 25: self.calibration_step = 2

                    elif self.calibration_step == 2:
                        calibration_msg = "Step 3/5: ⬇️ Look DOWN please!"
                        self.collected_ratios.append((ratio, jaw_ratio))
                        if ny > self.center_nose_y + 25: self.calibration_step = 3

                    elif self.calibration_step == 3:
                        calibration_msg = "Step 4/5: ➡️ Look RIGHT please!"
                        self.collected_ratios.append((ratio, jaw_ratio))
                        if nx > self.center_nose_x + 30: self.calibration_step = 4

                    elif self.calibration_step == 4:
                        calibration_msg = "Step 5/5: ⬅️ Look LEFT please!"
                        self.collected_ratios.append((ratio, jaw_ratio))
                        if nx < self.center_nose_x - 30:
                            self.is_calibrating = False
                            avg_ratio = np.mean([r[0] for r in self.collected_ratios])
                            avg_jaw = np.mean([r[1] for r in self.collected_ratios])
                            
                            if avg_ratio >= 1.30:
                                self.final_face_shape = "Oval 🟢"
                            elif avg_jaw > 0.75:
                                self.final_face_shape = "Square 🟧"
                            elif avg_ratio <= 1.15:
                                self.final_face_shape = "Round 🔵"
                            else:
                                self.final_face_shape = "Rectangle 🔶"

                            self.active_glasses_list = self.compatibility_rules.get(self.final_face_shape, ["round", "cateye", "aviator", "square"])
                            self.current_glasses_index = 0
                            status_text = self.final_face_shape

                else:
                    if self.active_glasses_list:
                        current_glass_type = self.active_glasses_list[self.current_glasses_index]
                        glass_img = self.glasses_images.get(current_glass_type)

                        if glass_img is not None:
                            nose_bridge = face_landmarks[6]
                            nx_b, ny_b = int(nose_bridge.x * image_width), int(nose_bridge.y * image_height)

                            left_eye = np.array([face_landmarks[33].x * image_width, face_landmarks[33].y * image_height])
                            right_eye = np.array([face_landmarks[263].x * image_width, face_landmarks[263].y * image_height])
                            eye_distance = int(np.linalg.norm(left_eye - right_eye))
                            
                            target_glasses_width = int(eye_distance * 1.6)
                            glasses_width, glasses_height = self.get_prescaled_dims(glass_img, target_glasses_width)

                            dy = right_eye[1] - left_eye[1]
                            dx = right_eye[0] - left_eye[0]
                            angle = np.degrees(np.arctan2(dy, dx))

                            frame = self.overlay_image(frame, glass_img, nx_b, ny_b, glasses_width, glasses_height, angle)

                        # رسم الكتالوج السفلي داخل الإطار
                        bar_y = image_height - 90
                        cv2.putText(frame, "Catalog (Click items):", (20, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        thumb_w = 80
                        start_x = 20
                        self.thumbnail_boxes = []
                        
                        for idx, g_name in enumerate(self.active_glasses_list):
                            g_img = self.glasses_images.get(g_name)
                            if g_img is not None:
                                th, tw = g_img.shape[:2]
                                t_height = int(thumb_w * (th / tw))
                                resized_thumb = cv2.resize(g_img, (thumb_w, t_height))
                                
                                tx1 = start_x + (idx * 110)
                                tx2 = tx1 + thumb_w
                                ty1 = bar_y
                                ty2 = bar_y + t_height
                                
                                self.thumbnail_boxes.append((tx1, ty1, tx2, ty2))
                                
                                if ty2 < image_height and tx2 < image_width:
                                    alpha_t = resized_thumb[:, :, 3] / 255.0
                                    alpha_t[alpha_t < 0.1] = 0.0
                                    for c in range(3):
                                        frame[ty1:ty2, tx1:tx2, c] = (alpha_t * resized_thumb[:, :, c] + (1 - alpha_t) * frame[ty1:ty2, tx1:tx2, c])
                                    
                                    border_color = (0, 255, 0) if idx == self.current_glasses_index else (255, 255, 255)
                                    cv2.rectangle(frame, (tx1-2, ty1-2), (tx2+2, ty2+2), border_color, 2)
                                    cv2.putText(frame, f"#{idx+1}", (tx1, ty1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)

        # عرض رسالة المعايرة بوضوح على الشاشة إذا كانت المعايرة نشطة
        if self.is_calibrating and calibration_msg:
            cv2.putText(frame, calibration_msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return frame, status_text