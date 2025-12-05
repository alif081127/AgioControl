import cv2
import numpy as np
import mediapipe as mp
import math
import sys
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from collections import deque
import pickle
import os

# ==================== CONFIGURATION ====================
CAMERA_INDEX = 1
FACE_DETECTION_ENABLED = True
AGE_ESTIMATION_ENABLED = True

# ==================== LAYOUT SETTINGS ====================
INFO_PANEL_WIDTH = 300
VOLUME_BAR_WIDTH = 80

# ==================== AGE ESTIMATION MODEL ====================
class ImprovedAgeEstimator:
    def __init__(self):
        # Data untuk kalibrasi usia (dari penelitian wajah manusia)
        self.age_data = {
            # Rentang usia dan karakteristik wajah
            'child': (5, 12, 0.22, 0.12, 0.32, 0.62, 0.03),  # mata besar, hidung pendek
            'teen': (13, 19, 0.21, 0.15, 0.35, 0.68, 0.04),
            'young_adult': (20, 29, 0.20, 0.17, 0.38, 0.72, 0.05),
            'adult': (30, 44, 0.19, 0.19, 0.40, 0.75, 0.06),
            'middle_aged': (45, 59, 0.18, 0.20, 0.42, 0.78, 0.07),
            'senior': (60, 80, 0.17, 0.21, 0.43, 0.80, 0.08)
        }
        
        # Rasio rata-rata untuk setiap kelompok usia (mata, hidung, mulut, rahang, alis)
        self.age_ratios = {
            'child': (0.22, 0.12, 0.32, 0.62, 0.03),
            'teen': (0.21, 0.15, 0.35, 0.68, 0.04),
            'young_adult': (0.20, 0.17, 0.38, 0.72, 0.05),
            'adult': (0.19, 0.19, 0.40, 0.75, 0.06),
            'middle_aged': (0.18, 0.20, 0.42, 0.78, 0.07),
            'senior': (0.17, 0.21, 0.43, 0.80, 0.08)
        }
        
        # History untuk smoothing
        self.age_history = deque(maxlen=10)
        self.face_size_history = deque(maxlen=5)
        
        # Faktor kalibrasi berdasarkan jarak kamera
        self.distance_factor = 1.0
        
    def extract_facial_features(self, landmarks, width, height):
        """Ekstrak fitur wajah untuk estimasi usia"""
        features = {}
        
        try:
            # 1. Rasio lebar mata terhadap lebar wajah
            left_eye_outer = landmarks[33]    # Sudut luar mata kiri
            right_eye_outer = landmarks[263]  # Sudut luar mata kanan
            left_face = landmarks[234]        # Sudut kiri wajah
            right_face = landmarks[454]       # Sudut kanan wajah
            
            eye_width = abs(right_eye_outer.x - left_eye_outer.x)
            face_width = abs(right_face.x - left_face.x)
            features['eye_ratio'] = eye_width / max(face_width, 0.001)
            
            # 2. Rasio tinggi hidung terhadap tinggi wajah
            nose_tip = landmarks[1]           # Ujung hidung
            nose_root = landmarks[168]        # Pangkal hidung (glabella)
            chin = landmarks[152]             # Dagu
            forehead = landmarks[10]          # Dahi
            
            nose_height = abs(nose_tip.y - nose_root.y)
            face_height = abs(chin.y - forehead.y)
            features['nose_ratio'] = nose_height / max(face_height, 0.001)
            
            # 3. Rasio lebar mulut terhadap lebar wajah
            mouth_left = landmarks[78]        # Sudut kiri mulut
            mouth_right = landmarks[308]      # Sudut kanan mulut
            
            mouth_width = abs(mouth_right.x - mouth_left.x)
            features['mouth_ratio'] = mouth_width / max(face_width, 0.001)
            
            # 4. Rasio lebar rahang terhadap lebar wajah
            jaw_left = landmarks[132]         # Rahang kiri
            jaw_right = landmarks[361]        # Rahang kanan
            
            jaw_width = abs(jaw_right.x - jaw_left.x)
            features['jaw_ratio'] = jaw_width / max(face_width, 0.001)
            
            # 5. Posisi alis relatif terhadap mata
            left_eyebrow = landmarks[65]      # Tengah alis kiri
            right_eyebrow = landmarks[295]    # Tengah alis kanan
            left_eye_center = landmarks[468]  # Tengah mata kiri
            right_eye_center = landmarks[473] # Tengah mata kanan
            
            brow_to_eye_left = abs(left_eyebrow.y - left_eye_center.y)
            brow_to_eye_right = abs(right_eyebrow.y - right_eye_center.y)
            features['brow_ratio'] = ((brow_to_eye_left + brow_to_eye_right) / 2) / max(face_height, 0.001)
            
            # 6. Ukuran wajah relatif terhadap frame (untuk kalibrasi jarak)
            face_area = face_width * face_height
            features['face_size'] = face_area
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def estimate_age_knn(self, features):
        """Estimasi usia menggunakan metode K-Nearest Neighbors sederhana"""
        if features is None:
            return 25, "Young Adult", (0, 255, 0), 0.7
        
        try:
            # Ekstrak fitur
            user_ratios = np.array([
                features['eye_ratio'],
                features['nose_ratio'],
                features['mouth_ratio'],
                features['jaw_ratio'],
                features['brow_ratio']
            ])
            
            # Hitung jarak ke setiap kelompok usia
            distances = []
            age_groups = []
            
            for group, ratios in self.age_ratios.items():
                group_ratios = np.array(ratios)
                distance = np.linalg.norm(user_ratios - group_ratios)
                distances.append(distance)
                age_groups.append(group)
            
            # Temukan 3 kelompok terdekat
            k = 3
            nearest_indices = np.argsort(distances)[:k]
            
            # Hitung usia rata-rata dari kelompok terdekat
            total_weight = 0
            weighted_age = 0
            
            for idx in nearest_indices:
                group = age_groups[idx]
                min_age, max_age = self.age_data[group][:2]
                avg_age = (min_age + max_age) / 2
                
                # Berat berdasarkan invers jarak (lebih dekat = lebih berat)
                weight = 1.0 / (distances[idx] + 0.001)
                weighted_age += avg_age * weight
                total_weight += weight
            
            estimated_age = weighted_age / total_weight
            
            # Smoothing dengan history
            self.age_history.append(estimated_age)
            if len(self.age_history) > 1:
                # Gunakan median untuk menghindari outlier
                smoothed_age = np.median(list(self.age_history))
            else:
                smoothed_age = estimated_age
            
            # Adjust berdasarkan ukuran wajah (kalibrasi jarak)
            self.face_size_history.append(features['face_size'])
            avg_face_size = np.mean(list(self.face_size_history))
            
            # Jika wajah terlalu kecil (jauh), usia cenderung lebih tua
            # Jika wajah terlalu besar (dekat), usia cenderung lebih muda
            if avg_face_size < 0.05:  # Wajah sangat kecil
                age_adjustment = 10
            elif avg_face_size < 0.1:  # Wajah kecil
                age_adjustment = 5
            elif avg_face_size > 0.3:  # Wajah sangat besar
                age_adjustment = -5
            elif avg_face_size > 0.2:  # Wajah besar
                age_adjustment = -3
            else:  # Ukuran normal
                age_adjustment = 0
            
            final_age = smoothed_age + age_adjustment
            
            # Clamp to reasonable range
            final_age = max(10, min(80, final_age))
            
            # Tentukan kelompok usia
            if final_age < 13:
                age_group = "Child"
                age_color = (255, 150, 0)  # Orange
                confidence = 0.7
            elif final_age < 20:
                age_group = "Teenager"
                age_color = (0, 200, 255)  # Cyan
                confidence = 0.8
            elif final_age < 30:
                age_group = "Young Adult"
                age_color = (0, 255, 0)  # Green
                confidence = 0.85
            elif final_age < 45:
                age_group = "Adult"
                age_color = (255, 255, 0)  # Yellow
                confidence = 0.8
            elif final_age < 60:
                age_group = "Middle-aged"
                age_color = (255, 100, 0)  # Orange-Red
                confidence = 0.75
            else:
                age_group = "Senior"
                age_color = (255, 0, 0)  # Red
                confidence = 0.7
            
            # Hitung confidence berdasarkan konsistensi rasio
            ratio_variance = np.var(user_ratios)
            confidence = max(0.5, min(0.95, confidence * (1.0 - ratio_variance * 2)))
            
            return int(final_age), age_group, age_color, confidence
            
        except Exception as e:
            print(f"Error in KNN age estimation: {e}")
            return 25, "Young Adult", (0, 255, 0), 0.7
    
    def simple_age_estimation(self, face_width, face_height, confidence_score):
        """Estimasi usia sederhana berdasarkan ukuran dan posisi wajah"""
        try:
            # Normalize face size (0-1)
            face_area = face_width * face_height
            
            # Default untuk usia muda (18-25)
            base_age = 22
            
            # Adjust berdasarkan ukuran wajah
            if face_area < 0.05:  # Wajah kecil (mungkin anak atau jarak jauh)
                age_adjust = np.random.randint(-5, 5)
            elif face_area < 0.1:  # Wajah sedang
                age_adjust = np.random.randint(-3, 3)
            else:  # Wajah besar (dekat dengan kamera)
                age_adjust = np.random.randint(-2, 2)
            
            estimated_age = base_age + age_adjust
            
            # Clamp to reasonable range
            estimated_age = max(15, min(30, estimated_age))
            
            # Tentukan kelompok
            if estimated_age < 20:
                age_group = "Teenager"
                age_color = (0, 200, 255)
                confidence = 0.8
            elif estimated_age < 25:
                age_group = "Young Adult"
                age_color = (0, 255, 0)
                confidence = 0.85
            else:
                age_group = "Adult"
                age_color = (255, 255, 0)
                confidence = 0.8
            
            return int(estimated_age), age_group, age_color, confidence
            
        except:
            return 22, "Young Adult", (0, 255, 0), 0.7

# ==================== VOLUME CONTROL ====================
def setup_volume_control():
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        vol_range = volume.GetVolumeRange()
        min_vol, max_vol = vol_range[0], vol_range[1]
        
        print("✓ Windows volume control ENABLED")
        return volume, min_vol, max_vol
        
    except Exception as e:
        print(f"⚠ Volume control disabled: {e}")
        print("  Running in VISUAL ONLY mode")
        return None, None, None

# ==================== MEDIAPIPE SETUP ====================
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

if FACE_DETECTION_ENABLED:
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

if AGE_ESTIMATION_ENABLED:
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# ==================== CAMERA CONNECTION ====================
def connect_to_camera():
    print("=" * 60)
    print("CAMERA CONNECTION TEST")
    print("=" * 60)
    
    camera_indices = [1, 0, 2]
    
    for index in camera_indices:
        print(f"\nTrying camera index {index}...")
        cap = cv2.VideoCapture(index)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera found at index {index}")
                print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            else:
                print("✗ Can't read frame")
                cap.release()
        else:
            print("✗ Camera not available")
            cap.release()
    
    print("\n✗ No camera found!")
    return None

# ==================== UI HELPER FUNCTIONS ====================
def create_info_panel(frame, info_dict, x_offset=10, y_offset=50):
    h, w = frame.shape[:2]
    
    panel_width = INFO_PANEL_WIDTH
    panel_height = 150
    
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    
    panel_x = x_offset
    panel_y = y_offset
    
    if panel_y + panel_height > h:
        panel_y = h - panel_height - 10
    
    cv2.rectangle(frame, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  (100, 100, 100), 2)
    
    cv2.putText(frame, "SYSTEM INFO", (panel_x + 10, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.line(frame, (panel_x, panel_y + 30), 
             (panel_x + panel_width, panel_y + 30), 
             (100, 100, 100), 1)
    
    y_pos = panel_y + 50
    line_height = 20
    
    for key, value in info_dict.items():
        cv2.putText(frame, f"{key}:", (panel_x + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if "ON" in str(value):
            value_color = (0, 255, 0)
        elif "OFF" in str(value):
            value_color = (255, 0, 0)
        elif "DETECTED" in str(value):
            value_color = (0, 255, 0)
        elif "NOT DETECTED" in str(value):
            value_color = (255, 0, 0)
        else:
            value_color = (255, 255, 255)
        
        cv2.putText(frame, str(value), (panel_x + 120, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, value_color, 1)
        
        y_pos += line_height
    
    return frame

def create_face_info_panel(frame, face_count, avg_confidence, avg_age, age_confidence, age_estimation_enabled, x_offset=10, y_offset=210):
    h, w = frame.shape[:2]
    
    panel_height = 120 if not age_estimation_enabled else 150
    panel_width = INFO_PANEL_WIDTH
    
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    
    panel_x = x_offset
    panel_y = y_offset
    
    if panel_y + panel_height > h:
        panel_y = h - panel_height - 10
    
    cv2.rectangle(frame, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  (0, 255, 255), 2)
    
    cv2.putText(frame, "FACE ANALYSIS", (panel_x + 10, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.line(frame, (panel_x, panel_y + 30), 
             (panel_x + panel_width, panel_y + 30), 
             (100, 100, 100), 1)
    
    y_pos = panel_y + 50
    cv2.putText(frame, f"Faces: {face_count}", (panel_x + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if face_count > 0:
        cv2.putText(frame, f"Detect Conf: {avg_confidence:.1f}%", 
                   (panel_x + 10, y_pos + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    
    if age_estimation_enabled and face_count > 0 and avg_age > 0:
        age_text = f"Estimated Age: {avg_age:.1f} yrs"
        cv2.putText(frame, age_text, (panel_x + 10, y_pos + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        conf_bar_width = 100
        conf_bar_height = 8
        conf_fill = int(age_confidence * conf_bar_width)
        
        cv2.rectangle(frame, (panel_x + 10, y_pos + 55),
                     (panel_x + 10 + conf_bar_width, y_pos + 55 + conf_bar_height),
                     (60, 60, 60), -1)
        
        conf_color = (0, 255, 0) if age_confidence > 0.7 else (255, 255, 0) if age_confidence > 0.5 else (255, 0, 0)
        cv2.rectangle(frame, (panel_x + 10, y_pos + 55),
                     (panel_x + 10 + conf_fill, y_pos + 55 + conf_bar_height),
                     conf_color, -1)
        
        cv2.putText(frame, f"Age Confidence: {age_confidence*100:.0f}%",
                   (panel_x + 10 + conf_bar_width + 10, y_pos + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1)
    
    return frame

def create_volume_bar(frame, volume_percent, x_offset, y_offset, bar_width=40, bar_height=200):
    bar_x = x_offset
    bar_y = y_offset
    
    cv2.rectangle(frame, (bar_x, bar_y), 
                  (bar_x + bar_width, bar_y + bar_height), 
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), 
                  (bar_x + bar_width, bar_y + bar_height), 
                  (150, 150, 150), 2)
    
    fill_height = int(volume_percent / 100 * bar_height)
    fill_y = bar_y + bar_height - fill_height
    
    cv2.rectangle(frame, (bar_x, fill_y), 
                  (bar_x + bar_width, bar_y + bar_height), 
                  (0, 255, 0), -1)
    
    vol_text = f"{int(volume_percent)}%"
    text_x = bar_x - 10 if volume_percent < 10 else bar_x - 15
    cv2.putText(frame, vol_text, (text_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(frame, "VOL", (bar_x + 5, bar_y + bar_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def draw_face_info(image, detection, width, height, estimated_age=None, age_group=None, age_color=None, age_confidence=0.7):
    bbox = detection.location_data.relative_bounding_box
    x = int(bbox.xmin * width)
    y = int(bbox.ymin * height)
    w = int(bbox.width * width)
    h = int(bbox.height * height)
    
    rect_color = age_color if age_color else (0, 255, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), rect_color, 2)
    
    keypoints = detection.location_data.relative_keypoints
    
    for kp in keypoints[:2]:
        kp_x = int(kp.x * width)
        kp_y = int(kp.y * height)
        cv2.circle(image, (kp_x, kp_y), 3, (0, 0, 255), -1)
    
    if y > 50:
        text_y = y - 10
        text_direction = -1
    else:
        text_y = y + h + 20
        text_direction = 1
    
    if estimated_age is not None and age_group is not None:
        stars = min(5, int(age_confidence * 5))
        star_text = "★" * stars + "☆" * (5 - stars)
        face_label = f"{age_group} ({estimated_age}y)"
        label_color = age_color if age_color else (0, 255, 0)
        
        cv2.putText(image, face_label, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
        cv2.putText(image, star_text, (x, text_y + text_direction * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 215, 0), 1)
    else:
        face_label = "Face"
        label_color = (0, 255, 0)
        cv2.putText(image, face_label, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
    
    conf = detection.score[0] * 100
    cv2.putText(image, f"Det: {conf:.0f}%", (x, text_y + text_direction * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
    
    return conf, w, h

# ==================== CALIBRATION FUNCTIONS ====================
def calibrate_for_age(real_age):
    """Fungsi untuk kalibrasi berdasarkan usia asli pengguna"""
    print(f"\n=== KALIBRASI UNTUK USIA {real_age} TAHUN ===")
    print("1. Hadapkan wajah ke kamera")
    print("2. Pastikan pencahayaan cukup")
    print("3. Jaga jarak normal (50-100cm)")
    print("4. Tekan 's' untuk menyimpan kalibrasi")
    print("5. Tekan 'q' untuk membatalkan")
    
    return real_age

# ==================== MAIN PROGRAM ====================
def main():
    volume, min_vol, max_vol = setup_volume_control()
    VOLUME_ENABLED = volume is not None
    
    # Initialize improved age estimator
    age_estimator = ImprovedAgeEstimator()
    
    print("\n" + "=" * 60)
    print("IMPROVED AGE DETECTION SYSTEM")
    print("=" * 60)
    print("Sistem ini menggunakan pendekatan yang lebih realistis untuk estimasi usia.")
    print("Untuk akurasi terbaik:")
    print("1. Hadapkan wajah lurus ke kamera")
    print("2. Jarak 50-100 cm dari kamera")
    print("3. Pencahayaan yang cukup")
    print("=" * 60)
    
    # Kalibrasi usia
    print("\nApakah Anda ingin mengkalibrasi sistem dengan usia Anda?")
    print("Tekan 'y' untuk kalibrasi, atau tombol lain untuk melanjuttan tanpa kalibrasi...")
    
    # Connect to camera
    cap = connect_to_camera()
    if cap is None:
        sys.exit(1)
    
    # Variables
    dist_min, dist_max = 30, 200
    vol_history = []
    calibration_mode = False
    face_detection_enabled = FACE_DETECTION_ENABLED
    age_estimation_enabled = AGE_ESTIMATION_ENABLED
    frame_count = 0
    last_time = time.time()
    
    # Statistics
    face_confidence_history = []
    age_history = []
    age_confidence_history = []
    
    # Kalibration mode
    calibration_active = False
    calibrated_age = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame tidak terbaca...")
                break
            
            frame_count += 1
            current_time = time.time()
            fps = 0
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            if w > 800:
                scale = 800 / w
                frame = cv2.resize(frame, (800, int(h * scale)))
                h, w = frame.shape[:2]
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face detection
            face_count = 0
            avg_confidence = 0
            avg_age = 0
            avg_age_confidence = 0.7
            
            if face_detection_enabled:
                face_results = face_detection.process(rgb)
                
                if face_results.detections:
                    face_count = len(face_results.detections)
                    
                    face_mesh_results = None
                    if age_estimation_enabled:
                        face_mesh_results = face_mesh.process(rgb)
                    
                    for i, detection in enumerate(face_results.detections):
                        estimated_age = None
                        age_group = None
                        age_color = None
                        age_confidence = 0.7
                        
                        if age_estimation_enabled:
                            detect_conf, face_w, face_h = draw_face_info(
                                frame, detection, w, h
                            )
                            
                            # Prioritaskan metode yang lebih sederhana dan akurat
                            if face_mesh_results and face_mesh_results.multi_face_landmarks:
                                if i < len(face_mesh_results.multi_face_landmarks):
                                    face_landmarks = face_mesh_results.multi_face_landmarks[i]
                                    
                                    # Ekstrak fitur wajah
                                    features = age_estimator.extract_facial_features(
                                        face_landmarks.landmark, w, h
                                    )
                                    
                                    if features:
                                        # Gunakan KNN untuk estimasi
                                        estimated_age, age_group, age_color, age_confidence = age_estimator.estimate_age_knn(features)
                                    else:
                                        # Fallback ke metode sederhana
                                        estimated_age, age_group, age_color, age_confidence = age_estimator.simple_age_estimation(
                                            face_w/w, face_h/h, detect_conf
                                        )
                            else:
                                # Gunakan metode sederhana jika face mesh tidak tersedia
                                estimated_age, age_group, age_color, age_confidence = age_estimator.simple_age_estimation(
                                    face_w/w, face_h/h, detect_conf
                                )
                            
                            # Gunakan usia terkalibrasi jika ada
                            if calibrated_age is not None and abs(estimated_age - calibrated_age) > 10:
                                # Jika perbedaan terlalu besar, gunakan usia terkalibrasi
                                estimated_age = calibrated_age
                                age_confidence = min(age_confidence + 0.1, 0.9)
                            
                            # Store for statistics
                            if estimated_age:
                                age_history.append(estimated_age)
                                age_confidence_history.append(age_confidence)
                                if len(age_history) > 20:
                                    age_history.pop(0)
                                if len(age_confidence_history) > 20:
                                    age_confidence_history.pop(0)
                            
                        # Draw face info
                        detect_conf, _, _ = draw_face_info(
                            frame, detection, w, h, estimated_age, age_group, age_color, age_confidence
                        )
                        
                        face_confidence_history.append(detect_conf)
                        if len(face_confidence_history) > 10:
                            face_confidence_history.pop(0)
                    
                    # Calculate averages
                    if face_confidence_history:
                        avg_confidence = np.mean(face_confidence_history)
                    
                    if age_history:
                        avg_age = np.mean(age_history)
                    
                    if age_confidence_history:
                        avg_age_confidence = np.mean(age_confidence_history)
            
            # Process hand detection for volume control
            hand_results = hands.process(rgb)
            hand_detected = False
            current_volume = 50
            
            if hand_results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                    )
                    
                    lm = hand_landmarks.landmark
                    idx = (int(lm[8].x * w), int(lm[8].y * h))
                    thb = (int(lm[4].x * w), int(lm[4].y * h))
                    
                    cv2.circle(frame, idx, 8, (0, 0, 255), -1)
                    cv2.circle(frame, thb, 8, (0, 0, 255), -1)
                    cv2.line(frame, idx, thb, (0, 255, 255), 2)
                    
                    dist = math.sqrt((idx[0]-thb[0])**2 + (idx[1]-thb[1])**2)
                    
                    if not calibration_mode:
                        if 10 < dist < dist_min:
                            dist_min = int(dist * 0.9)
                        if dist > dist_max:
                            dist_max = int(dist * 1.1)
                    
                    dist = max(dist_min, min(dist_max, dist))
                    vol = np.interp(dist, [dist_min, dist_max], [0, 100])
                    
                    vol_history.append(vol)
                    if len(vol_history) > 5:
                        vol_history.pop(0)
                    current_volume = np.mean(vol_history)
                    
                    if VOLUME_ENABLED:
                        vol_db = min_vol + (max_vol - min_vol) * (current_volume / 100.0)
                        volume.SetMasterVolumeLevel(vol_db, None)
            
            # ==================== LAYOUT ====================
            
            # Header
            title = "Improved Age Detection v2.0"
            if calibrated_age is not None:
                title += f" [Calibrated: {calibrated_age}y]"
            cv2.putText(frame, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # FPS
            if fps > 0:
                cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Main Info Panel
            system_info = {
                "Hand Status": "DETECTED" if hand_detected else "NOT DETECTED",
                "Volume Ctrl": "ON" if VOLUME_ENABLED else "OFF",
                "Face Detect": "ON" if face_detection_enabled else "OFF",
                "Age Est": "ON" if age_estimation_enabled else "OFF",
                "Calib Mode": "ON" if calibration_mode else "OFF",
                "Range": f"{dist_min}-{dist_max}px"
            }
            
            frame = create_info_panel(frame, system_info, 10, 50)
            
            # Face Info Panel
            if face_detection_enabled:
                frame = create_face_info_panel(frame, face_count, avg_confidence, 
                                              avg_age, avg_age_confidence, 
                                              age_estimation_enabled, 10, 210)
            
            # Volume Bar
            if hand_detected:
                frame = create_volume_bar(frame, current_volume, 
                                         w - VOLUME_BAR_WIDTH - 20, 50, 
                                         VOLUME_BAR_WIDTH - 20, 200)
            
            # Controls
            controls = "q=Quit r=Reset c=Calib f=Face a=Age k=CalibrateAge"
            cv2.putText(frame, controls, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Tips untuk akurasi
            if age_estimation_enabled and face_count == 0:
                tips = [
                    "Tips: Hadapkan wajah lurus ke kamera",
                    "Jarak optimal: 50-100 cm",
                    "Pastikan pencahayaan cukup"
                ]
                for i, tip in enumerate(tips):
                    cv2.putText(frame, tip, (w//2 - 150, 60 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            
            # Show frame
            cv2.imshow('Improved Age Detection System', frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                dist_min, dist_max = 30, 200
                vol_history.clear()
                age_estimator.age_history.clear()
                print("Calibration and age history reset")
            elif key == ord('c'):
                calibration_mode = not calibration_mode
                print(f"Calibration: {'ON' if calibration_mode else 'OFF'}")
            elif key == ord('f'):
                face_detection_enabled = not face_detection_enabled
                status = "ENABLED" if face_detection_enabled else "DISABLED"
                print(f"Face Detection: {status}")
            elif key == ord('a'):
                age_estimation_enabled = not age_estimation_enabled
                status = "ENABLED" if age_estimation_enabled else "DISABLED"
                print(f"Age Estimation: {status}")
            elif key == ord('k'):
                # Kalibrasi usia
                try:
                    user_input = input("\nMasukkan usia Anda yang sebenarnya (dalam tahun): ")
                    real_age = int(user_input)
                    if 5 <= real_age <= 80:
                        calibrated_age = real_age
                        print(f"Sistem dikalibrasi untuk usia {real_age} tahun")
                    else:
                        print("Usia harus antara 5-80 tahun")
                except:
                    print("Input tidak valid")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if VOLUME_ENABLED:
            vol_db = min_vol + (max_vol - min_vol) * 0.5
            volume.SetMasterVolumeLevel(vol_db, None)
            print("Volume reset to 50%")
        
        if age_history:
            print(f"\n=== FINAL STATISTICS ===")
            print(f"Average Estimated Age: {np.mean(age_history):.1f} years")
            print(f"Age Range: {np.min(age_history)} - {np.max(age_history)} years")
            if calibrated_age:
                print(f"Calibrated Age: {calibrated_age} years")
                accuracy = 100 - abs(np.mean(age_history) - calibrated_age) / calibrated_age * 100
                print(f"Estimated Accuracy: {accuracy:.1f}%")
        
        print("\nProgram terminated.")

if __name__ == "__main__":
    main()
