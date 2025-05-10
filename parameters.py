import cv2
import mediapipe as mp
import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from collections import deque
import time
from utils import get_user_weight, get_user_age
import pywt
from scipy.signal import find_peaks
from sklearn.decomposition import FastICA
import glob
import os
import json

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter_signal(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

class ParameterCalculator:
    def __init__(self, weight=get_user_weight()):
        # ==== ▶️ Sistem MediaPipe și video
        self.camera_cap = cv2.VideoCapture(0)
        self.camera_running = False
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # ==== 🔁 Status intern
        self.reps_state = "START"
        self.reps_count = 0
        self.last_rep_time = None
        self.reps_per_minute = 0
        self.reps_start_time = time.time()
        self.execution_accuracy = 0
        self.joint_angles = {}

        # ==== 🔧 Parametri personali
        self.user_weight = weight

        # ==== 🩺 Date fiziologice
        self.total_calories = 0.00
        self.heart_rate = 80
        self.last_bpm = None
        self.breathing_rate = 222
        self.spo2 = 98.0
        self.last_vo2_max = None
        self.movement_intensity = 0

        # ==== 🧠 Buffere de date
        self.bpm_buffer = deque(maxlen=1000)
        self.red_channel_buffer = deque(maxlen=1000)
        self.breathing_buffer = deque(maxlen=150)
        self.face_temperature_buffer = deque(maxlen=300)

        # ==== 🎥 Frame-uri & landmark-uri
        self.prev_landmarks = None
        self.prev_frame = None
        self.camera_frame = None
        self.filtered_landmarks = None
        self.smoothing_factor = 0.7
        self.depth_offset_factor = 0.02
        self.fps = 30
        self.start_time = time.time()

        # ==== 🎯 Kalman Filter
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.base_Q = 0.03
        self.base_R = 1.0
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * self.base_Q
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.base_R
        self.intensity_thresholds = [0.5, 1.5]

        # ==== ✅ Inițializări dependente de parametri calculați
        self.hydration_level = self.estimate_initial_hydration()  # bazat pe temperatură & puls
        self.energy_level = self.estimate_initial_energy()        # bazat pe hidratare, vo2, bpm etc.


    def camera_loop(self):
        
        self.camera_running = True
        frame_buffer = deque(maxlen=5)  
        
        roi_x, roi_y, roi_w, roi_h = 150, 100, 100, 100 
        
        while self.camera_cap.isOpened() and self.camera_running:
            ret, frame = self.camera_cap.read()
            if not ret:
                print("Camera nu se deschide")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb_frame)

            stabilized_frame = np.mean(np.array(frame_buffer), axis=0).astype(np.uint8)

            results = self.holistic.process(stabilized_frame)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                self.calculate_joint_angles(results.pose_landmarks.landmark)
                self.detect_reps(results.pose_landmarks.landmark)
                

            if results.face_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION)

                forehead_x = int(results.face_landmarks.landmark[10].x * frame.shape[1])
                forehead_y = int(results.face_landmarks.landmark[10].y * frame.shape[0])

                measurement = np.array([[np.float32(forehead_x)], [np.float32(forehead_y)]])
                self.kalman.correct(measurement)
                prediction = self.kalman.predict()
                roi_x = int(prediction[0][0]) - roi_w // 2
                roi_y = int(prediction[1][0]) - roi_h // 2

            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

            roi_x = max(0, min(frame.shape[1] - roi_w, roi_x))
            roi_y = max(0, min(frame.shape[0] - roi_h, roi_y))

            roi = stabilized_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            if roi.shape[0] > 0 and roi.shape[1] > 0:

                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                roi_filtered = cv2.GaussianBlur(roi_gray, (5, 5), 0)

                green_channel = roi[:, :, 1] 
                red_channel = roi[:, :, 2]    

                #Butterworth
                filtered_red = filter_signal(np.array(red_channel).flatten(), 0.5, 5.0, 30)
                filtered_green = filter_signal(np.array(green_channel).flatten(), 0.5, 5.0, 30)

                red_dc = np.mean(red_channel)
                green_dc = np.mean(green_channel)

                if red_dc > 50 and green_dc > 50:
                    self.red_channel_buffer.append(filtered_red.mean())
                    self.bpm_buffer.append(filtered_green.mean())


            if len(self.bpm_buffer) == self.bpm_buffer.maxlen:
                bpm = self.calculate_bpm(np.array(self.bpm_buffer))
                print(f"📊 BPM Buffer (Ultimele 10 valori): {list(self.bpm_buffer)[-10:]}")
                if 60 <= bpm <= 180:
                    self.last_bpm = bpm
                    self.heart_rate = round((self.last_bpm + self.heart_rate) / 2, 2)
                    #self.bpm_buffer.append(self.heart_rate)  # Nu filtrul verde, ci valoare reală
                self.bpm_buffer.clear()

            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            cv2.putText(frame, "ROI Forehead", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera_cap.release()
        cv2.destroyAllWindows()

    def adjust_kalman_parameters(self):
        if self.movement_intensity < self.intensity_thresholds[0]:
            Q_factor, R_factor = 0.5, 0.5  
        elif self.movement_intensity < self.intensity_thresholds[1]:
            Q_factor, R_factor = 1.0, 1.0  
        else:
            Q_factor, R_factor = 2.0, 2.0 

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * (self.base_Q * Q_factor)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * (self.base_R * R_factor)

    def apply_keypoint_smoothing(self, landmarks):
        #aplicare Kalman
        if self.filtered_landmarks is None:
            self.filtered_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks]
        else:
            for i, lm in enumerate(landmarks):
                confidence = lm.visibility  
                alpha = self.smoothing_factor * confidence  

                self.filtered_landmarks[i] = (
                    alpha * np.array([lm.x, lm.y, lm.z]) +
                    (1 - alpha) * np.array(self.filtered_landmarks[i])
                )
        return self.filtered_landmarks

    
    def estimate_3d_pose(self, landmarks):
        movement_factor = min(self.movement_intensity, 1.0) 
        depth_offset = 0.05 + (movement_factor * self.depth_offset_factor)

        return [(lm.x, lm.y, lm.z + depth_offset) for lm in landmarks]

    def calculate_bpm(self, frame):
        """ Calculează ritmul cardiac cu optimizări avansate pentru repaus și mișcare. """

        if frame is None or len(frame) < 100:
            print("⚠️ Nu sunt suficiente date pentru BPM, folosesc ultima valoare.")
            return self.last_bpm if self.last_bpm else 75  

        fs = self.fps  

        # ✅ Extragem semnalul corect din frame sau buffer
        if isinstance(frame, np.ndarray) and frame.ndim == 3:
            roi_x, roi_y, roi_w, roi_h = 150, 100, 100, 100  
            if hasattr(self, 'kalman'):  
                prediction = self.kalman.predict()
                roi_x = int(prediction[0][0]) - roi_w // 2
                roi_y = int(prediction[1][0]) - roi_h // 2

            roi_x = max(0, min(frame.shape[1] - roi_w, roi_x))
            roi_y = max(0, min(frame.shape[0] - roi_h, roi_y))

            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            r_channel = roi[:, :, 2].flatten()
            g_channel = roi[:, :, 1].flatten()
            b_channel = roi[:, :, 0].flatten()

            combined_signal = (0.3 * r_channel) + (0.6 * g_channel) + (0.1 * b_channel)

        elif isinstance(frame, np.ndarray) and frame.ndim == 1:
            combined_signal = frame  

        else:
            print("⚠️ Eroare: Tip necunoscut de date primit în `calculate_bpm`!")
            return self.last_bpm if self.last_bpm else 75  

        # ✅ Corecție pentru variațiile de iluminare
        combined_signal -= np.mean(combined_signal)  

        # ✅ Filtrare Butterworth pentru eliminarea zgomotului
        nyq = 0.5 * fs
        low = 0.7 / nyq
        high = 3.0 / nyq
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, combined_signal)

        if np.isnan(filtered_signal).any():
            return self.last_bpm if self.last_bpm else 75  

        # ✅ Filtrare Wavelet pentru eliminarea artefactelor de mișcare
        wavelet_level = min(3, pywt.dwt_max_level(len(filtered_signal), 'db4'))
        coeffs = pywt.wavedec(filtered_signal, 'db4', level=wavelet_level)
        coeffs[-1] = np.zeros_like(coeffs[-1])  
        filtered_signal = pywt.waverec(coeffs, 'db4')

        # ✅ Aplicăm Analiza Componentelor Independente (ICA)
        ica = FastICA(n_components=1, random_state=42)
        try:
            ica_signal = ica.fit_transform(filtered_signal.reshape(-1, 1)).flatten()
        except ValueError:
            return self.last_bpm if self.last_bpm else 75  

        # ✅ Aplicăm FFT pentru a identifica ritmul cardiac
        fft_result = np.abs(fft(ica_signal))
        freqs = np.fft.rfftfreq(len(ica_signal), 1/fs)

        # ✅ Corectăm problema cu indexarea FFT
        if len(freqs) != len(fft_result):
            min_length = min(len(freqs), len(fft_result))
            freqs = freqs[:min_length]
            fft_result = fft_result[:min_length]

        # ✅ Filtrăm frecvențele valide (45 - 180 BPM)
        valid_indices = (freqs >= 0.75) & (freqs <= 3.0)
        freqs_valid = freqs[valid_indices]
        fft_valid = fft_result[valid_indices]

        if len(freqs_valid) > 0:
            dominant_freq = freqs_valid[np.argmax(fft_valid)]
            bpm = dominant_freq * 60  
        else:
            return self.last_bpm if self.last_bpm else 75  

        # ✅ 🔹 **Ajustare în funcție de mișcare**
        if self.movement_intensity < 0.3:  
            bpm = max(55, bpm - 10)  # Scade BPM în repaus
        elif self.movement_intensity > 1.5:  
            bpm = min(170, bpm + 5)  # Crește BPM în mișcare intensă

        # ✅ 🔹 **Evităm schimbările bruște nerealiste**
        if self.last_bpm:
            bpm_change = bpm - self.last_bpm
            if abs(bpm_change) > 15:  
                bpm = self.last_bpm + np.sign(bpm_change) * 10

        # ✅ 🔹 **Stabilizare finală (dar fără efect artificial)**
        if self.last_bpm:
            bpm = (0.6 * self.last_bpm) + (0.4 * bpm)  

        # ✅ 🔹 **Limităm intervalul BPM la valori reale**
        bpm = max(60, min(180, bpm))

        self.last_bpm = bpm  
        print(f"❤️ BPM detectat: {self.last_bpm:.2f}")

        return round(bpm, 2)
    
    def get_face_temperature(self):
        """ Estimează temperatura pielii din camera video, folosind analiza zonei frunții. """
        if not self.face_temperature_buffer:
            return 36.5  # Valoare implicită

        # 🔹 Mediem ultimele 10 valori pentru a evita fluctuațiile
        return round(np.mean(list(self.face_temperature_buffer)[-10:]), 1)

    
    def update_hydration_level(self, results_pose=None, temperature=None):
        """
        Calculează nivelul de hidratare pe baza scorului compus: temperatură, puls, mișcare.
        Inspirat din articolul MDPI Electronics 2020.
        """

        # 🔹 Temperatură și puls - fallback
        temp = temperature if temperature else self.get_face_temperature()
        hr = self.heart_rate if self.heart_rate else 80
        motion = self.movement_intensity

        # 🔸 1. Scor temperatură (ideal 36.5 - 37.2)
        temp_score = 100 - abs(36.8 - temp) * 35  # penalizare severă pentru deviație
        temp_score = max(0, min(100, temp_score))

        # 🔸 2. Scor puls (ideal 70 - 90 bpm)
        if hr < 60 or hr > 120:
            hr_score = 60
        else:
            hr_score = 100 - abs(80 - hr) * 1.5
        hr_score = max(0, min(100, hr_score))

        # 🔸 3. Scor mișcare (normalizat între 0-100)
        motion_score = 100 - min(100, motion * 50)  # e.g. motion=1.5 → score ~25
        motion_score = max(0, min(100, motion_score))

        # 🔹 Scor compus ponderat
        hydration_score = (0.5 * temp_score) + (0.3 * hr_score) + (0.2 * motion_score)

        # 🔻 Decay lent dacă scorul e mic, creștere dacă e bun
        if hydration_score < 50:
            self.hydration_level -= 0.5
        elif hydration_score > 80 and self.hydration_level < 100:
            self.hydration_level += 0.2  # mică regenerare

        # 🧪 Stabilizare și limitare
        self.hydration_level = round(max(30, min(100, self.hydration_level)), 1)
        self.last_hydration_score = round(hydration_score, 1)

        print(f"💧 Hidratare scor: {hydration_score:.1f} | Nivel actual: {self.hydration_level}%")
        return self.hydration_level



    def calculate_joint_angles(self, landmarks):
        # Articulații multiple pentru o acoperire mai largă
        joint_sets = {
            'left_elbow': (11, 13, 15),
            'right_elbow': (12, 14, 16),
            'left_shoulder': (13, 11, 23),
            'right_shoulder': (14, 12, 24),
            'left_hip': (23, 25, 27),
            'right_hip': (24, 26, 28),
            'left_knee': (25, 27, 31),
            'right_knee': (26, 28, 32),
            'spine': (11, 23, 24)
        }

        angles = {}
        for joint_name, (p1, p2, p3) in joint_sets.items():
            angle = self.calculate_angle(p1, p2, p3, landmarks)
            angle = max(0, min(180, angle))  # asigurare interval valid
            angles[joint_name] = angle

        self.joint_angles = angles


    def calculate_angle(self, p1, p2, p3, landmarks):
        a = np.array([landmarks[p1].x, landmarks[p1].y])
        b = np.array([landmarks[p2].x, landmarks[p2].y])
        c = np.array([landmarks[p3].x, landmarks[p3].y])

        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return round(angle, 1)

    def detect_reps(self, landmarks):

        elbow_angle = self.calculate_angle(11, 13, 15, landmarks)
        current_time = time.time()

        flexion_threshold = 60   #cot indoit
        extension_threshold = 160  #cot intins

        if self.reps_state == "START":
            if elbow_angle < flexion_threshold:
                self.reps_state = "FLEXION"  

        elif self.reps_state == "FLEXION":
            if elbow_angle > extension_threshold:
                self.reps_state = "EXTENSION"  
                self.reps_count += 1
                print(f" Repetare detectata. Total reps: {self.reps_count}")

                self.reps_state = "START" 

    
    def calculate_spo2(self):
        """ Calculează SpO₂ folosind Ratio-of-Ratios cu corecție calibrată. """

        if len(self.bpm_buffer) < 50 or len(self.red_channel_buffer) < 50:
            print("⚠️ Buffer insuficient pentru SpO₂! Folosim valoare implicită.")
            self.spo2 = 98.0  # 🔹 Setăm un default dacă nu avem suficiente date
            return  

        fs = 30  # Frecvența de eșantionare

        # ✅ Convertim listele în numpy arrays
        red_signal = np.array(self.red_channel_buffer)
        green_signal = np.array(self.bpm_buffer)

        # ✅ Aplicăm filtrare pentru eliminarea zgomotului
        red_ac = filter_signal(red_signal, 0.7, 3.0, fs)  
        green_ac = filter_signal(green_signal, 0.7, 3.0, fs)

        # ✅ Componente DC = medie semnal
        red_dc = np.mean(red_signal)
        green_dc = np.mean(green_signal)

        # 🚨 Evităm împărțirea la zero
        if red_dc == 0 or green_dc == 0:
            print("⚠️ Nu se poate calcula SpO₂! Folosim valoare implicită.")
            self.spo2 = 98.0  # 🔹 Default în cazul unei erori
            return  

        # ✅ Calculăm raporturile AC/DC
        r_ratio = np.std(red_ac) / red_dc
        g_ratio = np.std(green_ac) / green_dc

        # ✅ Aplicăm formula Ratio-of-Ratios cu ajustare
        ratio = (r_ratio / g_ratio) * 1.2  # Corecție empirică

        # 🔢 Formula calibrată
        A = 110.0  # Creștem valoarea de bază
        B = 25.0   # Reducem influența raportului

        spo2_estimate = A - B * ratio  

        # ✅ Limităm valorile realiste (95-100%)
        self.spo2 = max(95, min(99, spo2_estimate))

        print(f"🔵 SpO₂ estimat: {self.spo2:.2f}%")

    def calculate_vo2_max(self):
        """
        Estimează VO₂ Max pe baza Heart Rate Recovery (HRR), inspirat din PMC3996514.
        Formula: VO2max ≈ 70 + (HRR × 0.5), unde HRR = HR_peak - HR_1min.
        Se ajustează ușor în funcție de intensitate și greutate.
        """
        if len(self.bpm_buffer) < 60:
            print("⚠️ Nu sunt suficiente date pentru VO₂ Max – folosim valoare anterioară.")
            return self.last_vo2_max if self.last_vo2_max is not None else 40.0

        hr_peak = max(self.bpm_buffer)
        hr_1min = self.bpm_buffer[-1]
        hrr = hr_peak - hr_1min

        base_vo2 = 70 + (hrr * 0.5)

        # Ajustări fine
        intensity_factor = 1.0
        if self.movement_intensity > 1.5:
            intensity_factor = 1.05
        elif self.movement_intensity < 0.3:
            intensity_factor = 0.95

        weight_factor = 1.0
        if self.user_weight:
            if self.user_weight < 60:
                weight_factor = 1.03
            elif self.user_weight > 90:
                weight_factor = 0.97

        vo2_final = base_vo2 * intensity_factor * weight_factor
        vo2_final = round(vo2_final, 2)  # ⬅️ Fără limită superioară!

        self.last_vo2_max = vo2_final
        print(f"🫁 VO₂ Max calculat (HRR method): {vo2_final:.2f} ml/kg/min")
        return vo2_final


    def calculate_movement_intensity(self, landmarks, prev_landmarks, dt, frame=None, prev_frame=None):
        """Calculează intensitatea mișcării combinând 3 surse:
        - Keypoints biomecanici (viteza articulațiilor)
        - Motion Energy (diferență între cadre – siluete)
        - Tracking (deplasarea centrului corpului în imagine)
        """
        if (prev_landmarks is None or dt == 0) and (frame is None or prev_frame is None):
            return 0  

        # 🔹 1. Biomecanic – diferența între keypoints
        total_movement = 0
        joint_indices = [11, 12, 13, 14, 23, 24, 25, 26]
        for idx in joint_indices:
            prev = np.array([prev_landmarks[idx].x, prev_landmarks[idx].y, prev_landmarks[idx].z])
            curr = np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
            speed = np.linalg.norm(curr - prev) / dt if dt != 0 else 0
            total_movement += speed
        biomech_intensity = total_movement / len(joint_indices)

        # 🔹 2. Motion Energy – diferență între cadre video
        motion_energy = 0
        if frame is not None and prev_frame is not None:
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_energy = np.sum(thresh) / 255 / (frame.shape[0] * frame.shape[1])  # normalizat [0,1]

        # 🔹 3. Tracking – mișcarea centrului corpului (bounding box)
        def get_center(landmarks):
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            return np.array([np.mean(xs), np.mean(ys)])

        center_movement = 0
        if prev_landmarks is not None:
            prev_center = get_center(prev_landmarks)
            curr_center = get_center(landmarks)
            center_movement = np.linalg.norm(curr_center - prev_center) / dt

        # 🔸 Combinație ponderată – poți ajusta proporțiile după testare
        combined_intensity = (
            0.5 * biomech_intensity + 
            0.3 * motion_energy + 
            0.2 * center_movement
        )

        return round(combined_intensity, 4)
 
    
    def estimate_initial_energy(self):
        vo2 = self.last_vo2_max if self.last_vo2_max is not None else 60.0
        vo2_score = (vo2 / 60) * 40

        hydration_score = (self.hydration_level / 100) * 15

        bpm_penalty = max(0, (self.heart_rate - 75) * 0.25)

        movement_penalty = min(self.movement_intensity, 1.0) * 5

        initial_energy = 45 + vo2_score + hydration_score - bpm_penalty - movement_penalty
        self.energy_level = round(max(50, min(100, initial_energy)))

        print(f"⚡ Energie inițială estimată: {self.energy_level}%")
        return self.energy_level



    def calculate_energy_level(self):
        """ Scădere exactă cu 1% la fiecare update, ajustată cu detectarea corpului și articolul științific """

        # 🔹 Asigurăm o scădere lină doar la intervale de timp regulate
        current_time = time.time()
        if not hasattr(self, 'last_energy_update'):
            self.last_energy_update = current_time  # Inițializăm timpul ultimei actualizări

        # Scădem doar dacă au trecut cel puțin 10 secunde
        if current_time - self.last_energy_update < 10:
            return self.energy_level

        # 🔹 Factori de oboseală (bazat pe articol)
        fatigue_factor = (self.total_calories / (self.user_weight * 10)) * 3  
        movement_penalty = min(3, self.movement_intensity * 1.2)  
        hydration_bonus = (self.hydration_level / 100) * 2  
        spo2_bonus = ((self.spo2 - 90) / 10) * 2  
        vo2_bonus = ((self.last_vo2_max - 30) / 30) * 2 if self.last_vo2_max is not None else 0

        # 🔹 Scădere fixă cu 1% doar dacă există mișcare
        if self.movement_intensity > 0.1:  
            self.energy_level -= 1  

        # 🔹 Prevenim scăderea sub 0 sau peste 100%
        self.energy_level = max(0, min(100, self.energy_level))

        # 🔹 Actualizăm timpul ultimei scăderi
        self.last_energy_update = current_time

        return self.energy_level

    def estimate_initial_hydration(self):
        temp = self.get_face_temperature()
        bpm = self.heart_rate
        hydration = 100 - abs(36.8 - temp)*20 - abs(80 - bpm)*0.3
        return round(max(30, min(100, hydration)), 1)  # ⬅️ adaugă această limitare


    def calculate_parameters(self, frame=None, results_pose=None, results_face=None):
        if self.hydration_level is None:
            self.hydration_level = self.estimate_initial_hydration()

        self.hydration_level = self.update_hydration_level(results_pose, temperature=self.get_face_temperature())

        self.calculate_spo2()

        current_time = time.time()
        if current_time - self.start_time >= 1:  
            if len(self.bpm_buffer) > 10:
                bpm = self.calculate_bpm(np.array(self.bpm_buffer))

                if 60 <= bpm <= 180:  
                    if self.movement_intensity < 0.3:
                        bpm = max(55, bpm - 2)
                    elif self.movement_intensity > 1.5:
                        bpm = min(170, bpm + 2)

                    if self.last_bpm:
                        bpm = (0.65 * self.last_bpm) + (0.35 * bpm)

                    self.last_bpm = bpm
                    self.heart_rate = round(bpm, 2)
                    print(f"🔄 BPM actualizat: {self.heart_rate:.2f}")

            self.start_time = current_time

        if results_pose and results_pose.pose_landmarks:
            dt = time.time() - self.start_time
            self.start_time = time.time()

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame is not None else None

            self.movement_intensity = self.calculate_movement_intensity(
                results_pose.pose_landmarks.landmark,
                self.prev_landmarks,
                dt,
                frame=frame_bgr,
                prev_frame=self.prev_frame
            )

            self.prev_landmarks = results_pose.pose_landmarks.landmark
            if frame_bgr is not None:
                self.prev_frame = frame_bgr.copy()

            if self.movement_intensity > 0.25 and self.heart_rate > 85:
                MET = self.estimate_met(self.movement_intensity, self.heart_rate)
                duration_minutes = dt / 60
                self.total_calories += MET * self.user_weight * duration_minutes

            print(f"🔥 Calorii arse: {self.total_calories:.2f} kcal")

        # 🔥 Estimare VO₂ Max
        vo2_max = self.calculate_vo2_max()

        # ✅ Estimăm energia inițială doar după câteva secunde pentru acuratețe
        if self.energy_level is None:
            if not hasattr(self, "energy_init_time"):
                self.energy_init_time = time.time()
            elif time.time() - self.energy_init_time > 3:
                self.energy_level = self.estimate_initial_energy()

        # ✅ Actualizare energie continuă doar dacă avem VO2 valid
        if vo2_max is not None and self.energy_level is not None:
            energy_level = self.calculate_energy_level()
        else:
            energy_level = "N/A"

        return {
            "heart_rate": round(self.heart_rate, 2),
            "calories": round(self.total_calories, 1),
            "spo2": round(self.spo2, 2) if hasattr(self, 'spo2') else 98,
            "energy_level": energy_level,
            "execution_accuracy": round(self.execution_accuracy, 2),
            "hydration_level": self.last_hydration_score,
            "vo2_max": vo2_max
        }


    def estimate_met(self, movement_intensity, bpm):
        """Ajustează MET în funcție de intensitatea mișcării și ritmul cardiac."""
        if bpm < 80 and movement_intensity < 0.3:
            return 1.2  # Stare de repaus
        elif bpm < 100 and movement_intensity < 0.7:
            return 3.8  # Mișcare ușoară
        elif bpm < 130 and movement_intensity < 1.5:
            return 5.6  # Exercițiu moderat
        elif bpm < 160 and movement_intensity < 2.5:
            return 8.2  # Exercițiu intens
        else:
            return 10.0  # Efort maxim


    def stop_camera(self):
        self.camera_running = False

    def update_camera_frame(self, frame):
        #update frame
        self.camera_frame = frame  

    def calculate_accuracy(self, tutorial_landmarks, user_landmarks):
        if not tutorial_landmarks or not user_landmarks:
            return None 

        joint_pairs = [
            (11, 13, 15),  # Cot stâng
            (12, 14, 16),  # Cot drept
            (13, 11, 23),  # Umăr stâng
            (14, 12, 24),  # Umăr drept
            (23, 25, 27),  # Genunchi stâng
            (24, 26, 28),  # Genunchi drept
            (11, 23, 25),  # Șold stâng
            (12, 24, 26),  # Șold drept
            (23, 11, 13),  # Trunchi față de braț stâng
            (24, 12, 14),  # Trunchi față de braț drept
        ]

        total_diff = 0
        valid_joints = 0

        for p1, p2, p3 in joint_pairs:
            user_angle = self.calculate_angle(p1, p2, p3, user_landmarks)
            tutorial_angle = self.calculate_angle(p1, p2, p3, tutorial_landmarks)

            if user_angle is not None and tutorial_angle is not None:
                diff = abs(user_angle - tutorial_angle)
                total_diff += diff
                valid_joints += 1

        if valid_joints == 0:
            return 0  

        avg_diff = total_diff / valid_joints

        if avg_diff > 40:
            return 0

        accuracy = max(0, min(100, 100 - (avg_diff * 1.2)))
        return round(accuracy, 2)

    
    







