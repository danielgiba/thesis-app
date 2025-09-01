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
        self.camera_cap = cv2.VideoCapture(0)

        self.camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

        self.camera_running = False
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.reps_state = "START"
        self.reps_count = 0
        self.last_rep_time = None
        self.reps_per_minute = 0
        self.reps_start_time = time.time()
        self.execution_accuracy = 0
        self.joint_angles = {}

        self.user_weight = weight

        #date init
        self.total_calories = 0.00
        self.heart_rate = 80
        self.last_bpm = None
        self.breathing_rate = 222
        self.spo2 = 98.0
        self.last_vo2_max = None
        self.movement_intensity = 0

        #buffere de date
        self.bpm_buffer = deque(maxlen=1000)
        self.red_channel_buffer = deque(maxlen=1000)
        self.breathing_buffer = deque(maxlen=150)
        self.face_temperature_buffer = deque(maxlen=300)

        #frameuri&landmarkuri
        self.prev_landmarks = None
        self.prev_frame = None
        self.camera_frame = None
        self.filtered_landmarks = None
        self.smoothing_factor = 0.7
        self.depth_offset_factor = 0.02
        self.fps = 30
        self.start_time = time.time()

        #filtru kalman
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

        #init
        self.hydration_level = self.estimate_initial_hydration()  
        self.energy_level = self.estimate_initial_energy() 
        #self.last_hydration_level = self.hydration_level   
        self.hydration_drain = 0.0   


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
                print(f"BPM Buffer (ultimele 10 valori): {list(self.bpm_buffer)[-10:]}")
                if 60 <= bpm <= 180:
                    self.last_bpm = bpm
                    self.heart_rate = round((self.last_bpm + self.heart_rate) / 2, 2)
                    #self.bpm_buffer.append(self.heart_rate) 
                self.bpm_buffer.clear()

            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            cv2.putText(frame, "ROI Forehead", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            #cv2.imshow("Camera Feed", frame)
            self.camera_frame = frame.copy()
            #time.sleep(1 / self.fps)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            if not self.camera_running:
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

        if frame is None or len(frame) < 100:
            print("Nu sunt suficiente date pentru BPM, folosesc ultima valoare")
            return self.last_bpm if self.last_bpm else 75  

        fs = self.fps  

        #extragere semnal bun
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
            print("Eroare tip data in 'calculate_bpm'")
            return self.last_bpm if self.last_bpm else 75  

        #corectie pentru variatiile de iluminare
        combined_signal -= np.mean(combined_signal)  

        #filtrare Butterworth pentru eliminarea zgomotului
        nyq = 0.5 * fs
        low = 0.7 / nyq
        high = 3.0 / nyq
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, combined_signal)

        if np.isnan(filtered_signal).any():
            return self.last_bpm if self.last_bpm else 75  

        #filtrarea wavelet pentru eliminarea artefactelor de miscare
        wavelet_level = min(3, pywt.dwt_max_level(len(filtered_signal), 'db4'))
        coeffs = pywt.wavedec(filtered_signal, 'db4', level=wavelet_level)
        coeffs[-1] = np.zeros_like(coeffs[-1])  
        filtered_signal = pywt.waverec(coeffs, 'db4')

        #aplicare ICA
        ica = FastICA(n_components=1, random_state=42)
        try:
            ica_signal = ica.fit_transform(filtered_signal.reshape(-1, 1)).flatten()
        except ValueError:
            return self.last_bpm if self.last_bpm else 75  

        #FFT pt puls
        fft_result = np.abs(fft(ica_signal))
        freqs = np.fft.rfftfreq(len(ica_signal), 1/fs)

        # corectare cu indexarea FFT
        if len(freqs) != len(fft_result):
            min_length = min(len(freqs), len(fft_result))
            freqs = freqs[:min_length]
            fft_result = fft_result[:min_length]

        #filtrarea frecv
        valid_indices = (freqs >= 0.75) & (freqs <= 3.0)
        freqs_valid = freqs[valid_indices]
        fft_valid = fft_result[valid_indices]

        if len(freqs_valid) > 0:
            dominant_freq = freqs_valid[np.argmax(fft_valid)]
            bpm = dominant_freq * 60  
        else:
            return self.last_bpm if self.last_bpm else 75  

        #ajustare cu miscare
        if self.movement_intensity < 0.3:  
            bpm = max(55, bpm - 10)  #bpm in repaus
        elif self.movement_intensity > 1.5:  
            bpm = min(170, bpm + 5)  #bpm ca intensitate

        #de evitat valorile nerealiste
        if self.last_bpm:
            bpm_change = bpm - self.last_bpm
            if abs(bpm_change) > 15:  
                bpm = self.last_bpm + np.sign(bpm_change) * 10

        #stabilizare finala
        if self.last_bpm:
            bpm = (0.6 * self.last_bpm) + (0.4 * bpm)  

        #limitare reala 
        bpm = max(60, min(180, bpm))

        self.last_bpm = bpm  
        print(f"BPM detectat: {self.last_bpm:.2f}")

        return round(bpm, 2)
    
    def get_face_temperature(self):
        if not self.face_temperature_buffer:
            return 36.5  

        #mediere fluctuatii
        return round(np.mean(list(self.face_temperature_buffer)[-10:]), 1)

    
    def update_hydration_level(self, results_pose=None, temperature=None):
        #calc nivelul de hidratare pe baza scorului cu temperatura, puls & miscare

        temp = temperature if temperature else self.get_face_temperature()
        hr = self.heart_rate if self.heart_rate else 80
        motion = self.movement_intensity

        #temp -> 36.5-37.2
        temp_score = 100 - abs(36.8 - temp) * 35 
        temp_score = max(0, min(100, temp_score))

        #pulsul -> 70-90
        if hr < 60 or hr > 120:
            hr_score = 60
        else:
            hr_score = 100 - abs(80 - hr) * 1.5
        hr_score = max(0, min(100, hr_score))

        motion_score = 100 - min(100, motion * 50) 
        motion_score = max(0, min(100, motion_score))

        hydration_score = (0.5 * temp_score) + (0.3 * hr_score) + (0.2 * motion_score)

        drain = 0.0

        if hydration_score < 60:
            drain += 0.1
        if self.heart_rate > 100:
            drain += 0.05
        if self.movement_intensity > 0.5:
            drain += 0.1

        self.hydration_drain = drain

        #check
        if hydration_score > 85 and self.hydration_level < 100:
            self.hydration_level += 0.1 
        self.hydration_level -= self.hydration_drain

        #stabilizare&limitare
        self.hydration_level = round(max(30, min(100, self.hydration_level)), 1)
        self.last_hydration_score = round(hydration_score, 1)

        print(f"Hidratare scor: {hydration_score:.1f} & Nivel actual: {self.hydration_level}%")
        return self.hydration_level



    def calculate_joint_angles(self, landmarks):
        #articulatiile
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
            #unghi realizabil
            angle = max(0, min(180, angle))  
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
        if len(self.bpm_buffer) < 50 or len(self.red_channel_buffer) < 50:
            print("Buffer insuficient pentru SpO2 => folosim valoare implicita")
            self.spo2 = 98.0 
            return  

        fs = 30 

        #convertire in arrays
        red_signal = np.array(self.red_channel_buffer)
        green_signal = np.array(self.bpm_buffer)

        #filtrarea eliminarii zgomotului
        red_ac = filter_signal(red_signal, 0.7, 3.0, fs)  
        green_ac = filter_signal(green_signal, 0.7, 3.0, fs)

        #DC ca media de semnal
        red_dc = np.mean(red_signal)
        green_dc = np.mean(green_signal)
        #sa nu impart la 0
        if red_dc == 0 or green_dc == 0:
            print("Nu se poate calcula SpO₂ => folosim valoare implicita")
            self.spo2 = 98.0  #default in caz de ceva
            return  

        #calc raporturile AC/DC
        r_ratio = np.std(red_ac) / red_dc
        g_ratio = np.std(green_ac) / green_dc

        #formula ratio-of-ratios + empiric
        ratio = (r_ratio / g_ratio) * 1.2  

        #coeficienti din art
        A = 110.0  
        B = 25.0  

        spo2_estimate = A - B * ratio  
        #limitare : 95-99
        if np.isnan(spo2_estimate) or spo2_estimate < 96:
            self.spo2 = 98.0  
        else:
            self.spo2 = round(min(99, spo2_estimate), 1)

        print(f"SpO2 estimat: {self.spo2:.2f}%")

    def calculate_vo2_max(self):
        if len(self.bpm_buffer) < 60:
            print("Nu sunt suficiente date pentru VO2Max => folosim valoare anterioara")
            return self.last_vo2_max if self.last_vo2_max is not None else 40.0

        hr_peak = max(self.bpm_buffer)
        hr_1min = self.bpm_buffer[-1]
        hrr = hr_peak - hr_1min

        base_vo2 = 70 + (hrr * 0.5)

        #ajustari
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
        vo2_final = round(vo2_final, 2)

        self.last_vo2_max = vo2_final
        print(f"VO2Max calculat cu HRR: {vo2_final:.2f} ml/kg/min")
        return vo2_final


    def calculate_movement_intensity(self, landmarks, prev_landmarks, dt, frame=None, prev_frame=None):
        #calc intensitatea miscarii combinand: keypoints, motion energy & tracking
        if (prev_landmarks is None or dt == 0) and (frame is None or prev_frame is None):
            return 0  

        #diferenta dintre keypoints
        total_movement = 0
        joint_indices = [11, 12, 13, 14, 23, 24, 25, 26]
        for idx in joint_indices:
            prev = np.array([prev_landmarks[idx].x, prev_landmarks[idx].y, prev_landmarks[idx].z])
            curr = np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
            speed = np.linalg.norm(curr - prev) / dt if dt != 0 else 0
            total_movement += speed
        biomech_intensity = total_movement / len(joint_indices)

        #pt motion energy
        motion_energy = 0
        if frame is not None and prev_frame is not None:
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_energy = np.sum(thresh) / 255 / (frame.shape[0] * frame.shape[1])  # normalizat [0,1]

        #miscarea centrului corpului => bounding box, ca tracking
        def get_center(landmarks):
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            return np.array([np.mean(xs), np.mean(ys)])

        center_movement = 0
        if prev_landmarks is not None:
            prev_center = get_center(prev_landmarks)
            curr_center = get_center(landmarks)
            center_movement = np.linalg.norm(curr_center - prev_center) / dt

        #comb poderata
        combined_intensity = (
            0.5 * biomech_intensity + 
            0.3 * motion_energy + 
            0.2 * center_movement
        )

        return round(combined_intensity, 4)
 
    
    def estimate_initial_energy(self):
        vo2 = self.last_vo2_max if self.last_vo2_max is not None else 50.0
        vo2_score = (vo2 / 60) * 40

        hydration_score = (self.hydration_level / 100) * 15

        bpm_penalty = max(0, (self.heart_rate - 75) * 0.25)

        movement_penalty = min(self.movement_intensity, 1.0) * 5

        initial_energy = 45 + vo2_score + hydration_score - bpm_penalty - movement_penalty
        self.energy_level = round(max(50, min(100, initial_energy)))

        print(f"Energie initiala: {self.energy_level}%")
        return self.energy_level
    
    def estimate_facial_fatigue(self, landmarks):
        landmarks = list(landmarks.landmark)
        def euclidean(p1, p2):
            return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
        #EAR
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        left_ear = (
            euclidean(landmarks[left_eye[1]], landmarks[left_eye[5]]) +
            euclidean(landmarks[left_eye[2]], landmarks[left_eye[4]])
        ) / (2.0 * euclidean(landmarks[left_eye[0]], landmarks[left_eye[3]]))

        right_ear = (
            euclidean(landmarks[right_eye[1]], landmarks[right_eye[5]]) +
            euclidean(landmarks[right_eye[2]], landmarks[right_eye[4]])
        ) / (2.0 * euclidean(landmarks[right_eye[0]], landmarks[right_eye[3]]))

        ear = (left_ear + right_ear) / 2.0

        #MAR
        mouth = [61, 291, 81, 178, 13, 14]
        mar = (
            euclidean(landmarks[mouth[2]], landmarks[mouth[3]]) +
            euclidean(landmarks[mouth[4]], landmarks[mouth[5]])
        ) / (2.0 * euclidean(landmarks[mouth[0]], landmarks[mouth[1]]))

        #praguri empirice
        fatigue_score = 0
        if ear < 0.22:
            fatigue_score += 1.0
        if mar > 0.6:
            fatigue_score += 1.5

        return fatigue_score  


    def calculate_energy_level(self, face_landmarks=None):
        #scadere cu 1% pe parcurs pe baza miscarii
        #scaderea cum trebuie 
        current_time = time.time()
        if not hasattr(self, 'last_energy_update'):
            self.last_energy_update = current_time  

        #scadere doar daca au trecut cel putin 10 secunde
        if current_time - self.last_energy_update < 10:
            return self.energy_level
        #fata pt oboseala
        fatigue_factor = (self.total_calories / (self.user_weight * 10)) * 3  
        movement_penalty = min(3, self.movement_intensity * 1.2)  
        hydration_bonus = (self.hydration_level / 100) * 2  
        spo2_bonus = ((self.spo2 - 90) / 10) * 2  
        vo2_bonus = ((self.last_vo2_max - 30) / 30) * 2 if self.last_vo2_max is not None else 0

        effort_score = 0

        #intensitatea miscarii
        if self.movement_intensity > 0.3:
            effort_score += 0.5

        #puls crescut
        if self.heart_rate > 100:
            effort_score += 0.3

        #hidratatre slaba
        if self.hydration_level < 60:
            effort_score += 0.3

        #acuratete scazuta
        if self.execution_accuracy is not None and self.execution_accuracy < 70:
            effort_score += 0.3

        #expresii faciale pt oboseala
        facial_fatigue = self.estimate_facial_fatigue(face_landmarks) if face_landmarks else 0
        facial_penalty = min(facial_fatigue * 0.4, 0.5)

        total_penalty = min(effort_score + facial_penalty, 1.5)
        self.energy_level -= total_penalty

        self.last_energy_update = current_time
        #avand 0-100%
        self.energy_level = max(0, min(100, self.energy_level))

        return self.energy_level

    def estimate_initial_hydration(self):
        temp = self.get_face_temperature()
        bpm = self.heart_rate
        hydration = 100 - abs(36.8 - temp)*20 - abs(80 - bpm)*0.3
        return round(max(30, min(100, hydration)), 1)  


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
                    print(f" BPM actualizat: {self.heart_rate:.2f}")

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

            print(f"Calorii arse: {self.total_calories:.2f} kcal")

        
        vo2_max = self.calculate_vo2_max()

        #energia init
        if self.energy_level is None:
            if not hasattr(self, "energy_init_time"):
                self.energy_init_time = time.time()
            elif time.time() - self.energy_init_time > 3:
                self.energy_level = self.estimate_initial_energy()

        #actualizare energie continua doar daca VO2 valid
        if vo2_max is not None and self.energy_level is not None:
            energy_level = energy_level = self.calculate_energy_level(face_landmarks=results_face.multi_face_landmarks[0] if results_face and results_face.multi_face_landmarks else None)
        else:
            energy_level = "N/A"

        return {
            "heart_rate": round(self.heart_rate, 2),
            "calories": round(self.total_calories, 1),
            "spo2": round(self.spo2, 2) if hasattr(self, 'spo2') else 98,
            "energy_level": energy_level,
            "execution_accuracy": round(self.execution_accuracy, 2),
            #"hydration_level": self.last_hydration_level,
            "hydration_level": round(self.hydration_level, 1),
            "vo2_max": vo2_max
        }


    def estimate_met(self, movement_intensity, bpm):
        #alege MET in functie de intensitate & bpm
        #repaus
        if bpm < 80 and movement_intensity < 0.3:
            return 1.2 
            #usoara 
        elif bpm < 100 and movement_intensity < 0.7:
            return 3.8
            #mediu  
        elif bpm < 130 and movement_intensity < 1.5:
            return 5.6
            #intens
        elif bpm < 160 and movement_intensity < 2.5:
            return 8.2
        else:
            #maxim
            return 10.0  


    # def stop_camera(self):
    #     self.camera_running = False

    def stop_camera(self):
        self.camera_running = False
        if self.camera_cap and self.camera_cap.isOpened():
            self.camera_cap.release()

    def update_camera_frame(self, frame):
        #update frame
        self.camera_frame = frame  

    def calculate_accuracy(self, tutorial_landmarks, user_landmarks):
        if not tutorial_landmarks or not user_landmarks:
            return None 

        joint_pairs = [
            (11, 13, 15),  #cot stang
            (12, 14, 16),  #cot drept
            (13, 11, 23),  #umar stang
            (14, 12, 24),  #umar drept
            (23, 25, 27),  #genunchi stang
            (24, 26, 28),  #genunchi drept
            (11, 23, 25),  #sold stang
            (12, 24, 26),  #sold drept
            (23, 11, 13),  #trunchi fata de brat stang
            (24, 12, 14),  #trunchi fata de brat drept
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

    
    







