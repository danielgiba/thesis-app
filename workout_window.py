from language_manager import t
import customtkinter as ctk
from tkinter import Frame, Label
import vlc
import threading
from parameters import ParameterCalculator
import mediapipe as mp
import cv2
from utils import get_user_weight
import os
import json
import pymongo
from datetime import datetime
from bson import ObjectId
from types import SimpleNamespace


def get_theme_colors():
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {
            "bg": "#000000",
            "panel": "#1e1e1e",
            "text": "white",
            "btn_text": "white",
            "accent_green": "green",
            "accent_orange": "orange",
            "accent_red": "red",
            "accent_blue": "blue",
            "feedback_bg": "#1e1e1e",
            "feedback_fg": "white"
        }
    else:
        return {
            "bg": "#ffffff",
            "panel": "#f2f2f2",
            "text": "black",
            "btn_text": "black",
            "accent_green": "#007E33",
            "accent_orange": "#FF8800",
            "accent_red": "#CC0000",
            "accent_blue": "#007BFF",
            "feedback_bg": "#f2f2f2",
            "feedback_fg": "black"
        }

class WorkoutWindow(ctk.CTkFrame):
    def __init__(self, parent, user_id, video_path, plan_name, day, back_callback, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.user_id = ObjectId(user_id)
        self.plan_name = plan_name
        self.day = day
        self.video_path = video_path
        self.back_callback = back_callback

        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["testapp"]
        self.users_collection = self.db["users"]

        self.user_weight = get_user_weight()
        self.colors = get_theme_colors()

        self.configure(fg_color=self.colors["bg"])

        self.video_cap = None
        self.video_cv = cv2.VideoCapture(self.video_path)

        self.vlc_instance = vlc.Instance()
        self.media_player = self.vlc_instance.media_player_new()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.param_calculator = ParameterCalculator(weight=self.user_weight)

        self.bpm_values = []
        self.calories_values = []
        self.hydration_values = []
        self.accuracy_values = []
        self.energy_values = []
        self.last_accuracies = []
        self.last_tutorial_landmarks = []

        self.finished = False

        top_frame = ctk.CTkFrame(self, fg_color=self.colors["bg"], height=60)
        top_frame.pack(fill="x")

        ctk.CTkButton(top_frame, text=t("key_113"), command=self.on_back, fg_color="transparent", text_color="cyan").pack(side="left", padx=10)

        if "day" in self.plan_name.lower():
            title_text = f"{self.plan_name} - Day {self.day}"
        else:
            title_text = f"{self.plan_name.capitalize()}"

        title_label = ctk.CTkLabel(top_frame, text=f"🏋️ {title_text}", font=("Arial", 20, "bold"), text_color=self.colors["text"])
        title_label.pack(side="top", pady=5)

        middle_frame = ctk.CTkFrame(self, fg_color=self.colors["bg"])
        middle_frame.pack(fill="both", expand=True, padx=20, pady=10)

        video_container = Frame(middle_frame, bg=self.colors["bg"], width=1024, height=576)
        video_container.pack(side="left", padx=(10, 20), pady=10)

        self.media_player.set_hwnd(video_container.winfo_id())

        right_panel = ctk.CTkFrame(middle_frame, fg_color=self.colors["panel"], width=300, height=205, corner_radius=12)
        right_panel.pack(side="right", fill="y", padx=10)

        self.feedback_label = Label(right_panel, text=t("key_118"), font=("Arial", 15), bg=self.colors["feedback_bg"], fg=self.colors["feedback_fg"])
        self.feedback_label.pack(fill="x", pady=(15, 5))

        self.parameters_label = Label(right_panel, text="", font=("Arial", 14), bg=self.colors["feedback_bg"], fg=self.colors["feedback_fg"], justify="left")
        self.parameters_label.pack(fill="x", padx=10)

        bottom_frame = ctk.CTkFrame(self, fg_color=self.colors["bg"], height=50)
        bottom_frame.pack(fill="x", pady=10)

        ctk.CTkButton(bottom_frame, text="▶️ " + t("key_114"), command=self.start_video, fg_color=self.colors["accent_green"], text_color=self.colors["btn_text"]).pack(side="left", padx=10)
        ctk.CTkButton(bottom_frame, text="⏸ " + t("key_115"), command=self.pause_video, fg_color=self.colors["accent_orange"], text_color=self.colors["btn_text"]).pack(side="left", padx=10)
        ctk.CTkButton(bottom_frame, text="⏹ " + t("key_116"), command=self.stop_video, fg_color=self.colors["accent_red"], text_color=self.colors["btn_text"]).pack(side="left", padx=10)
        ctk.CTkButton(bottom_frame, text="📷 " + t("key_117"), command=self.start_camera_thread, fg_color=self.colors["accent_blue"], text_color=self.colors["btn_text"]).pack(side="left", padx=10)

        self.video_cv.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.start_camera_thread()

        self.feedback_label.config(text=t("key_118"))
        self.parameters_label.config(
            text=(f"💓 {t('key_119')}: -\n"
                  f"🔥 {t('key_120')}: -\n"
                  f"💧 {t('key_121')}: -\n"
                  f"✅ {t('key_122')}: -\n"
                  f"⚡ {t('key_123')}: -\n"
                  f"🫁 {t('key_124')}: -")
        )
        self.after(10000, self.update_parameters)

    def start_video(self):
        if self.video_cap is None:
            self.video_cap = cv2.VideoCapture(self.video_path)
        self.media = self.vlc_instance.media_new(self.video_path)
        self.media_player.set_media(self.media)
        self.media_player.play()
        print("Video started")  
        self.after(1000, self.check_video_status)  

    def pause_video(self):
        self.media_player.pause()

    def stop_video(self):
        self.media_player.stop()

    def start_camera_thread(self):
        threading.Thread(target=self.param_calculator.camera_loop, daemon=True).start()

    def get_camera_frame(self):
        if not hasattr(self.param_calculator, "camera_cap") or not self.param_calculator.camera_cap.isOpened():
            return None
        ret, frame = self.param_calculator.camera_cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

    def check_video_status(self):
        state = self.media_player.get_state()
        print("Current state:", state)  # DEBUG
        if state == vlc.State.Ended and not self.finished:
            print("Video finished — triggering complete_workout") 
            self.finished = True
            self.complete_workout()
        else:
            self.after(1000, self.check_video_status)

    def on_back(self):
        self.param_calculator.stop_camera()
        self.back_callback()
        self.main_app.show_navigation_bar()

    def update_parameters(self):
        frame = self.get_camera_frame()
        results_pose = self.pose.process(frame) if frame is not None else None
        results_face = self.param_calculator.mp_face_mesh.process(frame) if frame is not None else None

        user_landmarks = results_pose.pose_landmarks.landmark if results_pose and results_pose.pose_landmarks else None
        if not user_landmarks:
            self.parameters_label.config(
                text=(f"💓 {t('key_119')}: N/A\n"
                      f"🔥 {t('key_120')}: N/A\n"
                      f"💧 {t('key_121')}: N/A\n"
                      f"✅ {t('key_122')}: N/A\n"
                      f"⚡ {t('key_123')}: N/A\n"
                      f"🫁 {t('key_124')}: N/A")
            )
            self.after(1000, self.update_parameters)
            return

        tutorial_landmarks = self.get_tutorial_landmarks()
        if tutorial_landmarks and isinstance(tutorial_landmarks[0], dict):
            tutorial_landmarks = [SimpleNamespace(**lm) for lm in tutorial_landmarks]

        if tutorial_landmarks and len(tutorial_landmarks) == 33:
            self.last_tutorial_landmarks.append(tutorial_landmarks)
            if len(self.last_tutorial_landmarks) > 5:
                self.last_tutorial_landmarks.pop(0)
        else:
            tutorial_landmarks = user_landmarks
            self.last_tutorial_landmarks.append(tutorial_landmarks)
            if len(self.last_tutorial_landmarks) > 5:
                self.last_tutorial_landmarks.pop(0)

        if user_landmarks and all(len(t) == 33 for t in self.last_tutorial_landmarks):
            accuracies = []
            for tutorial_frame in self.last_tutorial_landmarks:
                acc = self.param_calculator.calculate_accuracy(tutorial_frame, user_landmarks)
                if acc is not None:
                    accuracies.append(acc)

            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                self.last_accuracies.append(avg_acc)
                if len(self.last_accuracies) > 5:
                    self.last_accuracies.pop(0)
                smoothed_acc = sum(self.last_accuracies) / len(self.last_accuracies)
                self.param_calculator.execution_accuracy = round(smoothed_acc, 2)

        params = self.param_calculator.calculate_parameters(frame, results_pose, results_face)
        if params:
            self.bpm_values.append(params["heart_rate"])
            self.calories_values.append(params["calories"])
            self.hydration_values.append(params["hydration_level"])
            self.accuracy_values.append(params["execution_accuracy"])
            self.energy_values.append(params["energy_level"])

            workout_data = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plan_name": self.plan_name,
                "day": self.day,
                "calories": params["calories"],
                "heart_rate": params["heart_rate"],
                "hydration_level": params["hydration_level"],
                "execution_accuracy": params["execution_accuracy"],
                "energy_level": params["energy_level"],
                "vo2_max": params["vo2_max"]
            }

            self.users_collection.update_one(
                {"_id": self.user_id},
                {"$push": {"workout_progress": workout_data}}
            )

            try:
                self.parameters_label.config(
                    text=(f"💓 {t('key_119')}: {round(float(params['heart_rate']))} BPM\n"
                          f"🔥 {t('key_120')}: {round(float(params['calories']), 1)} kcal\n"
                          f"💧 {t('key_121')}: {round(float(params['hydration_level']))}%\n"
                          f"✅ {t('key_122')}: {round(float(params['execution_accuracy']))}%\n"
                          f"⚡ {t('key_123')}: {round(float(params['energy_level']))}%\n"
                          f"🫁 {t('key_124')}: {round(float(params['vo2_max']), 2)} ml/kg/min")
                )
            except Exception as e:
                print("Error displaying parameters:", e)

        self.after(1000, self.update_parameters)

    def complete_workout(self):
        def avg(values):
            numeric_values = [float(v) for v in values if isinstance(v, (int, float)) or str(v).replace('.', '', 1).isdigit()]
            return round(sum(numeric_values) / len(numeric_values), 2) if numeric_values else 0

        final = {
            "heart_rate": avg(self.bpm_values),
            "calories": avg(self.calories_values),
            "hydration_level": avg(self.hydration_values),
            "execution_accuracy": avg(self.accuracy_values),
            "energy_level": avg(self.energy_values),
            "vo2_max": self.param_calculator.vo2_max or 40
        }

        self.save_workout_progress(final)
        self.param_calculator.stop_camera()
        self.media_player.stop()
        self.back_callback()
        self.main_app.show_navigation_bar()

    def save_workout_progress(self, final_parameters):
        if not self.users_collection.find_one({"_id": self.user_id}):
            print("User not found.")
            return

        self.users_collection.update_one(
            {"_id": self.user_id},
            {"$push": {
                "workout_progress": {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "plan_name": self.plan_name,
                    "day": self.day,
                    **final_parameters
                }
            }}
        )

    def get_tutorial_landmarks(self):
        try:
            plan_clean = self.plan_name.lower().replace(" workout", "").replace(" ", "_")
            filename = plan_clean + "_workout.json"
            path = os.path.join("workouts_page", "tutorials", "landmarks", filename)
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Landmark JSON not found: {path}")
            return None
        except Exception as e:
            print(f"⚠️ Error loading landmarks: {e}")
            return None
