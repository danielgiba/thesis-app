import customtkinter as ctk
import os
import json
from workout_window import WorkoutWindow

def get_theme_colors():
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {
            "bg": "#000000",
            "text": "white",
            "card": "#1a1a1a",
            "card_text": "white",
            "header": "lightgray",
            "start": "green",
            "back": "red"
        }
    else:
        return {
            "bg": "#ffffff",
            "text": "black",
            "card": "#f2f2f2",
            "card_text": "black",
            "header": "#444444",
            "start": "#28a745",
            "back": "#dc3545"
        }

class DayReview(ctk.CTkFrame):
    def __init__(self, parent, main_app, plan_name, day, save_progress):
        super().__init__(parent)
        self.main_app = main_app
        self.plan_name = plan_name
        self.day = day
        self.save_progress = save_progress

        self.colors = get_theme_colors()
        self.configure(fg_color=self.colors["bg"])

        title_label = ctk.CTkLabel(
            self,
            text=f"{plan_name} - Day {day}",
            font=ctk.CTkFont(size=24),
            text_color=self.colors["text"]
        )
        title_label.pack(pady=10)

        exercises_dict = {
            "easy_plan_1": {
                1: [("Jumping Jacks", "15 s"), ("Step-up onto Chair", "10"), ("Push-ups", "3"), ("Abdominal Crunches", "10"), ("Plank", "10 s")],
                2: [("Squats", "12"), ("Lunges", "10"), ("Plank", "15 s")]
            },
            "easy_plan_2": {
                1: [("High Knees", "20 s"), ("Wall Sit", "15 s"), ("Push-ups", "5"), ("Plank", "20 s")],
                2: [("Burpees", "10"), ("Jump Squats", "12"), ("Plank", "15 s")]
            },
            "medium_plan_1": {
                1: [("Burpees", "10"), ("Mountain Climbers", "15 s"), ("Push-ups", "5"), ("Plank", "20 s")],
                2: [("Jump Rope", "30 s"), ("Lunges", "12"), ("Sit-ups", "10")]
            },
            "medium_plan_2": {
                1: [("Pull-ups", "5"), ("Squat Jumps", "12"), ("Plank Shoulder Taps", "15 s")],
                2: [("Box Jumps", "8"), ("Lunges", "12"), ("Plank", "20 s")]
            },
            "hard_plan_1": {
                1: [("Pull-ups", "5"), ("Dips", "10"), ("Push-ups", "15"), ("Plank", "30 s")],
                2: [("Burpees", "15"), ("Jump Rope", "45 s"), ("Push-ups", "20")]
            },
            "hard_plan_2": {
                1: [("Deadlifts", "8"), ("Squats", "15"), ("Push-ups", "20"), ("Plank", "40 s")],
                2: [("Snatch", "5"), ("Jump Lunges", "12"), ("Plank", "25 s")]
            }
        }

        exercise_list = exercises_dict.get(plan_name.lower().replace(" ", "_"), {}).get(day, [("Rest Day", "")])

        frame = ctk.CTkFrame(self, fg_color=self.colors["card"], corner_radius=10)
        frame.pack(pady=10, padx=20, fill="x")

        title_day = ctk.CTkLabel(
            frame,
            text=f"Day {day}",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors["card_text"]
        )
        title_day.pack(pady=5)

        header = ctk.CTkLabel(
            frame,
            text="Exercise               Reps",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors["header"]
        )
        header.pack(pady=5)

        for exercise, reps in exercise_list:
            ctk.CTkLabel(frame, text=f"{exercise:<20} {reps}", text_color=self.colors["card_text"]).pack(pady=2)

        back_button = ctk.CTkButton(
            self,
            text="Back",
            command=self.go_back,
            fg_color=self.colors["back"],
            text_color="white"
        )
        back_button.pack(pady=10)

        start_button = ctk.CTkButton(
            self,
            text="Start Workout",
            command=self.start_workout,
            fg_color=self.colors["start"],
            text_color="white"
        )
        start_button.pack(pady=10)

    def go_back(self):
        self.main_app.show_plan_30_days(self.plan_name)

    def start_workout(self):
        user_data = self.main_app.collection.find_one({"username": self.main_app.logged_user})

        if user_data:
            user_id = str(user_data["_id"])
        else:
            print("❌ Eroare: Utilizatorul logat nu există în MongoDB!")
            return

        video_path = f"home_plan_workouts/tutorials/{self.plan_name.lower().replace(' ', '').replace('_', '')}_day{self.day}.mp4"

        if not os.path.exists(video_path):
            print(f"❌ Eroare: Fișierul video {video_path} nu există!")
            return

        self.main_app.clear_content()
        workout_window = WorkoutWindow(
            self.main_app.content_frame,
            user_id,
            video_path,
            self.plan_name,
            self.day,
            self.go_back,
            self.main_app
        )
        workout_window.pack(fill="both", expand=True)
        workout_window.start_video()

    def complete_workout(self):
        progress_folder = "home_plan_workouts/progress"
        os.makedirs(progress_folder, exist_ok=True)

        progress_file = os.path.join(progress_folder, f"progress_{self.plan_name}.json")

        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                progress_data = json.load(f)
        else:
            progress_data = {}

        progress_data[str(self.day)] = "completed"

        with open(progress_file, "w") as f:
            json.dump(progress_data, f, indent=4)

        print(f"✅ Progres salvat pentru {self.plan_name} - Ziua {self.day}")
        self.main_app.show_home()
