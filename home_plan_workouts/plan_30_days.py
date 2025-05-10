import customtkinter as ctk
import json
import os

def get_theme_colors():
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {
            "bg": "#000000",
            "text": "white",
            "card": "#1a1a1a",
            "button_text": "white",
            "complete": "#1cc88a",
            "in_progress": "#e74a3b",
            "not_started": "#6c757d",
            "back": "#cc0000"
        }
    else:
        return {
            "bg": "#ffffff",
            "text": "black",
            "card": "#f2f2f2",
            "button_text": "black",
            "complete": "#28a745",
            "in_progress": "#dc3545",
            "not_started": "#adb5bd",
            "back": "#d9534f"
        }

class Plan30Days(ctk.CTkFrame):
    def __init__(self, parent, main_app, plan_name):
        super().__init__(parent)
        self.main_app = main_app
        self.plan_name = plan_name
        self.colors = get_theme_colors()

        self.configure(fg_color=self.colors["bg"])
        self.progress_data = self.load_progress()

        title_label = ctk.CTkLabel(
            self,
            text=f"{plan_name} - 30 Days Challenge",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors["text"]
        )
        title_label.pack(pady=10)

        self.days_frame = ctk.CTkFrame(self, fg_color=self.colors["bg"])
        self.days_frame.pack(pady=10)

        for i in range(1, 31):
            day_status = self.progress_data.get(str(i), "not_started")

            if day_status == "completed":
                color = self.colors["complete"]
                text = f"✅ {i}"
            elif day_status == "in_progress":
                color = self.colors["in_progress"]
                text = f"🟥 {i}"
            else:
                color = self.colors["not_started"]
                text = f"{i}"

            day_button = ctk.CTkButton(
                self.days_frame,
                text=text,
                fg_color=color,
                text_color=self.colors["button_text"],
                command=lambda d=i: self.show_day_review(d),
                width=70,
                height=40,
                corner_radius=8
            )
            day_button.grid(row=(i-1)//6, column=(i-1) % 6, padx=5, pady=5)

        completed_days = len([d for d in self.progress_data.values() if d == "completed"])
        progress_percent = round((completed_days / 30) * 100, 1)
        remaining_days = 30 - completed_days

        progress_label = ctk.CTkLabel(
            self,
            text=f"📊 Progress: {progress_percent}% - {remaining_days} Days Left",
            font=ctk.CTkFont(size=16),
            text_color=self.colors["text"]
        )
        progress_label.pack(pady=10)

        back_button = ctk.CTkButton(
            self,
            text="Back",
            command=self.go_back,
            fg_color=self.colors["back"],
            text_color="white",
            corner_radius=6
        )
        back_button.pack(pady=10)

    def show_day_review(self, day):
        from home_plan_workouts.day_review import DayReview
        self.main_app.clear_content()
        DayReview(self.main_app.content_frame, self.main_app, self.plan_name, day, self.save_progress).pack(fill="both", expand=True)

    def go_back(self):
        self.main_app.show_home()

    def load_progress(self):
        progress_folder = "home_plan_workouts/progress"
        os.makedirs(progress_folder, exist_ok=True)

        progress_file = os.path.join(progress_folder, f"progress_{self.plan_name}.json")

        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                return json.load(f)

        return {}

    def save_progress(self, day, status):
        progress_folder = "home_plan_workouts/progress"
        os.makedirs(progress_folder, exist_ok=True)

        progress_file = os.path.join(progress_folder, f"progress_{self.plan_name}.json")

        self.progress_data[str(day)] = status

        with open(progress_file, "w") as f:
            json.dump(self.progress_data, f, indent=4)
