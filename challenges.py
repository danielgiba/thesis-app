from language_manager import t, get_current_language
import customtkinter as ctk
import pymongo
import json
import random
import datetime
from bson import ObjectId
import os


def get_theme_colors():
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {
            "bg": "#000000",
            "card": "#1e1e1e",
            "text": "white",
            "subtext": "lightgray",
            "accent": "#3B8ED0",
            "progress_bg": "#444444",
            "progress_color": "#3B8ED0",
            "status_text": "lightgreen"
        }
    else:
        return {
            "bg": "#ffffff",
            "card": "#f7f7f7",
            "text": "black",
            "subtext": "#444444",
            "accent": "#0078D7",
            "progress_bg": "#cccccc",
            "progress_color": "#0078D7",
            "status_text": "green"
        }


class ChallengesPage(ctk.CTkFrame):
    def __init__(self, parent, user_id):
        super().__init__(parent)
        self.user_id = ObjectId(user_id)
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["testapp"]
        self.users_collection = self.db["users"]

        self.colors = get_theme_colors()
        self.lang = get_current_language()

        self.configure(fg_color=self.colors["bg"])
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        title_label = ctk.CTkLabel(self, text=t("key_80"), font=("Arial", 22, "bold"), text_color=self.colors["text"])
        title_label.grid(row=0, column=0, pady=10)

        self.challenges_frame = ctk.CTkFrame(self, fg_color=self.colors["bg"])
        self.challenges_frame.grid(row=1, column=0, pady=10, padx=20)

        self.status_label = ctk.CTkLabel(self, text="", font=("Arial", 14, "bold"), text_color=self.colors["status_text"])
        self.status_label.grid(row=2, column=0, pady=10)

        self.load_daily_challenges()
        self.display_challenges()
        self.auto_check_challenges()

    def load_daily_challenges(self):
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        user_data = self.users_collection.find_one({"_id": self.user_id})

        if not user_data or "daily_challenges" not in user_data or user_data.get("challenges_date") != today:
            path = os.path.join(os.path.dirname(__file__), "daily_challenges.json")
            with open(path, "r", encoding="utf-8") as f:
                challenges = json.load(f)["challenges"]
            daily_challenges = random.sample(challenges, 5)
            self.users_collection.update_one(
                {"_id": self.user_id},
                {"$set": {"daily_challenges": daily_challenges, "challenges_date": today}},
                upsert=True
            )
        else:
            daily_challenges = user_data["daily_challenges"]

        self.daily_challenges = daily_challenges

    def display_challenges(self):
        for widget in self.challenges_frame.winfo_children():
            widget.destroy()

        self.progress_bars = {}

        for challenge in self.daily_challenges:
            challenge_id = challenge["id"]
            challenge_text = challenge["text"].get(self.lang, challenge["text"].get("en", "Unknown"))

            card = ctk.CTkFrame(self.challenges_frame, fg_color=self.colors["card"], corner_radius=12)
            card.pack(pady=10, padx=20, fill="x")

            title = ctk.CTkLabel(
                card,
                text=challenge_text,
                font=("Arial", 16, "bold"),
                text_color=self.colors["text"]
            )
            title.pack(anchor="w", padx=15, pady=(10, 5))

            progress_bar = ctk.CTkProgressBar(
                card,
                fg_color=self.colors["progress_bg"],
                progress_color=self.colors["progress_color"],
                height=16
            )
            progress_bar.set(0)
            progress_bar.pack(padx=15, pady=(0, 10), fill="x")

            self.progress_bars[challenge_id] = progress_bar

    def auto_check_challenges(self):
        user_data = self.users_collection.find_one({"_id": self.user_id})
        progress = user_data.get("workout_progress", [])
        today = datetime.datetime.today().strftime('%Y-%m-%d')

        today_entries = [x for x in progress if x["date"].startswith(today)]
        all_completed = True

        for challenge in self.daily_challenges:
            challenge_id = challenge["id"]
            challenge_type = challenge["type"]
            challenge_value = challenge["value"]
            completion = 0

            if challenge_type == "calories":
                total_calories = sum([x["calories"] for x in today_entries])
                completion = min(1, total_calories / challenge_value)

            elif challenge_type == "workouts":
                num_workouts = len(today_entries)
                completion = min(1, num_workouts / challenge_value)

            elif challenge_type == "hydration":
                avg_hydration = sum([x["hydration_level"] for x in today_entries]) / max(len(today_entries), 1)
                completion = min(1, avg_hydration / challenge_value)

            elif challenge_type == "streak":
                streak = self.calculate_streak(progress)
                completion = min(1, streak / challenge_value)

            elif challenge_type == "heart_rate":
                avg_hr = sum([x["heart_rate"] for x in today_entries]) / max(len(today_entries), 1)
                completion = 1 if avg_hr <= challenge_value else 0

            elif challenge_type == "plan_name":
                completion = 1 if any(challenge_value in x["plan_name"] for x in today_entries) else 0

            elif challenge_type == "rest":
                completion = 1 if len(today_entries) >= challenge_value else 0

            if challenge_id in self.progress_bars:
                self.progress_bars[challenge_id].set(completion)

            if completion < 1:
                all_completed = False

        self.status_label.configure(
            text=t("key_81") if all_completed else t("key_82")
        )

        self.after(5000, self.auto_check_challenges)

    def calculate_streak(self, progress):
        dates = sorted(set(x["date"].split()[0] for x in progress))
        streak = 0
        today = datetime.datetime.today().date()

        for i in range(len(dates) - 1, -1, -1):
            day = datetime.datetime.strptime(dates[i], "%Y-%m-%d").date()
            expected_day = today - datetime.timedelta(days=streak)
            if day == expected_day:
                streak += 1
            else:
                break
        return streak
