from language_manager import t
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import StringVar
from itertools import cycle
from bson import ObjectId
from datetime import timedelta
import customtkinter as ctk
import pymongo
import pandas as pd
import numpy as np
import seaborn as sns
import os

def get_theme_colors():
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {
            "bg": "#000000",
            "text": "white",
            "accent": "#3B8ED0",
            "plot_bg": "#121212",
            "line": "white",
            "future": "yellow",
            "heatmap": "coolwarm",
            "msg_frame": "#1a1a1a"
        }
    else:
        return {
            "bg": "#ffffff",
            "text": "black",
            "accent": "#0078D7",
            "plot_bg": "#f0f0f0",
            "line": "black",
            "future": "orange",
            "heatmap": "coolwarm",
            "msg_frame": "#e6e6e6"
        }

class StatisticsPage(ctk.CTkFrame):
    def __init__(self, parent, user_id, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.colors = get_theme_colors()
        self.configure(fg_color=self.colors["bg"])

        self.main_app.bottom_nav_frame.pack(side="bottom", fill="x")
        self.main_app.bottom_nav_frame.lift()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        self.user_id = ObjectId(user_id)
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["testapp"]
        self.users_collection = self.db["users"]

        self.all_workouts = [
            "Classic Workout", "Abs Workout", "Legs Workout", "Arms Workout", "Stretch Workout",
            "Easy Plan 1", "Easy Plan 2", "Medium Plan 1", "Medium Plan 2", "Hard Plan 1", "Hard Plan 2"
        ]

        user_data = self.users_collection.find_one({"_id": self.user_id})
        self.workout_dict = {workout: [] for workout in self.all_workouts}
        if user_data and "workout_progress" in user_data:
            for entry in user_data["workout_progress"]:
                if entry["plan_name"] in self.workout_dict:
                    self.workout_dict[entry["plan_name"]].append(entry)

        self.workouts_list = list(self.workout_dict.keys())
        self.workout_cycle = cycle(self.workouts_list)
        self.selected_workout = StringVar()
        self.selected_workout.set(self.workouts_list[0])

        self.top_bar = ctk.CTkFrame(self, fg_color=self.colors["bg"])
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        ctk.CTkButton(self.top_bar, text=t("key_69"), command=self.show_previous_workout,
                      fg_color=self.colors["bg"], text_color=self.colors["text"]).pack(side="left", padx=5)

        self.workout_menu = ctk.CTkOptionMenu(self.top_bar, variable=self.selected_workout,
                                              values=self.workouts_list, fg_color=self.colors["bg"],
                                              text_color=self.colors["text"], command=self.plot_graphs)
        self.workout_menu.pack(side="left", expand=True, padx=10)

        ctk.CTkButton(self.top_bar, text=t("key_70"), command=self.show_next_workout,
                      fg_color=self.colors["bg"], text_color=self.colors["text"]).pack(side="right", padx=5)

        self.graph_frame = ctk.CTkScrollableFrame(self, fg_color=self.colors["bg"])
        self.graph_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.recommendation_label = ctk.CTkLabel(self, text="", fg_color=self.colors["bg"], text_color=self.colors["text"])
        self.recommendation_label.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.plot_graphs(self.selected_workout.get())

    def show_previous_workout(self):
        self.selected_workout.set(next(self.workout_cycle))
        self.plot_graphs(self.selected_workout.get())

    def show_next_workout(self):
        self.selected_workout.set(next(self.workout_cycle))
        self.plot_graphs(self.selected_workout.get())

    def load_user_data(self, workout_name):
        if workout_name in self.workout_dict and self.workout_dict[workout_name]:
            df = pd.DataFrame(self.workout_dict[workout_name])
            df["date"] = pd.to_datetime(df["date"]).dt.date
            numeric_cols = ["heart_rate", "calories", "vo2_max", "hydration_level", "energy_level"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["date"] + numeric_cols].groupby("date", as_index=False).mean()
            return df.dropna()
        return None

    def predict_future_vo2(self, df):
        if df.empty:
            return pd.DataFrame()
        last_date = pd.to_datetime(df["date"]).max()
        dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        x = np.arange(len(df))
        y = df["vo2_max"].values
        coef = np.polyfit(x, y, 1)
        future_x = np.arange(len(df), len(df) + 7)
        predicted = np.polyval(coef, future_x)
        return pd.DataFrame({"date": dates, "vo2_max": predicted})

    def classify_progress(self, vo2_series):
        clean_series = vo2_series.dropna()
        if len(clean_series) < 8:
            return "NECUNOSCUT"
        delta = clean_series.iloc[-1] - clean_series.iloc[-8]
        if delta > 1:
            return "PROGRES"
        elif delta < -1:
            return "REGRES"
        return "STAGNARE"

    def compute_fitness_score(self, df):
        return (
            df["vo2_max"] + (df["calories"] / 10) + df["energy_level"]
        ) / (df["heart_rate"] / 2)

    def generate_advice(self, corr):
        msgs = []
        if corr.loc["vo2_max", "energy_level"] > 0.5:
            msgs.append(t("key_172"))
        if corr.loc["hydration_level", "energy_level"] < -0.3:
            msgs.append(t("key_173"))
        if corr.loc["heart_rate", "calories"] > 0.5:
            msgs.append(t("key_174"))
        if corr.loc["hydration_level", "vo2_max"] > 0.4:
            msgs.append(t("key_175"))
        return msgs


    def plot_graphs(self, workout_name):
        df = self.load_user_data(workout_name)
        if df is None or df.empty:
            self.recommendation_label.configure(text=f"{t('key_72')} {workout_name}.")
            return

        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fitness_score = round(self.compute_fitness_score(df).mean(), 2)
        progress_status = self.classify_progress(df["vo2_max"])
        future_df = self.predict_future_vo2(df)

        ctk.CTkLabel(self.graph_frame, text=f"{t('key_71')}: {fitness_score}", font=("Arial", 20, "bold"),
                     text_color=self.colors["text"]).pack(pady=5)
        progress_status_key = {
            "PROGRES": "key_177",
            "REGRES": "key_178",
            "STAGNARE": "key_179",
            "NECUNOSCUT": "key_180"
        }.get(progress_status, "key_180")

        ctk.CTkLabel(
            self.graph_frame,
            text=f"{t('key_181')} {t(progress_status_key)}",
            font=("Arial", 16),
            text_color=self.colors["text"]
        ).pack(pady=5)


        metrics = ["heart_rate", "calories", "vo2_max", "hydration_level", "energy_level"]
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 14))
        fig.patch.set_facecolor(self.colors["plot_bg"])

        for i, col in enumerate(metrics):
            axes[i].plot(df["date"], df[col], label=col.capitalize(), marker="o", color=self.colors["line"])
            if col == "vo2_max" and not future_df.empty:
                axes[i].plot(future_df["date"], future_df[col], linestyle="dotted", color=self.colors["future"], label=t("key_176"))
            axes[i].set_facecolor(self.colors["plot_bg"])
            axes[i].set_title(col.capitalize(), color=self.colors["text"])
            axes[i].tick_params(colors=self.colors["text"])
            axes[i].legend()

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig) 

        corr = df[metrics].corr()
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        fig2.patch.set_facecolor(self.colors["plot_bg"])
        sns.heatmap(corr, annot=True, cmap=self.colors["heatmap"], fmt=".2f", ax=ax2,
                    cbar=True, cbar_kws={'shrink': 0.8})
        ax2.set_facecolor(self.colors["plot_bg"])
        ax2.set_title(t("key_79"), color=self.colors["text"])
        ax2.tick_params(colors=self.colors["text"])
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        fig2.tight_layout()

        canvas2 = FigureCanvasTkAgg(fig2, master=self.graph_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig2)  

        messages = self.generate_advice(corr)
        if messages:
            msg_frame = ctk.CTkFrame(self.graph_frame, fg_color=self.colors["msg_frame"])
            msg_frame.pack(pady=10, padx=10, fill="x")
            ctk.CTkLabel(msg_frame, text=t("key_73"), font=("Arial", 16, "bold"),
                         text_color=self.colors["text"]).pack(anchor="w", padx=10, pady=(0, 5))
            for msg in messages:
                ctk.CTkLabel(msg_frame, text=msg, text_color=self.colors["text"], anchor="w").pack(anchor="w", padx=10, pady=2)
