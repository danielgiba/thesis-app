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
import joblib
import os

# 🔁 Tema dinamică
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
        workout_name = workout_name.strip()
        if workout_name in self.workout_dict and self.workout_dict[workout_name]:
            df = pd.DataFrame(self.workout_dict[workout_name])
            
            # 🔥 Doar dacă datele au sens (ignoră day total)
            df["date"] = pd.to_datetime(df["date"]).dt.date

            # 🔧 Normalizează tipurile
            numeric_columns = ["heart_rate", "calories", "vo2_max", "hydration_level", "energy_level"]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # 📅 Grupăm doar după dată
            df = df[["date"] + numeric_columns]
            df = df.groupby("date", as_index=False).mean()
            
            # ✅ Dacă totuși rezultă un df gol (NaN-uri sau lipsă valori numerice)
            if df[numeric_columns].dropna().empty:
                return None
            
            return df
        return None



    def predict_future_vo2(self, df):
        model_path = "model_vo2_final.pkl"
        if not os.path.exists(model_path):
            return pd.DataFrame()
        model = joblib.load(model_path)
        future_dates = [df["date"].max() + timedelta(days=i) for i in range(1, 8)]
        ordinals = [pd.Timestamp(d).toordinal() for d in future_dates]
        predicted_vo2 = model.predict(np.array(ordinals).reshape(-1, 1))
        future_df = pd.DataFrame({"date": future_dates, "vo2_max": predicted_vo2})
        for col in ["heart_rate", "calories", "hydration_level", "energy_level"]:
            future_df[col] = df[col].mean()
        return future_df

    def get_progress_label(self, df):
        model_path = "model_progress_final.pkl"
        encoder_path = "label_encoder_final.pkl"
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            return "necunoscut"
        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
        last_week = df[["vo2_max", "energy_level", "hydration_level"]].tail(7).mean().to_frame().T
        try:
            label = model.predict(last_week)[0]
            return le.inverse_transform([label])[0]
        except:
            return "necunoscut"

    def plot_graphs(self, workout_name):
        df = self.load_user_data(workout_name)
        if df is None or df.empty:
            self.recommendation_label.configure(text=f"{t('key_72')} {workout_name}.")
            return

        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        plt.style.use("default")
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 14))
        fig.patch.set_facecolor(self.colors["plot_bg"])

        fitness_score = (
            df["vo2_max"].mean() + (df["calories"].sum() / 10) + df["energy_level"].mean()
        ) / (df["heart_rate"].mean() / 2)

        progress_status = self.get_progress_label(df)

        ctk.CTkLabel(self.graph_frame, text=f"{t('key_71')}: {round(fitness_score, 2)}", font=("Arial", 20, "bold"),
                     text_color=self.colors["text"]).pack(pady=5)

        ctk.CTkLabel(self.graph_frame, text=f"Stare progres: {progress_status.upper()}",
                     font=("Arial", 16), text_color=self.colors["text"]).pack(pady=5)

        future_df = self.predict_future_vo2(df)
        metrics = ["heart_rate", "calories", "vo2_max", "hydration_level", "energy_level"]

        for i, col in enumerate(metrics):
            axes[i].plot(df["date"], df[col], label=col.capitalize(), marker='o', color=self.colors["line"])
            axes[i].set_facecolor(self.colors["plot_bg"])  # <-- setează fundalul corect
            if not future_df.empty:
                axes[i].plot(future_df["date"], future_df[col], linestyle="dashed", color=self.colors["future"])
            axes[i].set_title(col.capitalize(), color=self.colors["text"])
            axes[i].tick_params(colors=self.colors["text"])
            axes[i].legend()

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        corr_fig, corr_ax = plt.subplots(figsize=(5, 4))
        corr = df[metrics].corr()
        import seaborn as sns
        sns.heatmap(corr, annot=True, cmap=self.colors["heatmap"], fmt=".2f", ax=corr_ax)
        corr_ax.set_facecolor(self.colors["plot_bg"])
        corr_ax.set_title(t("key_79"), color=self.colors["text"])
        corr_ax.tick_params(colors=self.colors["text"])
        corr_fig.patch.set_facecolor(self.colors["plot_bg"])

        corr_canvas = FigureCanvasTkAgg(corr_fig, master=self.graph_frame)
        corr_canvas.draw()
        corr_canvas.get_tk_widget().pack(fill="both", expand=True)

        messages = []
        if corr.loc["vo2_max", "energy_level"] > 0.5:
            messages.append(t("key_74"))
        if corr.loc["hydration_level", "energy_level"] < -0.3:
            messages.append(t("key_75"))
        if corr.loc["heart_rate", "calories"] > 0.5:
            messages.append(t("key_76"))
        if corr.loc["hydration_level", "vo2_max"] > 0.4:
            messages.append(t("key_77"))

        if messages:
            msg_frame = ctk.CTkFrame(self.graph_frame, fg_color=self.colors["msg_frame"])
            msg_frame.pack(pady=10, padx=10, fill="x")
            ctk.CTkLabel(msg_frame, text=t("key_73"), font=("Arial", 16, "bold"), text_color=self.colors["text"]).pack(anchor="w", padx=10, pady=(0, 5))
            for msg in messages:
                ctk.CTkLabel(msg_frame, text=msg, text_color=self.colors["text"], anchor="w").pack(anchor="w", padx=10, pady=2)

        if df["energy_level"].iloc[-1] < 40 or df["heart_rate"].iloc[-1] > 150:
            ctk.CTkLabel(self.graph_frame, text=t("key_78"), text_color="red").pack(pady=5)
