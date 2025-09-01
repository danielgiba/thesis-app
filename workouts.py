from language_manager import t
import customtkinter as ctk
from customtkinter import CTkImage
from PIL import Image
from workouts_page.classic import ClassicWorkout
from workouts_page.body_abs import BodyAbsWorkout
from workouts_page.legs import LegsWorkout
from workouts_page.arms import ArmsWorkout
from workouts_page.stretch import StretchWorkout

def get_theme_colors():
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {
            "bg": "#000000",
            "card": "#1a1a1a",
            "text": "white",
            "subtext": "gray",
            "accent": "#3B8ED0"
        }
    else:
        return {
            "bg": "#ffffff",
            "card": "#f2f2f2",
            "text": "black",
            "subtext": "#444444",
            "accent": "#0078D7"
        }

class WorkoutPage(ctk.CTkFrame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.colors = get_theme_colors()
        self.configure(fg_color=self.colors["bg"])

        title = ctk.CTkLabel(
            self, text=t("key_56"),
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=self.colors["text"]
        )
        title.pack(pady=20)

        self.scrollable_frame = ctk.CTkScrollableFrame(self, fg_color=self.colors["bg"])
        self.scrollable_frame.pack(fill="both", expand=True, padx=20, pady=10)

        workouts = [
            {"name": t("key_57"), "icon": "icons/classic_cardio.png", "desc": t("key_58"), "func": self.show_classic},
            {"name": t("key_59"), "icon": "icons/abs.png", "desc": t("key_60"), "func": self.show_abs},
            {"name": t("key_61"), "icon": "icons/leg.png", "desc": t("key_62"), "func": self.show_legs},
            {"name": t("key_63"), "icon": "icons/arm.png", "desc": t("key_64"), "func": self.show_arms},
            {"name": t("key_65"), "icon": "icons/stretch.png", "desc": t("key_66"), "func": self.show_stretch}
        ]

        for workout in workouts:
            self.create_workout_card(workout)

    def create_workout_card(self, workout):
        card = ctk.CTkFrame(self.scrollable_frame, fg_color=self.colors["card"], corner_radius=10)
        card.pack(pady=10, fill="x", padx=10)

        image = Image.open(workout["icon"]).convert("RGBA").resize((60, 60))
        tint_color = (255, 255, 255) if self.colors["text"] == "white" else (0, 0, 0)
        tinted_data = [(tint_color[0], tint_color[1], tint_color[2], a if a > 0 else 0) for r, g, b, a in image.getdata()]
        image.putdata(tinted_data)
        icon = CTkImage(light_image=image, size=(60, 60))

        image_label = ctk.CTkLabel(card, image=icon, text="")
        image_label.image = icon
        image_label.pack(side="left", padx=10, pady=10)

        text_frame = ctk.CTkFrame(card, fg_color=self.colors["card"])
        text_frame.pack(side="left", padx=5, fill="both", expand=True)

        workout_label = ctk.CTkLabel(
            text_frame,
            text=f"{workout['name']}",
            font=("Arial", 16),
            text_color=self.colors["text"]
        )
        workout_label.pack(anchor="w", pady=2)

        desc_label = ctk.CTkLabel(
            text_frame,
            text=workout["desc"],
            font=("Arial", 12),
            text_color=self.colors["subtext"]
        )
        desc_label.pack(anchor="w")

        start_button = ctk.CTkButton(
            card,
            text=t("key_67"),
            fg_color=self.colors["accent"],
            text_color="white",
            width=100,
            command=workout["func"]
        )
        start_button.pack(side="right", padx=10, pady=10)

    def show_classic(self):
        self.main_app.clear_content()
        self.main_app.hide_navigation_bar()
        ClassicWorkout(self.main_app.content_frame, self.main_app).pack(fill="both", expand=True)

    def show_abs(self):
        self.main_app.clear_content()
        self.main_app.hide_navigation_bar()
        BodyAbsWorkout(self.main_app.content_frame, self.main_app).pack(fill="both", expand=True)

    def show_legs(self):
        self.main_app.clear_content()
        self.main_app.hide_navigation_bar()
        LegsWorkout(self.main_app.content_frame, self.main_app).pack(fill="both", expand=True)

    def show_arms(self):
        self.main_app.clear_content()
        self.main_app.hide_navigation_bar()
        ArmsWorkout(self.main_app.content_frame, self.main_app).pack(fill="both", expand=True)

    def show_stretch(self):
        self.main_app.clear_content()
        self.main_app.hide_navigation_bar()
        StretchWorkout(self.main_app.content_frame, self.main_app).pack(fill="both", expand=True)
