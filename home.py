from language_manager import t
import customtkinter as ctk
from workouts import WorkoutPage
from profile import ProfilePage
from profile_files.edit_profile import EditProfilePage
from profile_files.reminders import RemindersPage
from profile_files.sound_settings import SoundSettingsPage
from profile_files.feedback import FeedbackPage
from workout_window import WorkoutWindow
from PIL import Image
import pymongo
import sys
from utils import get_user_weight
from app_statistics import StatisticsPage
from bson import ObjectId
from challenges import ChallengesPage

def get_theme_colors():
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {
            "bg": "#000000",
            "card": "#1a1a1a",
            "text": "white",
            "subtext": "lightgray",
            "accent": "#3B8ED0",
            "nav_bg": "#000000",
            "nav_text": "white"
        }
    else:
        return {
            "bg": "#ffffff",
            "card": "#f0f0f0",
            "text": "black",
            "subtext": "#333333",
            "accent": "#0078D7",
            "nav_bg": "#ffffff",
            "nav_text": "black"
        }

class HomeApp(ctk.CTk):
    def __init__(self, logged_user, theme="Dark"):
        super().__init__()

        ctk.set_appearance_mode(theme)
        #ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.logged_user = logged_user
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["testapp"]
        self.collection = self.db["users"]

        if not self.logged_user:
            self.redirect_to_login()
            return

        self.title("Fitness App")
        self.geometry("1280x720")
        self.current_data = self.load_profile_data(self.logged_user)
        self.user_weight = float(get_user_weight())

        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.pack(fill="both", expand=True)

        self.bottom_nav_frame = ctk.CTkFrame(self)
        self.bottom_nav_frame.pack(side="bottom", fill="x")

        self.home_icon = Image.open("icons/home.png").convert("RGBA")
        self.workout_icon = Image.open("icons/workouts.png").convert("RGBA")
        self.statistics_icon = Image.open("icons/statistics.png").convert("RGBA")
        self.profile_icon = Image.open("icons/profile.png").convert("RGBA")
        self.challenges_icon = Image.open("icons/challenges.png").convert("RGBA")

        self.theme_switch = ctk.CTkSwitch(
            self.bottom_nav_frame,
            text="Dark Mode",
            command=self.toggle_theme,
            switch_height=20,
            switch_width=40
        )
        self.theme_switch.select()

        self.create_nav_buttons()
        self.apply_theme_to_navbar()
        self.show_home()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_nav_buttons(self):
        self.home_button = ctk.CTkButton(self.bottom_nav_frame, image=self.create_tinted_icon(self.home_icon),
                                         text=t("key_38"), compound="top", command=self.show_home, fg_color="transparent")
        self.workout_button = ctk.CTkButton(self.bottom_nav_frame, image=self.create_tinted_icon(self.workout_icon),
                                            text=t("key_41"), compound="top", command=self.show_workouts, fg_color="transparent")
        self.statistics_button = ctk.CTkButton(self.bottom_nav_frame, image=self.create_tinted_icon(self.statistics_icon),
                                               text=t("key_40"), compound="top", command=self.show_statistics, fg_color="transparent")
        self.profile_button = ctk.CTkButton(self.bottom_nav_frame, image=self.create_tinted_icon(self.profile_icon),
                                            text=t("key_39"), compound="top", command=self.show_profile, fg_color="transparent")
        self.challenges_button = ctk.CTkButton(self.bottom_nav_frame, image=self.create_tinted_icon(self.challenges_icon),
                                               text=t("key_44"), compound="top", command=self.show_challenges, fg_color="transparent")

        self.home_button.pack(side="left", expand=True)
        self.workout_button.pack(side="left", expand=True)
        self.statistics_button.pack(side="left", expand=True)
        self.challenges_button.pack(side="left", expand=True)
        self.profile_button.pack(side="left", expand=True)
        self.theme_switch.pack(side="right", padx=10, pady=5)

    def apply_theme_to_navbar(self):
        colors = get_theme_colors()
        self.bottom_nav_frame.configure(fg_color=colors["nav_bg"])
        for btn in [self.home_button, self.workout_button, self.statistics_button,
                    self.profile_button, self.challenges_button]:
            btn.configure(text_color=colors["nav_text"])
        self.theme_switch.configure(
            text=t("key_167") if ctk.get_appearance_mode() == "Dark" else t("key_168"),
            text_color=colors["nav_text"]
        )

    def toggle_theme(self):
        if self.theme_switch.get():
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")
        self.apply_theme_to_navbar()
        self.update_nav_icons()
        self.clear_content()
        self.reload_current_page()

    def reload_current_page(self):
        if hasattr(self, "current_page"):
            if self.current_page == "home":
                self.show_home()
            elif self.current_page == "workouts":
                self.show_workouts()
            elif self.current_page == "profile":
                self.show_profile()
            elif self.current_page == "statistics":
                self.show_statistics()
            elif self.current_page == "challenges":
                self.show_challenges()

    def update_nav_icons(self):
        self.home_button.configure(image=self.create_tinted_icon(self.home_icon))
        self.workout_button.configure(image=self.create_tinted_icon(self.workout_icon))
        self.statistics_button.configure(image=self.create_tinted_icon(self.statistics_icon))
        self.profile_button.configure(image=self.create_tinted_icon(self.profile_icon))
        self.challenges_button.configure(image=self.create_tinted_icon(self.challenges_icon))

    def update_navbar_language(self):
        self.home_button.configure(text=t("key_38"))
        self.workout_button.configure(text=t("key_41"))
        self.statistics_button.configure(text=t("key_40"))
        self.challenges_button.configure(text=t("key_44"))
        self.profile_button.configure(text=t("key_39"))


    def create_tinted_icon(self, icon):
        mode = ctk.get_appearance_mode()
        tint_color = (255, 255, 255) if mode == "Dark" else (0, 0, 0)
        tinted_image = Image.new("RGBA", icon.size)
        for x in range(icon.width):
            for y in range(icon.height):
                r, g, b, a = icon.getpixel((x, y))
                tinted_image.putpixel((x, y), (*tint_color, a))
        return ctk.CTkImage(tinted_image, size=(32, 32))

    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def show_navigation_bar(self):
        self.bottom_nav_frame.pack(side="bottom", fill="x")

    def hide_navigation_bar(self):
        self.bottom_nav_frame.pack_forget()

    def show_home(self):
        self.current_page = "home"
        self.clear_content()
        self.show_navigation_bar()
        HomePage(self.content_frame, self).pack(fill="both", expand=True)

    def show_workouts(self):
        self.current_page = "workouts"
        self.clear_content()
        WorkoutPage(self.content_frame, self).pack(fill="both", expand=True)

    def show_statistics(self):
        self.current_page = "statistics"
        self.clear_content()
        user_data = self.collection.find_one({"username": self.logged_user})
        if user_data:
            StatisticsPage(self.content_frame, str(user_data["_id"]), self).pack(fill="both", expand=True)
            self.show_navigation_bar()

    def show_profile(self):
        self.current_page = "profile"
        self.clear_content()
        ProfilePage(self.content_frame, self).pack(fill="both", expand=True)
        #self.main_app.update_language_ui()

    def show_challenges(self):
        self.current_page = "challenges"
        self.clear_content()
        user_data = self.collection.find_one({"username": self.logged_user})
        if user_data:
            ChallengesPage(self.content_frame, str(user_data["_id"])).pack(fill="both", expand=True)
            self.show_navigation_bar()

    def show_edit_profile_page(self, user_data=None):
        self.clear_content()
        if user_data is None:
            user_data = self.current_data
        self.edit_profile_page = EditProfilePage(self.content_frame, self, user_data)
        self.edit_profile_page.pack(fill="both", expand=True)

    def show_reminders_page(self):
        self.clear_content()
        RemindersPage(self.content_frame, self).pack(fill="both", expand=True)

    def show_sound_settings_page(self):
        self.clear_content()
        SoundSettingsPage(self.content_frame, self).pack(fill="both", expand=True)

    def show_feedback_page(self):
        self.clear_content()
        FeedbackPage(self.content_frame, self).pack(fill="both", expand=True)

    def update_profile(self, updated_data):
        current_username = updated_data.get("username", "")
        user = self.collection.find_one({"username": current_username})
        if user:
            try:
                result = self.collection.update_one(
                    {"_id": user["_id"]},
                    {"$set": updated_data}
                )
                if result.modified_count > 0:
                    self.current_data = updated_data
                    return True
            except Exception as e:
                print(f"Error while updating user: {e}")
        return False
    
    def user_exists(self, username):
        return self.collection.find_one({"username": username}) is not None

    def update_language_ui(self):
        self.update_navbar_language()
        self.apply_theme_to_navbar()


    def on_close(self):
        if self.client:
            self.client.close()
        self.quit()
        self.destroy()
        sys.exit(0)

    def redirect_to_login(self):
        from login import LoginApp
        self.destroy()
        login_app = LoginApp()
        login_app.mainloop()

    def load_profile_data(self, username):
        user = self.collection.find_one({"username": username})
        return {
            "name": user.get("name", "Unknown"),
            "username": user["username"],
            "birthday": user.get("birthday", ""),
            "goal": user.get("goal", ""),
            "height_cm": user.get("height_cm", ""),
            "weight_kg": user.get("weight_kg", "")
        }
    
    def show_plan_30_days(self, plan_name):
        from home_plan_workouts.plan_30_days import Plan30Days
        self.current_page = "home_plan"
        self.clear_content()
        Plan30Days(self.content_frame, self, plan_name).pack(fill="both", expand=True)


class HomePage(ctk.CTkFrame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        colors = get_theme_colors()
        self.configure(fg_color=colors["bg"])

        title_label = ctk.CTkLabel(
            self, text=t("key_42"),
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=colors["text"]
        )
        title_label.pack(pady=(20, 10))

        scrollable_frame = ctk.CTkScrollableFrame(self, fg_color=colors["bg"], width=800, height=500)
        scrollable_frame.pack(pady=(10, 20), padx=20, fill="both", expand=True)

        plans = [
            {"title": t("key_45"), "goal": t("key_51"), "color": "gold"},
            {"title": t("key_46"), "goal": t("key_51"), "color": "silver"},
            {"title": t("key_47"), "goal": t("key_52"), "color": "green"},
            {"title": t("key_48"), "goal": t("key_52"), "color": "silver"},
            {"title": t("key_49"), "goal": t("key_53"), "color": "purple"},
            {"title": t("key_50"), "goal": t("key_53"), "color": "silver"}
        ]

        for plan in plans:
            self.add_plan_card(scrollable_frame, plan)

    def add_plan_card(self, parent, plan):
        colors = get_theme_colors()
        card = ctk.CTkFrame(parent, fg_color=colors["card"], corner_radius=10)
        card.pack(pady=(10, 5), padx=20, fill="x")

        #medal_icon = "🥇" if plan["color"] == "gold" else "🥈" if plan["color"] == "silver" else "🥉"

        title_label = ctk.CTkLabel(
            card,
            text=f"{plan['title']}",
            font=ctk.CTkFont(size=14),
            text_color=colors["text"]
        )
        title_label.pack(anchor="w", pady=5)

        goal_label = ctk.CTkLabel(
            card,
            text=f"{t('key_55')} {plan['goal']}",
            font=ctk.CTkFont(size=12),
            text_color=colors["subtext"]
        )
        goal_label.pack(anchor="w")

        view_button = ctk.CTkButton(
            card,
            text=t("key_54"),
            command=lambda: self.main_app.show_plan_30_days(plan["title"]),
            fg_color=colors["accent"],
            text_color="white"
        )
        view_button.pack(pady=(10, 5), anchor="e")

    

if __name__ == "__main__":
    from login import LoginApp
    login_app = LoginApp()
    login_app.mainloop()

    if login_app.logged_user:
        app = HomeApp(login_app.logged_user)
        app.mainloop()
