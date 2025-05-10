from language_manager import t, set_language, get_available_languages
import customtkinter as ctk
from PIL import Image
import pymongo


def get_theme_colors():
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {
            "bg": "#000000",
            "card": "#1a1a1a",
            "text": "white",
            "subtext": "lightgray",
            "accent": "#3B8ED0",
            "danger": "red",
            "hover": "#3B8ED0",
            "combo_fg": "white"
        }
    else:
        return {
            "bg": "#ffffff",
            "card": "#f2f2f2",
            "text": "black",
            "subtext": "#333333",
            "accent": "#0078D7",
            "danger": "#cc0000",
            "hover": "#005a9e",
            "combo_fg": "black"
        }


class ProfilePage(ctk.CTkFrame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.colors = get_theme_colors()

        self.configure(fg_color=self.colors["bg"])

        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["testapp"]
        self.collection = self.db["users"]

        # 🔹 Icons
        self.edit_profile_icon = Image.open("icons/edit_profile.png").convert("RGBA")
        self.reminders_icon = Image.open("icons/reminders.png").convert("RGBA")
        self.sound_icon = Image.open("icons/sound.png").convert("RGBA")
        self.feedback_icon = Image.open("icons/feedback.png").convert("RGBA")
        self.logout_icon = Image.open("icons/logout.png").convert("RGBA")
        self.delete_icon = Image.open("icons/delete-account.png").convert("RGBA")

        # 🔹 Titlu pagină
        profile_label = ctk.CTkLabel(self, text=t("key_102"), font=ctk.CTkFont(size=24, weight="bold"), text_color=self.colors["text"])
        profile_label.pack(pady=20)

        # 🔘 Butoane funcționale
        self.edit_profile_button = self.create_button(t("key_103"), self.edit_profile, self.edit_profile_icon)
        self.reminders_button = self.create_button(t("key_104"), self.open_reminders, self.reminders_icon)
        self.sound_button = self.create_button(t("key_105"), self.open_sound_settings, self.sound_icon)
        self.feedback_button = self.create_button(t("key_106"), self.open_feedback, self.feedback_icon)
        self.logout_button = self.create_button(t("key_107"), self.logout, self.logout_icon)

        # 🔴 Ștergere cont
        self.delete_account_button = ctk.CTkButton(
            self,
            text=t("key_108"),
            command=self.delete_account,
            image=self.create_tinted_icon(self.delete_icon),
            fg_color=self.colors["danger"],
            text_color="white",
            border_width=1,
            hover_color="#8b0000",
            width=200
        )
        self.delete_account_button.pack(pady=(5, 10), anchor="center")

        # 🔻 Dropdown limba
        ctk.CTkLabel(self, text=t("key_112"), text_color=self.colors["text"]).pack(pady=(10, 5), anchor="center")
        self.language_dropdown = ctk.CTkComboBox(
            self,
            values=get_available_languages(),
            command=self.change_language,
            width=160,
            state="readonly",
            dropdown_fg_color=self.colors["card"],
            dropdown_text_color=self.colors["combo_fg"]
        )
        self.language_dropdown.set(t("key_112"))
        self.language_dropdown.pack(pady=(0, 20), anchor="center")

        self.main_app.update_navbar_language()


    def create_button(self, text, command, icon):
        btn = ctk.CTkButton(
            self,
            text=text,
            command=command,
            image=self.create_tinted_icon(icon),
            fg_color=self.colors["card"],
            text_color=self.colors["text"],
            border_width=1,
            hover_color=self.colors["hover"],
            width=200
        )
        btn.pack(pady=5, anchor="center")
        return btn

    def create_tinted_icon(self, icon):
        mode = ctk.get_appearance_mode()
        tint_color = (255, 255, 255) if mode == "Dark" else (0, 0, 0)
        tinted_image = Image.new("RGBA", icon.size)
        for x in range(icon.width):
            for y in range(icon.height):
                r, g, b, a = icon.getpixel((x, y))
                tinted_image.putpixel((x, y), (*tint_color, a))
        return ctk.CTkImage(tinted_image, size=(32, 32))

    def change_language(self, lang_code):
        set_language(lang_code)
        if hasattr(self.main_app, "update_navbar_language"):
            self.main_app.update_navbar_language()
        if hasattr(self.main_app, "show_profile"):
            self.main_app.show_profile()

    def edit_profile(self):
        self.main_app.show_edit_profile_page()

    def open_reminders(self):
        self.main_app.show_reminders_page()

    def open_sound_settings(self):
        self.main_app.show_sound_settings_page()

    def open_feedback(self):
        self.main_app.show_feedback_page()

    def logout(self):
        try:
            from login import LoginApp
            self.collection.update_many({}, {"$set": {"is_logged_in": False}})
            self.main_app.destroy()
            login_app = LoginApp()
            login_app.mainloop()
        except Exception as e:
            print(f"Error during logout: {e}")

    def delete_account(self):
        confirm_label = ctk.CTkLabel(self, text=t("key_109"), text_color=self.colors["danger"])
        confirm_label.pack(pady=10)

        def confirm_delete():
            from login import LoginApp
            self.collection.delete_one({"username": self.main_app.logged_user})
            self.main_app.logged_user = None
            self.main_app.destroy()
            login_app = LoginApp()
            login_app.mainloop()

        yes_button = ctk.CTkButton(self, text=t("key_110"), fg_color=self.colors["danger"], command=confirm_delete, width=200)
        yes_button.pack(pady=5, anchor="center")

        no_button = ctk.CTkButton(self, text=t("key_111"), fg_color="gray", command=confirm_label.destroy, width=200)
        no_button.pack(pady=5, anchor="center")
