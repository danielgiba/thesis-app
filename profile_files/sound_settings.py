from language_manager import t
import customtkinter as ctk

class SoundSettingsPage(ctk.CTkFrame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.configure(fg_color=ctk.ThemeManager.theme["CTkFrame"]["fg_color"])

        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", pady=(10, 0))

        ctk.CTkButton(top_frame, text=t("key_113"), fg_color="transparent", text_color="blue",
                      command=self.back_to_profile).pack(side="left", padx=20)

        ctk.CTkLabel(self, text=t("key_150"), font=("Arial", 20, "bold"),
                     text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"]).pack(pady=10)

        self.music_label = ctk.CTkLabel(self, text=t("key_151"))
        self.music_label.pack()
        self.music_slider = ctk.CTkSlider(self, from_=0, to=100, command=self.update_music)
        self.music_slider.pack(pady=5)

        self.voice_label = ctk.CTkLabel(self, text=t("key_152"))
        self.voice_label.pack()
        self.voice_slider = ctk.CTkSlider(self, from_=0, to=100, command=self.update_voice)
        self.voice_slider.pack(pady=5)

        ctk.CTkButton(self, text=t("key_153"), command=self.save_preferences).pack(pady=10)
        self.status_label = ctk.CTkLabel(self, text="")
        self.status_label.pack()

    def update_music(self, value):
        pass

    def update_voice(self, value):
        pass

    def save_preferences(self):
        self.status_label.configure(text=t("key_154"), text_color="green")

    def back_to_profile(self):
        self.main_app.show_profile()
