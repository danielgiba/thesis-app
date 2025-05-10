from language_manager import t
import customtkinter as ctk

class RemindersPage(ctk.CTkFrame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        self.configure(fg_color=ctk.ThemeManager.theme["CTkFrame"]["fg_color"])

        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", pady=(10, 0))

        ctk.CTkButton(top_frame, text=t("key_113"), fg_color="transparent", text_color="blue",
                      command=self.back_to_profile).pack(side="left", padx=20)

        ctk.CTkLabel(self, text=t("key_155"), font=("Arial", 20, "bold"),
                     text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"]).pack(pady=(10, 5))

        self.hour_label = ctk.CTkLabel(self, text=t("key_157"))
        self.hour_label.pack()
        self.hour_entry = ctk.CTkEntry(self)
        self.hour_entry.pack(pady=5)

        self.minute_label = ctk.CTkLabel(self, text=t("key_158"))
        self.minute_label.pack()
        self.minute_entry = ctk.CTkEntry(self)
        self.minute_entry.pack(pady=5)

        ctk.CTkButton(self, text=t("key_159"), command=self.add_reminder).pack(pady=10)

        self.status_label = ctk.CTkLabel(self, text="")
        self.status_label.pack()

    def add_reminder(self):
        self.status_label.configure(text=t("key_160"), text_color="green")

    def back_to_profile(self):
        self.main_app.show_profile()
