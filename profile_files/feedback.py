from language_manager import t
import customtkinter as ctk

class FeedbackPage(ctk.CTkFrame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.configure(fg_color=ctk.ThemeManager.theme["CTkFrame"]["fg_color"])

        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", pady=(10, 0))

        ctk.CTkButton(top_frame, text=t("key_113"), fg_color="transparent", text_color="blue",
                      command=self.back_to_profile).pack(side="left", padx=20)

        ctk.CTkLabel(self, text=t("key_162"), font=("Arial", 20, "bold"),
                     text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"]).pack(pady=10)

        ctk.CTkLabel(self, text=t("key_163"), text_color="gray").pack()
        self.textbox = ctk.CTkTextbox(self, width=400, height=150)
        self.textbox.pack(pady=10)

        ctk.CTkButton(self, text=t("key_165"), command=self.send_feedback).pack()
        self.status_label = ctk.CTkLabel(self, text="")
        self.status_label.pack(pady=5)

    def send_feedback(self):
        feedback = self.textbox.get("1.0", "end").strip()
        if feedback:
            print(f"Feedback primit: {feedback}")
            self.textbox.delete("1.0", "end")
            self.status_label.configure(text=t("key_166"), text_color="green")

    def back_to_profile(self):
        self.main_app.show_profile()
