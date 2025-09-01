from language_manager import t
import customtkinter as ctk
from tkinter import StringVar

def get_theme_colors():
    mode = ctk.get_appearance_mode()
    if mode == "Dark":
        return {
            "bg": "#000000",
            "card": "#1a1a1a",
            "text": "white",
            "entry_bg": "#222222",
            "accent": "#3B8ED0",
            "error": "red",
            "success": "lightgreen"
        }
    else:
        return {
            "bg": "#ffffff",
            "card": "#f2f2f2",
            "text": "black",
            "entry_bg": "#ffffff",
            "accent": "#0078D7",
            "error": "red",
            "success": "green"
        }

class EditProfilePage(ctk.CTkFrame):
    def __init__(self, parent, main_app, current_data=None):
        super().__init__(parent)
        self.main_app = main_app
        self.colors = get_theme_colors()

        if current_data is None:
            self.current_data = self.main_app.load_profile_data(self.main_app.logged_user["username"])
        else:
            self.current_data = current_data

        self.configure(fg_color=self.colors["bg"])

        top_buttons_frame = ctk.CTkFrame(self, fg_color=self.colors["bg"])
        top_buttons_frame.pack(fill="x", pady=(10, 0))

        back_button = ctk.CTkButton(
            top_buttons_frame,
            text=t("key_27"),
            fg_color=self.colors["bg"],
            text_color=self.colors["accent"],
            command=self.back_to_profile
        )
        back_button.pack(side="left", padx=(20, 0))

        save_button = ctk.CTkButton(
            top_buttons_frame,
            text=t("key_142"),
            fg_color=self.colors["accent"],
            text_color="white",
            command=self.save_changes
        )
        save_button.pack(side="right", padx=(0, 20))

        # search_frame = ctk.CTkFrame(self, fg_color=self.colors["bg"])
        # search_frame.pack(fill="x", pady=(10, 10), padx=20)

        # search_label = ctk.CTkLabel(search_frame, text=t("key_133"), text_color=self.colors["text"])
        # search_label.pack(side="left", padx=(0, 10))

        # self.search_var = StringVar()
        # search_entry = ctk.CTkEntry(search_frame, textvariable=self.search_var, fg_color=self.colors["entry_bg"], text_color=self.colors["text"])
        # search_entry.pack(side="left", fill="x", expand=True)

        # search_button = ctk.CTkButton(
        #     search_frame,
        #     text=t("key_134"),
        #     fg_color=self.colors["accent"],
        #     text_color="white",
        #     command=self.search_user
        # )
        # search_button.pack(side="left", padx=(10, 0))

        fields = [
            (t("key_143"), "name"),
            (t("key_144"), "username"),
            (t("key_145"), "gender", [t("key_126"), t("key_127")]),
            (t("key_146"), "birthday"),
            (t("key_147"), "goal", [t("key_130"), t("key_131"), t("key_132")]),
            (t("key_148"), "height_cm"),
            (t("key_149"), "weight_kg")
        ]

        self.variables = {}
        for label_text, field, *values in fields:
            self.add_field(label_text, field, values[0] if values else None)

        self.status_label = ctk.CTkLabel(self, text="", text_color=self.colors["success"])
        self.status_label.pack(pady=(10, 0))

    def add_field(self, label_text, field, options=None):
        label = ctk.CTkLabel(self, text=label_text, text_color=self.colors["text"])
        label.pack(pady=(5, 2))

        if options:
            var = StringVar(value=self.current_data.get(field, options[0]))
            menu = ctk.CTkOptionMenu(self, values=options, variable=var, fg_color=self.colors["accent"], text_color="white")
            menu.pack(pady=(0, 5))
            self.variables[field] = var
        else:
            var = StringVar(value=self.current_data.get(field, ""))
            entry = ctk.CTkEntry(self, textvariable=var, fg_color=self.colors["entry_bg"], text_color=self.colors["text"])
            entry.pack(pady=(0, 5))
            self.variables[field] = var

    def search_user(self):
        username = self.search_var.get()
        if not username:
            self.status_label.configure(text=t("key_135"), text_color=self.colors["error"])
            return

        user_data = self.main_app.load_profile_data(username)
        if user_data:
            self.status_label.configure(text=t("key_136").format(username=username), text_color=self.colors["success"])
            for field, var in self.variables.items():
                var.set(user_data.get(field, ""))
            self.current_data = user_data
        else:
            self.status_label.configure(text=t("key_137").format(username=username), text_color=self.colors["error"])

    def back_to_profile(self):
        self.main_app.show_profile()

    def save_changes(self):
        updated_data = {field: var.get() for field, var in self.variables.items()}

        user_exists = self.main_app.user_exists(updated_data["username"])
        if user_exists:
            success = self.main_app.update_profile(updated_data)
            if success:
                self.status_label.configure(text=t("key_138"), text_color=self.colors["success"])
            else:
                self.status_label.configure(text=t("key_139"), text_color=self.colors["error"])
        else:
            success = self.main_app.create_user(updated_data)
            if success:
                self.status_label.configure(text=t("key_140"), text_color=self.colors["success"])
                self.main_app.save_current_user(updated_data["username"])
            else:
                self.status_label.configure(text=t("key_141"), text_color=self.colors["error"])
