from language_manager import t, set_language, get_current_language
import customtkinter as ctk
import pymongo
import bcrypt
import pyotp
from email.message import EmailMessage
import re
import smtplib

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

def hash_password(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def check_password(stored_password, input_password):
    return bcrypt.checkpw(input_password.encode(), stored_password.encode("utf-8"))

def generate_otp():
    return str(pyotp.TOTP(pyotp.random_base32()).now())

def send_otp(email, otp_code):
    sender_email = "gibadaniel717@gmail.com"
    sender_password = "kwut tipm elfa hdgt"

    msg = EmailMessage()
    msg.set_content(f"Your verification code is: {otp_code}")
    msg["Subject"] = "Your Fitness App Verification Code"
    msg["From"] = f"Fitness App <{sender_email}>"
    msg["To"] = email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("OTP sent successfully!")
    except Exception as e:
        print(f"⚠ Failed to send OTP: {e}")

class LoginApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Login - Fitness App")
        self.geometry("500x800")
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["testapp"]
        self.collection = self.db["users"]
        self.logged_user = None

        set_language(get_current_language())
        self.theme_label = None
        self.theme_switch = None
        self.create_widgets()

    def create_widgets(self):
        self.theme_label = ctk.CTkLabel(self, text=ctk.get_appearance_mode())
        self.theme_label.place(relx=0.03, rely=0.02, anchor="nw")

        self.theme_switch = ctk.CTkSwitch(
            self,
            text="",
            command=self.toggle_theme
        )
        self.theme_switch.place(relx=0.13, rely=0.02, anchor="nw")
        self.theme_switch.select() if ctk.get_appearance_mode() == "Dark" else self.theme_switch.deselect()

        self.lang_dropdown = ctk.CTkComboBox(self, values=["ro", "en"], width=80)
        self.lang_dropdown.set(get_current_language())
        self.lang_dropdown.place(relx=0.98, rely=0.02, anchor="ne")

        self.lang_dropdown.configure(command=lambda lang: [set_language(lang), self.update_texts()])

        self.label = ctk.CTkLabel(self, text=t("key_1"), font=("Arial", 24))
        self.label.pack(pady=50)

        self.username_entry = ctk.CTkEntry(self, width=220, placeholder_text=t("key_2"))
        self.username_entry.pack(pady=10)

        self.password_entry = ctk.CTkEntry(self, width=220, show="*", placeholder_text=t("key_3"))
        self.password_entry.pack(pady=10)

        self.show_password_var = ctk.BooleanVar()
        self.show_password_check = ctk.CTkCheckBox(self, text=t("key_4"), variable=self.show_password_var, command=self.toggle_password_visibility)
        self.show_password_check.pack()

        self.login_button = ctk.CTkButton(self, text=t("key_1"), command=self.login, width=200)
        self.login_button.pack(pady=10)

        self.signup_button = ctk.CTkButton(self, text=t("key_5"), command=self.show_signup, width=200)
        self.signup_button.pack(pady=10)

        self.status_label = ctk.CTkLabel(self, text="", text_color="red")
        self.status_label.pack(pady=10)

        self.forgot_password_button = ctk.CTkButton(
            self,
            text=t("key_6"),
            font=("Arial", 12),
            fg_color="transparent",
            text_color=("black", "white"),
            hover_color=("#e6f2ff", "#333333"),
            command=self.show_password_reset_inline
        )
        self.forgot_password_button.pack(pady=(5, 0))


        self.reset_frame = ctk.CTkFrame(self)
        self.reset_frame.pack_forget()

        self.reset_label = ctk.CTkLabel(self.reset_frame, text=t("key_7"))
        self.reset_label.pack(pady=5)

        self.reset_email_entry = ctk.CTkEntry(self.reset_frame, width=220, placeholder_text=t("key_8"))
        self.reset_email_entry.pack(pady=5)

        self.reset_status_label = ctk.CTkLabel(self.reset_frame, text="", text_color="red")
        self.reset_status_label.pack(pady=5)

        self.send_otp_button = ctk.CTkButton(self.reset_frame, text=t("key_9"), command=self.send_reset_email, width=200)
        self.send_otp_button.pack(pady=5)

        self.update_texts()

    def toggle_theme(self):
        current = ctk.get_appearance_mode()
        new_theme = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_theme)
        self.theme_label.configure(text=new_theme)

    def update_texts(self):
        self.label.configure(text=t("key_1"))
        self.username_entry.configure(placeholder_text=t("key_2"))
        self.password_entry.configure(placeholder_text=t("key_3"))
        self.show_password_check.configure(text=t("key_4"))
        self.login_button.configure(text=t("key_1"))
        self.signup_button.configure(text=t("key_5"))
        self.forgot_password_button.configure(text=t("key_6"))
        self.reset_label.configure(text=t("key_7"))
        self.reset_email_entry.configure(placeholder_text=t("key_8"))
        self.send_otp_button.configure(text=t("key_9"))
        self.reset_status_label.configure(text="")

    def login(self):
        from home import HomeApp
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        user = self.collection.find_one({"username": username})
        if user and check_password(user["password"], password):
            self.collection.update_one({"username": username}, {"$set": {"is_logged_in": True}})
            self.logged_user = username
            self.destroy()
            home_app = HomeApp(self.logged_user, ctk.get_appearance_mode())
            home_app.mainloop()
        else:
            self.status_label.configure(text=t("key_10"))

    def toggle_password_visibility(self):
        if self.show_password_var.get():
            self.password_entry.configure(show="")
        else:
            self.password_entry.configure(show="*")

    def show_password_reset_inline(self):
        self.reset_frame.pack(pady=10)

    def send_reset_email(self):
        self.email_for_reset = self.reset_email_entry.get().strip()
        user = self.collection.find_one({"email": self.email_for_reset})
        if user:
            self.generated_otp = generate_otp()
            send_otp(self.email_for_reset, self.generated_otp)

            self.reset_status_label.configure(text=t("key_13"), text_color="green")

            self.new_password_entry = ctk.CTkEntry(self.reset_frame, placeholder_text=t("key_14"), show="*")
            self.new_password_entry.pack(pady=5)

            self.confirm_password_entry = ctk.CTkEntry(self.reset_frame, placeholder_text=t("key_15"), show="*")
            self.confirm_password_entry.pack(pady=5)

            self.otp_entry = ctk.CTkEntry(self.reset_frame, placeholder_text=t("key_16"))
            self.otp_entry.pack(pady=5)

            self.reset_confirm_button = ctk.CTkButton(
                self.reset_frame, text=t("key_17"), command=self.verify_otp_and_reset_password
            )
            self.reset_confirm_button.pack(pady=5)
        else:
            self.reset_status_label.configure(text=t("key_12"), text_color="red")

    def verify_otp_and_reset_password(self):
        entered_otp = self.otp_entry.get().strip()
        new_pass = self.new_password_entry.get().strip()
        confirm_pass = self.confirm_password_entry.get().strip()

        if entered_otp != self.generated_otp:
            self.reset_status_label.configure(text=t("key_18"), text_color="red")
            return
        if new_pass != confirm_pass:
            self.reset_status_label.configure(text=t("key_19"), text_color="red")
            return
        if len(new_pass) < 6:
            self.reset_status_label.configure(text=t("key_20"), text_color="red")
            return

        hashed = hash_password(new_pass)
        self.collection.update_one({"email": self.email_for_reset}, {"$set": {"password": hashed}})
        self.reset_status_label.configure(text=t("key_21"), text_color="green")

    def show_signup(self):
        self.destroy()
        SignupApp().mainloop()

class SignupApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sign Up - Fitness App")
        self.geometry("500x750")
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["testapp"]
        self.collection = self.db["users"]
        set_language(get_current_language())

        self.theme_label = None
        self.theme_switch = None

        self.create_widgets()

    def create_widgets(self):
        self.theme_label = ctk.CTkLabel(self, text=ctk.get_appearance_mode())
        self.theme_label.place(relx=0.03, rely=0.02, anchor="nw")

        self.theme_switch = ctk.CTkSwitch(
            self,
            text="",
            command=self.toggle_theme
        )
        self.theme_switch.place(relx=0.13, rely=0.02, anchor="nw")
        self.theme_switch.select() if ctk.get_appearance_mode() == "Dark" else self.theme_switch.deselect()

        self.lang_dropdown = ctk.CTkComboBox(self, values=["ro", "en"], width=80)
        self.lang_dropdown.set(get_current_language())
        self.lang_dropdown.place(relx=0.98, rely=0.02, anchor="ne")
        self.lang_dropdown.configure(command=lambda lang: [set_language(lang), self.update_texts()])

        self.label = ctk.CTkLabel(self, text=t("key_22"), font=("Arial", 24))
        self.label.pack(pady=40)

        self.name_entry = ctk.CTkEntry(self, width=220, placeholder_text=t("key_23"))
        self.name_entry.pack(pady=10)

        self.username_entry = ctk.CTkEntry(self, width=220, placeholder_text=t("key_2"))
        self.username_entry.pack(pady=10)
        self.username_status = ctk.CTkLabel(self, text="", text_color="green")
        self.username_status.pack()
        self.username_entry.bind("<KeyRelease>", self.check_username_availability)

        self.email_entry = ctk.CTkEntry(self, width=220, placeholder_text=t("key_8"))
        self.email_entry.pack(pady=10)

        self.password_entry = ctk.CTkEntry(self, width=220, placeholder_text=t("key_3"), show="*")
        self.password_entry.pack(pady=10)

        self.show_password_var = ctk.BooleanVar()
        self.show_password_check = ctk.CTkCheckBox(
            self, text=t("key_4"), variable=self.show_password_var,
            command=self.toggle_password_visibility
        )
        self.show_password_check.pack()

        self.gender_selector = ctk.CTkComboBox(
            self, values=[t("key_126"), t("key_127"), t("key_128")], state="readonly", width=220
        )
        self.gender_selector.set(t("key_125"))
        self.gender_selector.pack(pady=10)

        self.birthday_entry = ctk.CTkEntry(self, width=220, placeholder_text=t("key_24"))
        self.birthday_entry.pack(pady=10)

        self.goal_selector = ctk.CTkComboBox(
            self, values=[t("key_130"), t("key_131"), t("key_132")], state="readonly", width=220
        )
        self.goal_selector.set(t("key_129"))
        self.goal_selector.pack(pady=10)

        self.height_entry = ctk.CTkEntry(self, width=220, placeholder_text=t("key_25"))
        self.height_entry.pack(pady=10)

        self.weight_entry = ctk.CTkEntry(self, width=220, placeholder_text=t("key_26"))
        self.weight_entry.pack(pady=10)

        self.signup_button = ctk.CTkButton(self, text=t("key_22"), command=self.signup, width=200)
        self.signup_button.pack(pady=10)

        self.back_button = ctk.CTkButton(self, text=t("key_27"), command=self.go_back_to_login, width=200)
        self.back_button.pack(pady=5)

        self.status_label = ctk.CTkLabel(self, text="", text_color="red")
        self.status_label.pack(pady=10)

        self.update_texts()

    def toggle_theme(self):
        current = ctk.get_appearance_mode()
        new_theme = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_theme)
        self.theme_label.configure(text=new_theme)

    def update_texts(self):
        self.label.configure(text=t("key_22"))
        self.name_entry.configure(placeholder_text=t("key_23"))
        self.username_entry.configure(placeholder_text=t("key_2"))
        self.email_entry.configure(placeholder_text=t("key_8"))
        self.password_entry.configure(placeholder_text=t("key_3"))
        self.gender_selector.configure(values=[t("key_126"), t("key_127"), t("key_128")])
        self.gender_selector.set(t("key_125"))
        self.birthday_entry.configure(placeholder_text=t("key_24"))
        self.goal_selector.configure(values=[t("key_130"), t("key_131"), t("key_132")])
        self.goal_selector.set(t("key_129"))
        self.height_entry.configure(placeholder_text=t("key_25"))
        self.weight_entry.configure(placeholder_text=t("key_26"))
        self.show_password_check.configure(text=t("key_4"))
        self.signup_button.configure(text=t("key_22"))
        self.back_button.configure(text=t("key_27"))

    def toggle_password_visibility(self):
        if self.show_password_var.get():
            self.password_entry.configure(show="")
        else:
            self.password_entry.configure(show="*")

    def check_username_availability(self, event=None):
        username = self.username_entry.get().strip()
        if not re.fullmatch(r"[a-z0-9._]+", username):
            self.username_status.configure(text=t("key_34"), text_color="red")
            return
        if self.collection.find_one({"username": username}):
            self.username_status.configure(text=t("key_35"), text_color="red")
        else:
            self.username_status.configure(text=t("key_36"), text_color="green")

    def go_back_to_login(self):
        self.destroy()
        LoginApp().mainloop()

    def signup(self):
        name = self.name_entry.get().strip()
        username = self.username_entry.get().strip()
        if not re.fullmatch(r"[a-z._]+", username):
            self.status_label.configure(text=t("key_28"))
            return

        email = self.email_entry.get().strip()
        password = self.password_entry.get().strip()
        gender = self.gender_selector.get().strip()
        birthday = self.birthday_entry.get().strip()
        goal = self.goal_selector.get().strip()
        height = self.height_entry.get().strip()
        weight = self.weight_entry.get().strip()

        if self.collection.find_one({"username": username}):
            self.status_label.configure(text=t("key_29"))
            return
        if self.collection.find_one({"email": email}):
            self.status_label.configure(text=t("key_30"))
            return
        try:
            weight = float(weight)
        except ValueError:
            self.status_label.configure(text=t("key_31"))
            return
        try:
            height = int(height)
        except ValueError:
            self.status_label.configure(text=t("key_32"))
            return

        otp_code = generate_otp()
        send_otp(email, otp_code)

        self.otp_entry = ctk.CTkEntry(self, placeholder_text=t("key_16"), width=300)
        self.otp_entry.pack(pady=10)
        self.verify_button = ctk.CTkButton(
            self, text=t("key_33"),
            command=lambda: self.verify_signup_otp(
                otp_code, name, username, email, password,
                gender, birthday, goal, height, weight
            ),
            width=300
        )
        self.verify_button.pack(pady=10)

    def verify_signup_otp(self, correct_otp, name, username, email, password, gender, birthday, goal, height, weight):
        entered_otp = self.otp_entry.get().strip()
        if entered_otp == correct_otp:
            hashed_password = hash_password(password)
            self.collection.insert_one({
                "name": name,
                "username": username,
                "email": email,
                "password": hashed_password,
                "gender": gender,
                "birthday": birthday,
                "goal": goal,
                "height_cm": height,
                "weight_kg": weight
            })
            self.destroy()
            LoginApp().mainloop()
        else:
            self.status_label.configure(text=t("key_37"))

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")  # Default
    login_app = LoginApp()
    login_app.mainloop()
