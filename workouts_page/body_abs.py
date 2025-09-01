import customtkinter as ctk
from workout_window import WorkoutWindow

class BodyAbsWorkout(WorkoutWindow):
    def __init__(self, parent, main_app):
        user_data = main_app.collection.find_one({"username": main_app.logged_user})
        user_id = str(user_data["_id"]) if user_data else "user_id_placeholder"
        self.main_app = main_app  

        super().__init__(
            parent,
            user_id,
            "workouts_page/tutorials/abs_workout.mp4",
            "Abs Workout",
            1,
            main_app.show_workouts,
            main_app
        )
        self.start_video()
