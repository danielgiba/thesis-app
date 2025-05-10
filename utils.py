import pymongo
from datetime import datetime

def get_user_weight():
    """Extrage greutatea utilizatorului logat din baza de date MongoDB și afișează mesaje de debugging."""

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["testapp"]
    users_collection = db["users"]

    user = users_collection.find_one({"is_logged_in": True})  # 🔥 Verificăm doar utilizatorii logați

    if user:
        if "weight_kg" in user:
            try:
                weight = user["weight_kg"]
                if isinstance(weight, str):  # 🔥 Convertim la float dacă e string
                    weight = float(weight)

                print(f"✅ Greutatea utilizatorului preluată din MongoDB: {weight} kg")
                return weight
            except ValueError:
                print(f"⚠️ Eroare: Greutatea utilizatorului '{user['weight_kg']}' nu poate fi convertită la float! Folosim 70 kg.")
                return 70.0
        else:
            print("⚠️ Greutatea utilizatorului NU EXISTĂ în MongoDB! Folosim 70 kg.")
            return 70.0
    else:
        print("❌ Nu s-a găsit niciun utilizator logat! Folosim 70 kg.")
        return 70.0


def get_user_age():
    """Extrage vârsta utilizatorului din baza de date MongoDB și o returnează."""
    
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["testapp"]
    users_collection = db["users"]

    user = users_collection.find_one({"is_logged_in": True})  # Găsim utilizatorul conectat

    if user and "birthday" in user:
        try:
            # Convertim din formatul DD/MM/YYYY în obiect datetime
            birth_date = datetime.strptime(user["birthday"], "%d/%m/%Y")
            today = datetime.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return age
        except Exception as e:
            print(f"⚠️ Eroare la conversia datei de naștere: {e}")
            return None  # NU returnăm o valoare default, ci None
    else:
        print("⚠️ Data de naștere nu este setată în baza de date!")
        return None  # NU returnăm o valoare default, ci None