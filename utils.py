import pymongo
from datetime import datetime

def get_user_weight():
    #to get the weight
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["testapp"]
    users_collection = db["users"]

    user = users_collection.find_one({"is_logged_in": True})  

    if user:
        if "weight_kg" in user:
            try:
                weight = user["weight_kg"]
                if isinstance(weight, str):  
                    weight = float(weight)

                print(f"Greutatea utilizatorului preluata din MongoDB: {weight} kg")
                return weight
            except ValueError:
                print(f"Eroare: Greutatea utilizatorului '{user['weight_kg']}' nu poate fi convertita la float, folosim 70 kg")
                return 70.0
        else:
            print("Greutatea utilizatorului NU EXISTA in MongoDB, folosim 70 kg.")
            return 70.0
    else:
        print("Nu s-a gasit niciun utilizator logat, folosim 70 kg")
        return 70.0


def get_user_age():
    #to get the age    
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["testapp"]
    users_collection = db["users"]

    user = users_collection.find_one({"is_logged_in": True})  

    if user and "birthday" in user:
        try:
            #convert the data
            birth_date = datetime.strptime(user["birthday"], "%d/%m/%Y")
            today = datetime.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return age
        except Exception as e:
            print(f"Eroare la conversia datei de nastere: {e}")
            return None  
    else:
        print("Data de nastere nu este setata in baza de date")
        return None