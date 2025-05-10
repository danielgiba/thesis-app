import json
import os

LANG_PATH = os.path.join(os.path.dirname(__file__), "translations.json")
LANG_FILE = os.path.join(os.path.dirname(__file__), "lang.txt")

_current_lang = "ro"

def get_current_language():
    global _current_lang
    return _current_lang

def set_language(lang_code):
    global _current_lang
    _current_lang = lang_code
    with open(LANG_FILE, "w", encoding="utf-8") as f:
        f.write(lang_code)

def load_translations():
    with open(LANG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

translations = load_translations()

def t(key):
    return translations.get(get_current_language(), {}).get(key, key)

def get_available_languages():
    return ["ro", "en"]
