from pathlib import Path
import re


current_file = Path(__file__).resolve()

FOLDERS = [
    r"C:\Users\eddya\OneDrive\מסמכים\Python",
]

EXTENSIONS = {
    ".py", ".txt", ".json", ".env", ".ini", ".cfg", ".yaml", ".yml", ".conf", ".md", ".log", ".csv",
}

KEYWORDS = [
    "eddy",

    "password",
    "gmail",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "authorization",
    "smtp(",
    "username",
    "login",
    "email",
]

ENABLE_EMAIL_SEARCH = True
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


for folder in FOLDERS:
    folder = Path(folder)

    for file in folder.rglob("*"):
        if not file.is_file():
            continue
        if file == current_file:
            continue
        if file.suffix.lower() not in EXTENSIONS:
            continue

        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, start=1):
                    text = line.strip()

                    keyword_hit = [k for k in KEYWORDS if k.lower() in text.lower()]
                    if ENABLE_EMAIL_SEARCH:
                        email_hit = EMAIL_RE.search(text)

                    if keyword_hit:
                        print(f"\nkeyword_hit ({keyword_hit}):")
                        print(f"{file}: Line {lineno}")
                        print(text)

                    if ENABLE_EMAIL_SEARCH and email_hit:
                        print(f"\nemail_hit:")
                        print(f"{file}: Line {lineno}")
                        print(text)

        except Exception as e:
            print(f"Could not read {file}: {e}")
