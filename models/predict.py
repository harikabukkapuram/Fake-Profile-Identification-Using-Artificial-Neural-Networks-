# models/predict_cli.py
"""
Interactive CLI for FakeProfileDetection predictions.

Run from project root:
  python models/predict_cli.py

Default (press Enter): Test CSV mode (asks for id; Enter -> first row)
Type '1' then Enter: Manual mode (prompts for fields)
"""

import sys
from pathlib import Path
import json
import pandas as pd

# make sure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import Predictor
try:
    from models.predictor import Predictor
except Exception as e:
    print("ERROR: could not import models.predictor — ensure models/predictor.py exists and is valid.")
    raise

TEST_CSV = ROOT / "data" / "test" / "test.csv"

# colored output via colorama (optional)
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except Exception:
    COLORAMA_AVAILABLE = False
    # fallback ANSI
    class _C:
        RED = "\033[31m"
        GREEN = "\033[32m"
        RESET = "\033[0m"
    Fore = type("F", (), {"RED": _C.RED, "GREEN": _C.GREEN})
    Style = type("S", (), {"RESET_ALL": _C.RESET})

def input_def(prompt, default=""):
    v = input(prompt).strip()
    if v == "":
        return default
    return v

def pretty_print_prediction(out: dict):
    """
    Print prediction summary in color, with both human-friendly and raw metrics:
      - label_confidence: confidence in chosen label (percentage)
      - final_score: raw probability of 'fake'
      - certainty: distance from 0.5 (0..100)
    """
    label = out.get("label", "unknown").upper()
    final = out.get("final_score", None)
    label_conf = out.get("label_confidence", None)
    certainty = out.get("certainty", None)

    # format
    try:
        label_conf_str = f"{float(label_conf):.2f}%"
    except:
        label_conf_str = "N/A"
    try:
        certainty_str = f"{float(certainty):.2f}%"
    except:
        certainty_str = "N/A"
    try:
        final_str = f"{float(final):.6f}"
    except:
        final_str = str(final)

    # color decide
    if label == "GENUINE":
        color_start = Fore.GREEN if COLORAMA_AVAILABLE else Fore.GREEN
    elif label == "FAKE":
        color_start = Fore.RED if COLORAMA_AVAILABLE else Fore.RED
    else:
        color_start = ""
    reset = Style.RESET_ALL if COLORAMA_AVAILABLE else Style.RESET_ALL

    print()
    print(f"{color_start}===== FINAL PREDICTION ====={reset}")
    print(f"{color_start}Prediction : {label}{reset}")
    print(f"{color_start}Confidence : {label_conf_str}{reset}    (raw final_score = {final_str})")
    print(f"{color_start}Certainty  : {certainty_str}{reset}")
    print(f"{color_start}============================{reset}\n")

    print("Full details:")
    print(json.dumps(out, indent=2))

def manual_mode(P: Predictor):
    print("\n--- MANUAL MODE --- (press Enter for defaults)\n")
    prof = {}
    prof["screen_name"] = input_def("screen_name (default 'manual_user'): ", "manual_user")
    prof["description"] = input_def("description: ", "")
    try:
        prof["followers_count"] = int(input_def("followers_count (0): ", "0"))
    except:
        prof["followers_count"] = 0
    try:
        prof["friends_count"] = int(input_def("friends_count (0): ", "0"))
    except:
        prof["friends_count"] = 0
    try:
        prof["statuses_count"] = int(input_def("statuses_count (0): ", "0"))
    except:
        prof["statuses_count"] = 0
    try:
        prof["favourites_count"] = int(input_def("favourites_count (0): ", "0"))
    except:
        prof["favourites_count"] = 0
    try:
        prof["listed_count"] = int(input_def("listed_count (0): ", "0"))
    except:
        prof["listed_count"] = 0
    try:
        prof["utc_offset"] = int(input_def("utc_offset (0): ", "0"))
    except:
        prof["utc_offset"] = 0
    prof["profile_image_url"] = input_def("profile_image_url (leave blank): ", "")
    lang = input_def("lang_label (numeric) or leave blank: ", "")
    try:
        prof["lang_label"] = int(lang) if lang != "" else 0
    except:
        prof["lang_label"] = 0

    print("\nRunning prediction...")
    out = P.predict_manual(prof, screen_name=prof.get("screen_name"))
    pretty_print_prediction(out)

def test_csv_mode(P: Predictor):
    if not TEST_CSV.exists():
        print(f"ERROR: {TEST_CSV} not found. Create the file and try again.")
        return
    df = pd.read_csv(TEST_CSV, dtype=str).fillna("")
    print(f"\nLoaded test.csv with {len(df)} rows.")
    id_input = input("Enter id from test.csv (press Enter -> use first row): ").strip()
    if id_input == "":
        row = df.iloc[0]
        print(f"Using first row id = {row.get('id','<no-id>')}")
    else:
        if "id" not in df.columns:
            print("ERROR: test.csv missing 'id' column.")
            return
        matched = df[df["id"].astype(str) == id_input]
        if matched.empty:
            matched = df[df["id"].astype(str).str.lower() == id_input.lower()]
        if matched.empty:
            print(f"No row with id='{id_input}'. Aborting.")
            return
        row = matched.iloc[0]

    profile = {
        "screen_name": row.get("screen_name", "") or row.get("profile_name", "") or f"test_{row.get('id','')}",
        "description": row.get("description", "") or "",
        "followers_count": int(float(row.get("followers_count", 0) or 0)),
        "friends_count": int(float(row.get("friends_count", 0) or 0)),
        "statuses_count": int(float(row.get("statuses_count", 0) or 0)),
        "favourites_count": int(float(row.get("favourites_count", 0) or 0)),
        "listed_count": int(float(row.get("listed_count", 0) or 0)),
        "utc_offset": int(float(row.get("utc_offset", 0) or 0)),
        "profile_image_url": row.get("profile_image_url", "") or "",
        "lang_label": int(float(row.get("lang_label", 0) or 0)) if row.get("lang_label", "") != "" else 0,
    }

    print("\nRunning prediction...")
    out = P.predict_manual(profile, screen_name=profile.get("screen_name"))
    pretty_print_prediction(out)

def main():
    print("=== FakeProfileDetection CLI ===")
    print("Press Enter for TEST CSV mode (default), or type '1' then Enter for MANUAL mode.")
    choice = input("Choice (Enter=default / '1'=manual): ").strip()
    P = Predictor()
    try:
        if choice == "1":
            manual_mode(P)
        else:
            test_csv_mode(P)
    except KeyboardInterrupt:
        print("\nAborted.")

if __name__ == "__main__":
    main()
