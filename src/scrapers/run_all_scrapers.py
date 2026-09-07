import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "yahoo.py",
    "cnbc.py",
    "benzinga.py",
    "marketwatch.py",
    "finviz.py",
    "reddit_scraper.py",
]

processes = []

print("[START] Running all scrapers...\n")

for script in SCRIPTS:
    full_path = os.path.join(BASE_DIR, script)
    print(f"[RUN] {script}")
    p = subprocess.Popen(["python", full_path])
    processes.append(p)

print("\n[INFO] All scrapers started \n")

for p in processes:
    p.wait()