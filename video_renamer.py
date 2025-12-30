import os
import re
import csv

# --- Folder containing your video files ---
folder = r"C:\Users\TejasRanjith\Desktop\FINAL MAIN\LipReading\dataset\Youtube_Videos_01\\"

# --- Log file path ---
log_file = os.path.join(folder, "rename_log.csv")

# --- Regex pattern ---
# Extracts "Name - EpisodeNumber" OR "Name EpisodeNumber"
# MODIFIED PATTERN:
pattern = re.compile(r"^(.+?)\s*[- ]\s*(\d+)", re.IGNORECASE)

# --- Create or overwrite the log file ---
with open(log_file, mode='w', newline='', encoding='utf-8') as csvfile:
    log_writer = csv.writer(csvfile)
    log_writer.writerow(["Old Filename", "New Filename", "Status"])

    for filename in os.listdir(folder):
        if not filename.lower().endswith(".mp4"):
            continue
        
        old_path = os.path.join(folder, filename)
        name_match = pattern.search(filename)

        if not name_match:
            print(f"⚠️ Skipped (pattern not found): {filename}")
            log_writer.writerow([filename, "", "Skipped (No Match)"])
            continue

        # Extract name parts
        person_name = name_match.group(1).strip()
        episode_num = name_match.group(2).strip()

        # Clean up and reformat
        person_name = re.sub(r"[｜|_]", "-", person_name) # Replaces special chars with hyphen
        person_name = re.sub(r"\s+", "-", person_name)   # Replaces spaces with hyphen

        new_name = f"{person_name}-{episode_num}.mp4"
        new_path = os.path.join(folder, new_name)

        # --- Safety: skip if exists ---
        if os.path.exists(new_path):
            print(f"⚠️ Skipped (already exists): {new_name}")
            log_writer.writerow([filename, new_name, "Skipped (Exists)"])
            continue

        # --- Perform rename ---
        try:
            os.rename(old_path, new_path)
            print(f"✅ Renamed: {filename}  →  {new_name}")
            log_writer.writerow([filename, new_name, "Renamed"])
        except Exception as e:
            print(f"❌ Error renaming {filename}: {e}")
            log_writer.writerow([filename, new_name, f"Error: {e}"])

print(f"\n📄 Rename log saved to: {log_file}")