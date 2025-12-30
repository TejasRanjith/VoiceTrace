import os
import re
from datetime import datetime

# --- Configuration ---
# *** Please update these paths if they are incorrect ***
transcript_dir = r"C:\Users\TejasRanjith\Desktop\FINAL MAIN\LipReading\dataset\Transcripts"
del_path = r"C:\Users\TejasRanjith\Desktop\FINAL MAIN\LipReading\dataset"

# Log file paths
deleted_log_path = os.path.join(del_path, "deleted_files.txt")
cleaned_log_path = os.path.join(del_path, "cleaned_files.txt")

# --- Pattern Definitions ---
malayalam_pattern = re.compile(r'[\u0D00-\u0D7F]')  # Malayalam Unicode range
english_pattern = re.compile(r'[A-Za-z]')         # English letters
unknown_pattern = re.compile(r'[�]')          # Unknown characters
numeric_pattern = re.compile(r'[0-9]')             # Numbers (0-9)


def process_file(file_path):
    """
    Cleans or deletes a single transcript file.

    - Deletes if:
        - Empty
        - Contains English
        - Contains no valid Malayalam
    - Cleans if:
        - Contains unknown '' characters
        - Contains numbers
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return "Deleted", f"Failed to read file: {e}"

    if not lines:
        return "Deleted", "Empty file"

    valid_malayalam_found = False
    english_found = False
    new_lines = []
    changes_made = False  # Flag to track if we actually clean anything

    for line in lines:
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue  # Skip improperly formatted lines

        original_word = parts[2]

        # --- Language Check ---
        if english_pattern.search(original_word):
            english_found = True
        if malayalam_pattern.search(original_word):
            valid_malayalam_found = True

        # --- Cleaning ---
        # 1. Remove unknown characters
        clean_word = unknown_pattern.sub("", original_word)
        # 2. Remove numbers
        clean_word = numeric_pattern.sub("", clean_word)

        if original_word != clean_word:
            changes_made = True

        new_lines.append(f"{parts[0]} {parts[1]} {clean_word}\n")

    # --- Decision ---
    # Delete if it contains English or has no valid Malayalam text
    if english_found or not valid_malayalam_found:
        try:
            os.remove(file_path)
            reason = "Contains English" if english_found else "Not valid Malayalam"
            return "Deleted", reason
        except Exception as e:
            return "Deleted", f"Failed to remove file: {e}"

    # If no changes were needed, just report as unchanged
    if not changes_made:
        return "Unchanged", "File is already clean"

    # --- Overwrite file with cleaned content ---
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return "Cleaned", "Removed unknown chars or numbers"
    except Exception as e:
        return "Deleted", f"Failed to write cleaned file: {e}"


def main():
    """
    Iterates through the transcript directory, processes each file,
    and logs the results in append mode.
    """
    deleted_files_log = []
    cleaned_files_log = []
    unchanged_count = 0
    
    print(f"Starting cleanup in: {transcript_dir}\n")

    for file_name in os.listdir(transcript_dir):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(transcript_dir, file_name)

        try:
            status, reason = process_file(file_path)

            if status == "Deleted":
                deleted_files_log.append((file_name, reason))
            elif status == "Cleaned":
                cleaned_files_log.append((file_name, reason))
            elif status == "Unchanged":
                unchanged_count += 1

        except Exception as e:
            # Catch any unexpected errors during processing
            deleted_files_log.append((file_name, f"Unhandled Error: {e}"))

    # --- Logging ---
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write deleted file log (append mode)
    if deleted_files_log:
        with open(deleted_log_path, "a", encoding="utf-8") as log:
            log.write(f"\n--- Log Entry: {current_time} ---\n")
            for name, reason in deleted_files_log:
                log.write(f"{name}: {reason}\n")

    # Write cleaned file log (append mode)
    if cleaned_files_log:
        with open(cleaned_log_path, "a", encoding="utf-8") as log:
            log.write(f"\n--- Log Entry: {current_time} ---\n")
            for name, reason in cleaned_files_log:
                log.write(f"{name}: {reason}\n")

    # --- Summary ---
    print("\n--- Cleanup Summary ---")
    print(f"Files cleaned (newly): {len(cleaned_files_log)}")
    print(f"Files unchanged (already clean): {unchanged_count}")
    print(f"Files deleted or failed: {len(deleted_files_log)}")
    print("-------------------------")
    
    if cleaned_files_log:
        print(f"Cleaned file log appended to: {cleaned_log_path}")
    if deleted_files_log:
        print(f"Deleted file log appended to: {deleted_log_path}")
    
    print("\nCleanup complete.")


if __name__ == "__main__":
    main()

