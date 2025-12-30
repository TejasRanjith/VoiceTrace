import sys
from pathlib import Path
from collections import defaultdict

def count_words_in_directory(directory_path_str):
    """
    Reads all .txt files in a directory, counts words from a specific format
    (expecting lines with at least 3 parts, taking the last part as the word),
    and saves the counts to an output file, sorted by count in descending order.
    """
    
    directory_path = Path(directory_path_str)

    # 1. Check if the provided path is a valid directory
    if not directory_path.is_dir():
        print(f"Error: '{directory_path_str}' is not a valid directory.")
        print("Please run the script again with the correct path.")
        return

    # Use defaultdict to automatically handle new words with a count of 0
    word_counts = defaultdict(int)
    
    # 2. Find all .txt files in the directory
    txt_files = list(directory_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{directory_path_str}'.")
        return

    print(f"Found {len(txt_files)} .txt files. Processing...")
    
    processed_files = 0
    output_filename = "word_counts_output.txt" 

    for txt_file in txt_files:
        # Skip the output file itself if it exists in the same folder
        if txt_file.name == output_filename:
            continue

        try:
            # 3. Read each file with 'utf-8' encoding
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 4. Parse the line to get the word
                    parts = line.split()
                    
                    # Your specific logic: Check for at least 3 parts and take the last one
                    if len(parts) >= 3:
                        word = parts[-1]
                        # 5. Count the word
                        word_counts[word] += 1
            
            processed_files += 1

        except Exception as e:
            print(f"Error processing file {txt_file}: {e}")

    if processed_files == 0:
        print(f"No files were processed.")
        return

    if not word_counts:
        print("No words were found to count in the processed files.")
        return

    # 6. Save the results to the output file
    output_file_path = directory_path / output_filename
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            
            # --- SORTING LOGIC ---
            # Primary sort key: count (item[1]) in Descending order (negative value)
            # Secondary sort key: word (item[0]) in Ascending order
            # This ensures ties in counts are listed alphabetically.
            sorted_word_counts = sorted(
                word_counts.items(), 
                key=lambda item: (-item[1], item[0]) 
            )
            
            # Write the sorted list to the file
            for word, count in sorted_word_counts:
                out_f.write(f"{word} {count}\n")
        
        print(f"\nSuccessfully processed {processed_files} files.")
        print(f"Results sorted by count (descending) and saved to:\n{output_file_path}")

    except Exception as e:
        print(f"\nError writing output file: {e}")

# --- Main part of the script ---
if __name__ == "__main__":
    # Path provided in your code
    path_input = r"C:\Users\TejasRanjith\Desktop\FINAL MAIN\LipReading\dataset\Transcripts"
    
    count_words_in_directory(path_input)