import os
import torch
import cv2
import mediapipe as mp
import subprocess
from dotenv import load_dotenv
import time
from tqdm import tqdm
import multiprocessing

load_dotenv()

# --- Configuration ---
OUTPUT_CLIPS_DIR = "C:/Users/TejasRanjith/Desktop/FINAL MAIN/LipReading/dataset/Clipped_Videos/"
TEMP_AUDIO_DIR = "C:/Users/TejasRanjith/Desktop/FINAL MAIN/LipReading/dataset/temp_audio/"
VIDEO_DIRECTORY = r"C:\Users\TejasRanjith\Desktop\FINAL MAIN\LipReading\dataset\Youtube_Videos_02\\"

# --- NEW: Logging Configuration ---
LOG_CLIPPED_VIDEOS = True  # Set to True to save a log of all clipped filenames
LOG_FILENAME = "videos_clipped_log02.txt"  # The name of the log file

# --- Constants ---
MIN_CLIP_DURATION = 0.5
FACE_DETECTION_INTERVAL = 0.5  # Check for a face every 0.5 seconds
VAD_SAMPLING_RATE = 16000      # Silero VAD expects 16000Hz audio

# --- Globals for worker processes ---
# These will be initialized by init_worker and will be
# unique to each worker process.
g_silero_model = None
g_silero_utils = None
g_face_detector = None

def init_worker():
    """
    Initializer function for each worker process.
    Loads models into global variables for THIS process.
    """
    global g_silero_model, g_silero_utils, g_face_detector
    
    # Get the process ID for logging, so we know which worker is which
    pid = os.getpid()
    print(f"Initializing worker (PID: {pid})...")
    
    # Initialize Silero VAD
    try:
        # **CRUCIAL FIX 1: Set force_reload=False**
        # This uses the local cache instead of re-downloading
        # and triggering the rate limit.
        g_silero_model, g_silero_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False 
        )
    except Exception as e:
        print(f"PID {pid}: Failed to load Silero VAD model: {e}")
        return

    # Initialize Face Detector
    try:
        mp_face_detection = mp.solutions.face_detection
        g_face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
    except Exception as e:
        print(f"PID {pid}: Failed to load MediaPipe Face Detector: {e}")
    
    print(f"Worker (PID: {pid}) initialized successfully.")


# --- Helper Functions (unchanged) ---

def get_face_timestamps(video_path, face_detector):
    """
    Scans the entire video once to find all timestamps where faces are present.
    Returns a set of timestamps for fast lookups.
    """
    face_timestamps = set()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return face_timestamps

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Could not determine FPS for {video_path}. Assuming 30.")
        fps = 30 # Provide a default fallback
        
    frame_interval = int(fps * FACE_DETECTION_INTERVAL)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(frame_rgb)
                if results.detections:
                    current_time_sec = frame_count / fps
                    face_timestamps.add(round(current_time_sec, 2))
            except Exception as e:
                # This error can be spammy, so only enable if debugging a specific video
                # print(f"Error during face detection on frame {frame_count}: {e}")
                pass

        frame_count += 1
        
    cap.release()
    return face_timestamps

def clip_with_ffmpeg(input_video, output_video, start_sec, end_sec):
    """
    Uses a direct FFmpeg command with the 'ultrafast' preset for speed.
    """
    command = [
        "ffmpeg",
        "-i", input_video,
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-y",
        "-loglevel", "error",
        output_video
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg clipping failed for {output_video}. Error: {e}")

def extract_audio_with_ffmpeg(video_path, temp_audio_path):
    """Uses FFmpeg to extract audio into a WAV file at the required sample rate."""
    command = [
        "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", str(VAD_SAMPLING_RATE), "-ac", "1", "-y", "-loglevel", "error", temp_audio_path
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error! Could not extract audio from {video_path}.")
        raise

# --- Main processing logic for a single video (MODIFIED) ---

def process_video(video_path, silero_model, silero_utils, face_detector):
    """
    Processes a single video file using Silero VAD for voice detection.
    
    **MODIFIED:** Returns a list of saved clip filenames, not just the count.
    """
    # Check if models were loaded correctly in the worker
    if not all([silero_model, silero_utils, face_detector]):
         print(f"Skipping {os.path.basename(video_path)} because models are not loaded in this worker (PID: {os.getpid()}).")
         return [] # Return empty list

    base_filename = os.path.basename(video_path)
    
    # --- 1. Voice Activity Detection with Silero VAD ---
    temp_audio_path = os.path.join(TEMP_AUDIO_DIR, f"temp_{base_filename}.wav")
    speech_timestamps = []
    try:
        extract_audio_with_ffmpeg(video_path, temp_audio_path)
        
        # Unpack the Silero utility functions
        (get_speech_timestamps, _, read_audio, *_) = silero_utils

        # Read the audio file
        wav = read_audio(temp_audio_path, sampling_rate=VAD_SAMPLING_RATE)
        
        # Get speech segments from Silero VAD
        # The output is a list of dicts with 'start' and 'end' in samples
        speech_segments_samples = get_speech_timestamps(wav, silero_model, sampling_rate=VAD_SAMPLING_RATE)
        
        # Convert samples to seconds for the rest of the script
        for seg in speech_segments_samples:
            start_sec = seg['start'] / VAD_SAMPLING_RATE
            end_sec = seg['end'] / VAD_SAMPLING_RATE
            speech_timestamps.append((start_sec, end_sec))

    except Exception as e:
        print(f"Could not perform VAD on {base_filename}. Error: {e}")
        return [] # Return empty list
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except PermissionError:
                print(f"Warning: Could not remove temp file {temp_audio_path}. It might be in use.")
            
    if not speech_timestamps:
        # This is not an error, just common. No need to print.
        # print(f"No speech found in {base_filename}.")
        return [] # Return empty list

    # --- 2. Batched Face Detection ---
    face_timestamps = get_face_timestamps(video_path, face_detector)
    if not face_timestamps:
        # print(f"No faces found in {base_filename}.")
        return [] # Return empty list
        
    # --- 3. Find Overlapping Segments & Clip ---
    saved_clips_list = [] # MODIFIED: Was clip_count
    for i, (start_sec, end_sec) in enumerate(speech_timestamps):
        duration = end_sec - start_sec
        if duration < MIN_CLIP_DURATION:
            continue

        has_overlap = False
        for t in face_timestamps:
            if start_sec - FACE_DETECTION_INTERVAL <= t <= end_sec + FACE_DETECTION_INTERVAL:
                has_overlap = True
                break
        
        if has_overlap:
            output_filename = os.path.join(
                OUTPUT_CLIPS_DIR,
                f"{os.path.splitext(base_filename)[0]}_clip_{i+1:03d}.mp4" # Use index 'i' for a more unique ID
            )
            clip_with_ffmpeg(video_path, output_filename, start_sec, end_sec)
            saved_clips_list.append(output_filename) # MODIFIED: Add filename to list

    # Only print if we actually saved something, to reduce console spam
    if saved_clips_list:
        print(f"--- Processed {base_filename}. Saved {len(saved_clips_list)} clips. ---")
    
    return saved_clips_list # MODIFIED: Return the list of filenames

# --- MODIFIED: worker function ---
def worker(video_path):
    """
    A worker function that uses the *globally initialized* models
    to process a single video file.
    """
    # Models are now in g_silero_model, g_silero_utils, g_face_detector
    # We retrieve them from the global scope of this worker process
    global g_silero_model, g_silero_utils, g_face_detector
    
    clips_saved_list = [] # MODIFIED
    try:
        # Pass the globally loaded models to the processing function
        clips_saved_list = process_video(video_path, g_silero_model, g_silero_utils, g_face_detector) # MODIFIED
    except Exception as e:
        print(f"Unhandled error processing {video_path} in worker {os.getpid()}: {e}")
    
    # We don't .close() the models here, as they are shared
    # by this worker for all its tasks.
    
    return clips_saved_list # MODIFIED: Return the list

# --- Main execution block with Parallel Processing (MODIFIED) ---

if __name__ == "__main__":
    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    
    # Set the multiprocessing start method for compatibility (especially on Windows/macOS)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, which is fine
        pass

    video_files = [
        os.path.join(VIDEO_DIRECTORY, f) for f in os.listdir(VIDEO_DIRECTORY)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ]

    if not video_files:
        print(f"No video files found in '{VIDEO_DIRECTORY}'")
        exit()
        
    try:
        num_cores = os.cpu_count()
        num_workers = max(1, min(num_cores // 2, 4)) 
    except NotImplementedError:
        num_workers = 2
    
    print(f"\n🚀 Starting parallel processing with {num_workers} workers...")
    
    start_time = time.time()
    
    # **CRUCIAL FIX 3: Add the 'initializer' argument**
    # This tells the Pool to run our init_worker() function
    # once for each of the 4 worker processes it creates.
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:
        # pool.imap will now call the modified worker() function
        # 'results' will be a list of lists (e.g., [ ['clip1.mp4'], [], ['clip2.mp4', 'clip3.mp4'] ])
        results_list_of_lists = list(tqdm(pool.imap(worker, video_files), total=len(video_files), desc="Processing Videos"))

    # --- MODIFIED: Process results and write log ---

    # Flatten the list of lists into a single list of all clipped filenames
    all_clipped_filenames = [
        filename 
        for sublist in results_list_of_lists 
        for filename in sublist
    ]

    total_clips_saved = len(all_clipped_filenames)
    end_time = time.time()
    
    # --- NEW: Write the log file ---
    if LOG_CLIPPED_VIDEOS and all_clipped_filenames:
        log_file_path = os.path.join(OUTPUT_CLIPS_DIR, LOG_FILENAME)
        print(f"\n📝 Writing {total_clips_saved} clip names to {log_file_path}...")
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                for full_path in all_clipped_filenames:
                    # Write just the filename, not the full path, for a cleaner log
                    f.write(f"{os.path.basename(full_path)}\n")
            print(f"✅ Successfully saved log file.")
        except Exception as e:
            print(f"⚠️ Error writing log file: {e}")
    elif LOG_CLIPPED_VIDEOS:
        print("\n📝 No clips were saved, so no log file was created.")
    
    print("\n" + "="*50)
    print("✅ All Videos Processed!")
    print(f"Total clips saved: {total_clips_saved}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print("="*50)