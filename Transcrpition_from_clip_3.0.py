import whisperx
import torch
import os
import subprocess
import json
import math
import glob
from dotenv import load_dotenv
import multiprocessing  # For parallel CPU processing

load_dotenv()

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

input_video_dir = "C:/Users/TejasRanjith/Desktop/FINAL MAIN/LipReading/dataset/Clipped_Videoss"
output_transcript_dir = "C:/Users/TejasRanjith/Desktop/FINAL MAIN/LipReading/dataset/Transcripts"

batch_size = 16
compute_type = "float16"

# Determine number of CPU workers
try:
    NUM_CPU_WORKERS = os.cpu_count() - 1
except Exception:
    NUM_CPU_WORKERS = 3  # Fallback

YOUR_HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# --- Helper Functions ---
def convert_video_to_audio(video_input_path, audio_output_path):
    command = [
        "ffmpeg", "-i", video_input_path, "-y", "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_output_path
    except Exception:
        return None

def get_video_fps(video_path):
    if not os.path.exists(video_path):
        return None
    try:
        command = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "json", video_path
        ]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = json.loads(result.stdout)
        frame_rate_str = data["streams"][0]["r_frame_rate"]
        num, den = map(int, frame_rate_str.split('/'))
        return num / den
    except Exception:
        return None

def save_transcript_with_frames(transcription_result, fps, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in transcription_result.get("segments", []):
            for word_info in segment.get("words", []):
                if 'start' in word_info and 'end' in word_info:
                    start_time, end_time, word = word_info['start'], word_info['end'], word_info['word']
                    start_frame = int(start_time * fps)
                    end_frame = int(math.ceil(end_time * fps))
                    f.write(f"{start_frame} {end_frame} {word.strip()}\n")

# --- CPU Worker ---
def cpu_prepare_audio(args):
    video_path, temp_audio_path, transcript_path = args
    print(f"[CPU Worker] Processing: {os.path.basename(video_path)}")

    fps = get_video_fps(video_path)
    if not fps:
        print(f"[CPU Worker] Failed to get FPS for {os.path.basename(video_path)}")
        return None

    if not convert_video_to_audio(video_path, temp_audio_path):
        print(f"[CPU Worker] Failed to convert {os.path.basename(video_path)} to audio.")
        return None

    print(f"[CPU Worker] Finished preparing audio for: {os.path.basename(video_path)}")
    return (temp_audio_path, transcript_path, fps)

# --- GPU Transcription ---
def gpu_transcribe_and_align(audio_path, transcript_path, fps, model, align_model, align_metadata):
    print(f"[GPU Consumer] Transcribing: {os.path.basename(audio_path)}")
    try:
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size)
        result = whisperx.align(result["segments"], align_model, align_metadata, audio, device, return_char_alignments=False)
        save_transcript_with_frames(result, fps, transcript_path)
        print(f"[GPU Consumer] ✅ Transcript saved for: {os.path.basename(transcript_path)}")
    except Exception as e:
        print(f"[GPU Consumer] ❌ Error processing {os.path.basename(audio_path)}: {e}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# --- Main ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.makedirs(output_transcript_dir, exist_ok=True)

    print("--- Loading models into GPU memory... ---")
    transcribe_model = whisperx.load_model("kurianbenoy/vegam-whisper-medium-ml", device, compute_type=compute_type)
    align_model, align_metadata = whisperx.load_align_model(language_code="ml", device=device)
    print("--- Models loaded successfully ---")

    # Step 1: Find all videos
    video_files = []
    for ext in ["*.mp4", "*.mov", "*.avi", "*.mkv"]:
        video_files.extend(glob.glob(os.path.join(input_video_dir, ext)))

    if not video_files:
        print(f"No video files found in '{input_video_dir}'. Exiting.")
        exit()

    # Step 2: Filter out already processed videos
    tasks = []
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        transcript_path = os.path.join(output_transcript_dir, f"{base_name}.txt")
        temp_audio_path = os.path.join(output_transcript_dir, f"{base_name}_temp_audio.wav")

        # ✅ Skip clips that already have transcripts
        if os.path.exists(transcript_path) and os.path.getsize(transcript_path) > 0:
            print(f"[SKIP] Transcript already exists for: {base_name}")
            continue

        tasks.append((video_path, temp_audio_path, transcript_path))

    if not tasks:
        print("\n✅ All clips already have transcripts. Nothing to process.")
        exit()

    print(f"\nFound {len(tasks)} clips to process. Using {NUM_CPU_WORKERS} CPU workers.")
    print("-" * 60)

    # Step 3: Parallel pipeline
    with multiprocessing.Pool(processes=NUM_CPU_WORKERS) as pool:
        for result in pool.imap_unordered(cpu_prepare_audio, tasks):
            if result:
                temp_audio_path, transcript_path, fps = result
                gpu_transcribe_and_align(
                    temp_audio_path, transcript_path, fps,
                    transcribe_model, align_model, align_metadata
                )

    print("-" * 60)
    print("\n🎉 All remaining clips have been processed successfully.")
