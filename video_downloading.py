import os
import sys # Import sys to handle exit
import yt_dlp


LINKS_FILE = "playlist_links.txt"

OUTPUT_FOLDER = "downloads"

# --- Main Script ---

def download_videos_from_file(file_path, output_dir):
    """
    Reads a file line-by-line and downloads the YouTube video from each URL.

    Args:
        file_path (str): The path to the text file containing YouTube URLs.
        output_dir (str): The directory where videos will be saved.
    """
    # Create the output directory if it doesn't already exist
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # yt-dlp options
    ydl_opts = {
    'format': 'bestvideo[height=1080]+bestaudio/best',
    'outtmpl': '%(title)s.%(ext)s',
    'paths': {'home': output_dir},
    'merge_output_format': 'mp4',
    'retries': 5,
    'fragment_retries': 10,
    'skip_unavailable_fragments': True,
    'geo_bypass': True,
    'http_headers': {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/130.0.6723.69 Safari/537.36'
        ),
    },
    'extractor_args': {'youtube': {'player_client': ['android']}},
    'compat_opts': ['no-youtube-unavailable-videos', 'no-youtube-dash-manifest'],
    'postprocessors': [{
        'key': 'FFmpegVideoConvertor',
        'preferedformat': 'mp4',
    }],
}

    # Read the links from the file
    try:
        with open(file_path, 'r') as f:
            urls = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please create this file and add your YouTube links to it.")
        return

    if not urls:
        print(f"The file '{file_path}' is empty. No videos to download.")
        return

    print(f"Found {len(urls)} links in '{file_path}'. Starting download process...")
    print("Press Ctrl+C at any time to stop safely.")

    # --- START OF MODIFICATION ---
    try:
        # Loop through each URL and download the video
        limit = 0
        for i, url in enumerate(urls):
            # Remove any leading/trailing whitespace (like newlines)
            clean_url = url.strip()

            if not clean_url:
                # Skip empty lines
                continue

            print(f"\n--- Downloading video {i+1} of {len(urls)} ---")
            print(f"URL: {clean_url}")

            try:
                # The main download command
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([clean_url])
                print(f"Successfully downloaded video from: {clean_url}")
                limit += 1
            except Exception as e:
                # This inner try/except handles errors for a single video
                print(f"Error downloading {clean_url}. Reason: {e}")
                print("Skipping to the next video.")

            
            if limit == 25:
                break 

    except KeyboardInterrupt:
        # This outer try/except handles the user pressing Ctrl+C
        print("\n\nDownload process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    # --- END OF MODIFICATION ---

    print("\n--- All downloads complete! ---")


if __name__ == "__main__":
    # Check for yt-dlp installation before running
    try:
        import yt_dlp
    except ImportError:
        print("Error: 'yt-dlp' is not installed.")
        print("Please install it by running: pip install yt-dlp")
    else:
        download_videos_from_file(LINKS_FILE, OUTPUT_FOLDER)