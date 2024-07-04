
import sys
import os
import re
import subprocess
from urllib.parse import urlparse, parse_qs
from pytube import YouTube
import time

def download_video(url):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    output_file = audio_stream.download(output_path="music/")
    return output_file

def get_start_time(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    if 't' in query_params:
        print("TIME = ", int(query_params['t'][0]))
        return int(query_params['t'][0])
    return None

def crop_audio(input_file, start_time):
    output_file = f"music/cropped_{input_file.split('/')[-1]}"
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-ss", str(start_time),
        "-acodec", "copy",
        output_file
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)
    os.remove(input_file)
    os.rename(output_file, input_file)

def main():
    if len(sys.argv) != 2:
        print("Usage: python ytdl.py <youtube_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    
    try:
        output_file = download_video(url)
        print(f"Downloaded: {output_file}")
        
        start_time = get_start_time(url)
        if start_time is not None:
            crop_audio(output_file, start_time)
            print(f"Cropped audio to start at {start_time} seconds")
        
        print("Processing complete!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
