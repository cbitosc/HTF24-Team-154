

from moviepy.editor import VideoFileClip
import numpy as np

def segment_video(video_path, duration=30):
    video = VideoFileClip(video_path)
    segments = []


    for start in np.arange(0, video.duration, duration):
        end = min(start + duration, video.duration)
        segments.append(video.subclip(start, end))

    return segments

!pip install SpeechRecognition
import speech_recognition as sr

def transcribe_audio(segment):
    recognizer = sr.Recognizer()
    audio = segment.audio.to_soundarray(fps=16000)

    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    return text

from transformers import pipeline

def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

def process_video_for_summary(video_path, duration=30):
    segments = segment_video(video_path, duration)
    summaries = []

    for segment in segments:
        # Transcribe segment
        transcript = transcribe_audio(segment)

        # Summarize segment
        summary = summarize_text(transcript)

        # Add to results
        summaries.append({
            "summary": summary,
            "video_snippet": segment
        })

    return summaries



!pip install SpeechRecognition
import speech_recognition as sr

def transcribe_audio(segment):
    recognizer = sr.Recognizer()
    # Check if the segment has audio
    if segment.audio is None:
        print("Warning: This video segment has no audio.")
        return ""  # Return an empty string to avoid errors in later stages

    audio = segment.audio.to_soundarray(fps=16000)

    with sr.AudioFile(audio) as source: # This line is likely incorrect
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    return text

!pip install SpeechRecognition moviepy
import speech_recognition as sr
import numpy as np
from moviepy.editor import AudioFileClip


def transcribe_audio(segment):
    recognizer = sr.Recognizer()
    # Check if the segment has audio
    if segment.audio is None:
        print("Warning: This video segment has no audio.")
        return ""  # Return an empty string to avoid errors in later stages

    # Get the audio as a NumPy array using moviepy's AudioFileClip
    audio_clip = AudioFileClip(segment.audio.filename)
    audio_array = audio_clip.to_soundarray(fps=16000)
    #audio = segment.audio.to_soundarray(fps=16000)

    # Convert the NumPy array to audio data that SpeechRecognition can handle
    #audio_data = sr.AudioData(audio_array.tobytes(), 16000, 2)
    with sr.AudioFile(audio_array) as source: # This line is likely incorrect
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    return text

import os
from moviepy.editor import VideoFileClip
import speech_recognition as sr

def extract_audio_from_segment(segment, output_filename="temp_audio.wav"):
    # Export audio from video segment as a WAV file
    segment.audio.write_audiofile(output_filename, codec='pcm_s16le')
    return output_filename

def transcribe_audio_from_file(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

def process_video_for_summary(video_path, duration=30):
    video = VideoFileClip(video_path)
    summaries = []

    # Segment the video and process each segment
    for start in range(0, int(video.duration), duration):
        segment = video.subclip(start, min(start + duration, video.duration))

        # Extract audio from segment and transcribe
        audio_file_path = extract_audio_from_segment(segment)
        transcript = transcribe_audio_from_file(audio_file_path)

        # Summarize the transcript
        summary = summarize_text(transcript)

        # Store summary and corresponding video snippet
        summaries.append({
            "summary": summary,
            "video_snippet": segment
        })

        # Clean up temporary audio file
        os.remove(audio_file_path)

    return summaries

import os
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from transformers import pipeline

# Initialize the summarization pipeline once, outside the function
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def extract_audio_from_segment(segment, output_filename="temp_audio.wav"):
    segment.audio.write_audiofile(output_filename, codec='pcm_s16le')
    return output_filename

def transcribe_audio_from_file(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            print(f"Could not understand audio in segment {audio_file_path}")
            text = None
        return text

def summarize_text(text):
    if text:
        return summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
    return None

def process_video_for_summary(video_path, duration=30):
    video = VideoFileClip(video_path)
    summaries = []

    for start in range(0, int(video.duration), duration):
        segment = video.subclip(start, min(start + duration, video.duration))

        # Extract and transcribe audio
        audio_file_path = extract_audio_from_segment(segment)
        transcript = transcribe_audio_from_file(audio_file_path)
        os.remove(audio_file_path)

        # Check if transcription was successful
        if transcript:
            summary = summarize_text(transcript)
            summaries.append({
                "summary": summary,
                "video_snippet": segment
            })
        else:
            print(f"Skipping segment from {start}s to {start + duration}s due to transcription error.")

    return summaries

video_path = "/content/Nastya and a collection of funny stories about dad and Nastya's friends.mp4"
results = process_video_for_summary(video_path, duration=30)

for idx, result in enumerate(results):
    if result["summary"]:
        print(f"Summary {idx + 1}: {result['summary']}")
        result["video_snippet"].write_videofile(f"snippet_{idx + 1}.mp4")

from moviepy.editor import VideoFileClip, concatenate_videoclips

# List of input video files to merge (replace with actual filenames)
video_files = ["/content/snippet_1.mp4", "/content/snippet_10.mp4", "/content/snippet_11.mp4"]

# Check if each video file exists
for file in video_files:
    if not os.path.exists(file):
        print(f"Error: File not found: {file}")
        # Handle the missing file, e.g., skip it, raise an exception, or exit
        # Here, we'll skip the missing file
        video_files.remove(file)
        print(f"Skipping file: {file}")
        continue  # Move to the next file in the list

# Load video clips for existing files
clips = [VideoFileClip(file) for file in video_files]

# Concatenate the clips
final_clip = concatenate_videoclips(clips, method="compose")

# Export the merged video as compressed MP4
output_file = "merged_output.mp4"
final_clip.write_videofile(
    output_file,
    codec="libx264",
    fps=24,
    preset="slow",
    bitrate="500k"
)

# Close all clips to release resources
for clip in clips:
    clip.close()
final_clip.close()

print(f"Merged video saved as {output_file}")

import pickle
from flask import Flask, request, jsonify

import pickle

# Assuming 'model' is your trained model
with open('hacktober.pkl', 'wb') as f:
    pickle.dump(summarizer, f)

!pip install flask flask-ngrok

from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app runs

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Mock prediction for example; replace with your model logic
    result = {"prediction": "This is a test prediction"}  # Replace with actual model output
    return jsonify(result)

if __name__ == "__main__":
    app.run()

!pip install flask-cors
from flask_cors import CORS

CORS(app)