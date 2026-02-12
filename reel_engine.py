import os
import cv2
import numpy as np
# New MoviePy v2.x imports
from moviepy import ImageClip, concatenate_videoclips, AudioFileClip
from moviepy.video.fx import Resize # For resizing in v2.x

from models import Photo, Event
from config import Config

def score_photo(photo_path):
    """
    AI Scoring Logic:
    1. Sharpness (Laplacian Variance)
    2. Brightness (Mean Pixel Value)
    3. Faces (Haar Cascades)
    """
    score = 0
    try:
        img = cv2.imread(photo_path)
        if img is None: return 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness Test (+2)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance > 100: score += 2

        # 2. Lighting Test (+2)
        brightness = np.mean(gray)
        if 100 < brightness < 200: score += 2

        # 3. Face Detection (+3 per face, max +5)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            score += 3
        if len(faces) > 1:
            score += 2 # Group photo bonus
            
    except Exception as e:
        print(f"Error scoring {photo_path}: {e}")
    
    return score

def generate_reel(event_id, db):
    """
    Main Engine to create the .mp4 file using MoviePy 2.x
    """
    event = db.session.get(Event, event_id)
    photos = Photo.query.filter_by(event_id=event_id).all()
    
    if len(photos) < 3:
        return None, "Need at least 3 photos to make a reel."

    # 1. Score all photos
    scored_list = []
    for p in photos:
        path = os.path.join(Config.PHOTOS_FOLDER, p.filename)
        if os.path.exists(path):
            score = score_photo(path)
            scored_list.append((score, path))

    # 2. Sort by score and pick top 15
    scored_list.sort(key=lambda x: x[0], reverse=True)
    top_photos = scored_list[:15]

    # 3. Create Video Clips
    clips = []
    for score, path in top_photos:
        # Create clip and set duration
        clip = ImageClip(path).with_duration(2)
        
        # Apply Resize Effect (Standard for v2.x)
        # Resizing to 1080p height for vertical/reel format
        clip = clip.with_effects([Resize(height=1080)])
        
        clips.append(clip)

    # 4. Join clips
    # Using 'compose' ensures different image sizes are handled gracefully
    video = concatenate_videoclips(clips, method="compose")

    # 5. Add Music
    music_path = os.path.join('static', 'music', 'background.mp3')
    if os.path.exists(music_path):
        try:
            audio = AudioFileClip(music_path).with_duration(video.duration)
            video = video.with_audio(audio)
        except Exception as e:
            print(f"Music error: {e}")

    # 6. Export
    output_filename = f"reel_event_{event_id}.mp4"
    output_path = os.path.join(Config.UPLOAD_FOLDER, output_filename)
    
    # Export with ultra-fast settings for startup demo
    video.write_videofile(
        output_path, 
        fps=24, 
        codec="libx264", 
        audio_codec="aac", 
        threads=4
    )
    
    return output_filename, None