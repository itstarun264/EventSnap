import os

class Config:
    SECRET_KEY = 'eventsnap-secret-key-2024'
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload folders
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    PHOTOS_FOLDER = os.path.join(UPLOAD_FOLDER, 'photos')
    POSTERS_FOLDER = os.path.join(UPLOAD_FOLDER, 'posters')
    SELFIES_FOLDER = os.path.join(UPLOAD_FOLDER, 'selfies')
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size