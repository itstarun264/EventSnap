import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'eventsnap-secret-key-2024')
    
    # Check if running on Hugging Face Space with /data persistent storage
    if os.path.exists('/data'):
        STORAGE_DIR = '/data'
        DATABASE_PATH = os.path.join(STORAGE_DIR, 'eventsnap.db')
        UPLOAD_FOLDER = os.path.join(STORAGE_DIR, 'uploads')
    else:
        # Default local setup
        STORAGE_DIR = os.path.abspath(os.path.dirname(__file__))
        DATABASE_PATH = os.path.join(STORAGE_DIR, 'instance', 'eventsnap.db')
        UPLOAD_FOLDER = os.path.join(STORAGE_DIR, 'static', 'uploads')

    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload folders
    PHOTOS_FOLDER = os.path.join(UPLOAD_FOLDER, 'photos')
    POSTERS_FOLDER = os.path.join(UPLOAD_FOLDER, 'posters')
    SELFIES_FOLDER = os.path.join(UPLOAD_FOLDER, 'selfies')
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size