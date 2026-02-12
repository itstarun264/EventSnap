import os
import json
import secrets
import zipfile
import base64
from io import BytesIO
from datetime import datetime
from functools import wraps
from flask_socketio import SocketIO, emit, join_room, leave_room

from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# --- AI & ML INTEGRATION ---
# 1. Image Classification (CLIP)
try:
    from ml_classifier import analyze_image
    ML_CLASSIFIER_AVAILABLE = True
except ImportError:
    ML_CLASSIFIER_AVAILABLE = False
    print("⚠️ ml_classifier.py not found. AI Auto-tagging will be disabled.")

# 2. Face Recognition (DeepFace or FaceRecognition)
FACE_RECOGNITION_AVAILABLE = False
DEEPFACE_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ DeepFace is available for AI face matching!")
except ImportError:
    try:
        import face_recognition
        FACE_RECOGNITION_AVAILABLE = True
        print("✅ face_recognition library is available!")
    except ImportError:
        print("⚠️ No face recognition library available. 'Find My Photos' will be disabled.")

from config import Config
from models import db, User, Event, Photo, VolunteerAssignment, EventAccess

# --- APP INITIALIZATION ---
app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure all upload directories exist
for folder in [Config.UPLOAD_FOLDER, Config.PHOTOS_FOLDER, Config.POSTERS_FOLDER, Config.SELFIES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- DECORATORS & HELPERS ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if current_user.role not in roles:
                flash('Access denied. Insufficient permissions.', 'error')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- TEST ROUTES (AUDIO ISOLATION) ---
@app.route('/test/audio/organizer')
def test_audio_organizer():
    return render_template('test/audio_organizer.html')

@app.route('/test/audio/viewer')
def test_audio_viewer():
    return render_template('test/audio_viewer.html')

def save_base64_image(base64_data, folder, prefix="camera"):
    """Saves base64 data (from webcam) as a JPG file"""
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        image_data = base64.b64decode(base64_data)
        filename = f"{prefix}_{secrets.token_hex(8)}.jpg"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        return filename, filepath
    except Exception as e:
        print(f"Error saving base64 image: {e}")
        return None, None

# --- AI CORE FUNCTIONS ---

def find_matching_photos(selfie_path, event_id):
    """
    Scans an event's gallery to find photos matching the uploaded selfie.
    Uses DeepFace (preferred) or face_recognition.
    """
    matched_photos = []
    photos = Photo.query.filter_by(event_id=event_id).all()
    
    if not os.path.exists(selfie_path):
        return []

    for photo in photos:
        photo_path = os.path.join(Config.PHOTOS_FOLDER, photo.filename)
        if not os.path.exists(photo_path):
            continue
        
        try:
            if DEEPFACE_AVAILABLE:
                # DeepFace verification (High Accuracy)
                result = DeepFace.verify(
                    img1_path=selfie_path, 
                    img2_path=photo_path, 
                    model_name='VGG-Face', 
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                if result['verified']:
                    matched_photos.append(photo)
            
            elif FACE_RECOGNITION_AVAILABLE:
                # Traditional face_recognition (Fast)
                import face_recognition
                known_img = face_recognition.load_image_file(selfie_path)
                unknown_img = face_recognition.load_image_file(photo_path)
                
                known_encs = face_recognition.face_encodings(known_img)
                unknown_encs = face_recognition.face_encodings(unknown_img)
                
                if known_encs and unknown_encs:
                    # Compare first face found in selfie with all faces in event photo
                    results = face_recognition.compare_faces(unknown_encs, known_encs[0], tolerance=0.6)
                    if True in results:
                        matched_photos.append(photo)
        except Exception as e:
            print(f"Matching error for photo {photo.id}: {e}")
            continue
            
    return matched_photos

# ==============================================================================
#                                PUBLIC ROUTES
# ==============================================================================

@app.route('/')
def index():
    # Fetch both Approved AND Completed events for the landing page
    events = Event.query.filter(Event.status.in_(['approved', 'completed'])).filter_by(event_type='public').order_by(Event.event_date.desc()).all()
    return render_template('index.html', events=events)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email').lower().strip()
        password = request.form.get('password')
        
        # 1. Try to find user in database
        user = User.query.filter_by(email=email).first()
        
        if user:
            if user.check_password(password):
                login_user(user)
                flash(f'Welcome back, {user.name}!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid password.', 'error')
        else:
            # 2. SMART LOGIN: Auto-detect @college.com students
            if email.endswith('@college.com'):
                if password.isdigit():
                    # Auto-create Student/Viewer account
                    name_part = email.split('@')[0].replace('.', ' ').title()
                    new_viewer = User(
                        email=email,
                        name=name_part,
                        role='viewer'
                    )
                    new_viewer.set_password(password)
                    db.session.add(new_viewer)
                    db.session.commit()
                    
                    login_user(new_viewer)
                    flash('Welcome! Student account created automatically.', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Students must use their numeric ID as the password.', 'error')
            else:
                flash('Account not found. Please register or use your college email.', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email').lower().strip()
        name = request.form.get('name')
        password = request.form.get('password')
        role = request.form.get('role', 'organizer') # Default to organizer
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('register'))
        
        user = User(email=email, name=name, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please sign in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out safely.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.role == 'organizer':
        return redirect(url_for('organizer_dashboard'))
    elif current_user.role == 'volunteer':
        return redirect(url_for('volunteer_dashboard'))
    else:
        return redirect(url_for('viewer_dashboard'))

# ==============================================================================
#                                ADMIN ROUTES
# ==============================================================================

@app.route('/admin/dashboard')
@login_required
@role_required('admin')
def admin_dashboard():
    # Show organizers grouped with their events
    organizers = User.query.filter_by(role='organizer').all()
    stats = {
        'total_events': Event.query.count(),
        'pending': Event.query.filter_by(status='pending').count(),
        'organizers': len(organizers),
        'total_photos': Photo.query.count()
    }
    return render_template('admin/dashboard.html', organizers=organizers, stats=stats)

@app.route('/admin/event/<int:event_id>/approve', methods=['POST'])
@login_required
@role_required('admin')
def approve_event(event_id):
    event = Event.query.get_or_404(event_id)
    event.status = 'approved'
    db.session.commit()
    flash(f'Event "{event.name}" is now live!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/event/<int:event_id>/reject', methods=['POST'])
@login_required
@role_required('admin')
def reject_event(event_id):
    event = Event.query.get_or_404(event_id)
    event.status = 'rejected'
    db.session.commit()
    flash(f'Event "{event.name}" was rejected.', 'warning')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/event/<int:event_id>/complete', methods=['POST'])
@login_required
@role_required('admin')
def complete_event(event_id):
    event = Event.query.get_or_404(event_id)
    event.status = 'completed'
    db.session.commit()
    flash(f'Event "{event.name}" marked as completed.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/users')
@login_required
@role_required('admin')
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users.html', users=users, current_user=current_user)

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
@role_required('admin')
def delete_user(user_id):
    if user_id == current_user.id:
        flash('You cannot delete yourself.', 'error')
        return redirect(url_for('admin_users'))
    
    user = User.query.get_or_404(user_id)
    
    # Optional: Delete associated files (photos, etc.) if needed via cascade or manual cleanup
    # For now, relying on DB cascade if set up, or just deleting the user record.
    # Note: Models.py shows cascade on Event-Photos, but User-Events/Photos might not auto-delete files from disk.
    # Ideally should clean up files, but for MVP just DB delete.
    
    name = user.name
    db.session.delete(user)
    db.session.commit()
    flash(f'User "{name}" has been deleted.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/user/<int:user_id>/role', methods=['POST'])
@login_required
@role_required('admin')
def update_user_role(user_id):
    if user_id == current_user.id:
        flash('You cannot change your own role.', 'error')
        return redirect(url_for('admin_users'))
        
    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role')
    
    if new_role not in ['admin', 'organizer', 'volunteer', 'viewer']:
        flash('Invalid role selected.', 'error')
        return redirect(url_for('admin_users'))
        
    user.role = new_role
    db.session.commit()
    flash(f'User "{user.name}" is now a {new_role.capitalize()}.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/event/<int:event_id>/force-delete', methods=['POST'])
@login_required
@role_required('admin')
def force_delete_event(event_id):
    event = Event.query.get_or_404(event_id)
    
    # 1. Cleaner File Deletion
    # Delete poster
    if event.poster_filename:
        try:
            os.remove(os.path.join(Config.POSTERS_FOLDER, event.poster_filename))
        except: pass
        
    # Delete all photos from disk
    for photo in event.photos:
        try:
            os.remove(os.path.join(Config.PHOTOS_FOLDER, photo.filename))
        except: pass
        
    db.session.delete(event)
    db.session.commit()
    flash(f'Event "{event.name}" and all associated data permanently deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/event/<int:event_id>/clear-photos', methods=['POST'])
@login_required
@role_required('admin')
def clear_photos(event_id):
    event = Event.query.get_or_404(event_id)
    # Physically delete files
    for photo in event.photos:
        try:
            os.remove(os.path.join(Config.PHOTOS_FOLDER, photo.filename))
        except:
            pass
    # Remove DB records
    Photo.query.filter_by(event_id=event_id).delete()
    db.session.commit()
    flash('Gallery cleared to save storage.', 'success')
    return redirect(url_for('admin_dashboard'))

# ==============================================================================
#                              ORGANIZER ROUTES
# ==============================================================================

@app.route('/organizer/dashboard')
@login_required
@role_required('organizer')
def organizer_dashboard():
    events = Event.query.filter_by(organizer_id=current_user.id).order_by(Event.created_at.desc()).all()
    return render_template('organizer/dashboard.html', events=events)

@app.route('/organizer/event/create', methods=['GET', 'POST'])
@login_required
@role_required('organizer')
def create_event():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        venue = request.form.get('venue')
        date_str = request.form.get('event_date')
        event_type = request.form.get('event_type')
        
        event = Event(
            name=name,
            description=description,
            venue=venue,
            event_date=datetime.strptime(date_str, '%Y-%m-%dT%H:%M'),
            event_type=event_type,
            organizer_id=current_user.id
        )
        
        if event_type == 'private':
            event.generate_access_code()
            
        poster = request.files.get('poster')
        if poster and allowed_file(poster.filename):
            fn = secure_filename(f"{secrets.token_hex(4)}_{poster.filename}")
            poster.save(os.path.join(Config.POSTERS_FOLDER, fn))
            event.poster_filename = fn
            
        db.session.add(event)
        db.session.commit()
        flash('Event submitted for admin approval.', 'success')
        return redirect(url_for('organizer_dashboard'))
    
    return render_template('organizer/create_event.html')



from datetime import datetime

@app.route('/organizer/event/update/<int:event_id>', methods=['POST'])
@login_required
# It is good practice to add your role decorator here too, though your manual check handles it
# @role_required('organizer') 
def organizer_update_event(event_id):
    event = Event.query.get_or_404(event_id)
    
    # Security check: ensure current user owns this event
    if event.organizer_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('organizer_dashboard'))

    # Update fields
    event.name = request.form.get('name')
    event.venue = request.form.get('venue')
    event.description = request.form.get('description')
    
    # Handle Date conversion
    date_str = request.form.get('event_date')
    
    # Check if date_str is not empty before converting
    if date_str:
        try:
            # Note: This sets time to 00:00:00. If you need time, 
            # you must update the HTML input to type="datetime-local" 
            # and format here to '%Y-%m-%dT%H:%M'
            event.event_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            flash('Invalid date format', 'danger')
            # FIX 1: Changed 'organizer_manage_event' to 'manage_event'
            return redirect(url_for('manage_event', event_id=event.id))

    db.session.commit()
    flash('Event details updated successfully!', 'success')
    
    # FIX 2: Changed 'organizer_manage_event' to 'manage_event'
    # This must match the name of the function: def manage_event(event_id):
    return redirect(url_for('manage_event', event_id=event.id))

@app.route('/organizer/event/<int:event_id>/manage')
@login_required
@role_required('organizer')
def manage_event(event_id):
    event = Event.query.get_or_404(event_id)
    if event.organizer_id != current_user.id:
        return redirect(url_for('organizer_dashboard'))
    
    volunteers = VolunteerAssignment.query.filter_by(event_id=event_id).all()
    photos = Photo.query.filter_by(event_id=event_id).all()
    return render_template('organizer/manage_event.html', event=event, volunteers=volunteers, photos=photos)

@app.route('/organizer/event/<int:event_id>/upload-photos', methods=['POST'])
@login_required
@role_required('organizer')
def organizer_upload_photos(event_id):
    event = Event.query.get_or_404(event_id)
    if event.status != 'approved':
        flash('You can only upload photos after admin approval.', 'error')
        return redirect(url_for('manage_event', event_id=event_id))
    
    files = request.files.getlist('photos')
    for file in files:
        if file and allowed_file(file.filename):
            fn = secure_filename(f"{secrets.token_hex(8)}_{file.filename}")
            filepath = os.path.join(Config.PHOTOS_FOLDER, fn)
            file.save(filepath)
            
            # --- AI ML IMAGE ANALYSIS (Auto Tagging) ---
            tags = "event"
            if ML_CLASSIFIER_AVAILABLE:
                tags = analyze_image(filepath)
            
            new_photo = Photo(
                filename=fn,
                original_filename=file.filename,
                tags=tags,
                event_id=event_id,
                uploaded_by=current_user.id
            )
            db.session.add(new_photo)
            
    db.session.commit()
    flash('Gallery updated and analyzed by AI.', 'success')
    return redirect(url_for('manage_event', event_id=event_id))

@app.route('/organizer/event/<int:event_id>/add-volunteer', methods=['GET', 'POST'])
@login_required
@role_required('organizer')
def add_volunteer(event_id):
    event = Event.query.get_or_404(event_id)
    if request.method == 'POST':
        email = request.form.get('email').strip().lower()
        name = request.form.get('name')
        password = request.form.get('password')
        
        # Check if user exists
        v_user = User.query.filter_by(email=email).first()
        if not v_user:
            v_user = User(email=email, name=name, role='volunteer')
            v_user.set_password(password)
            db.session.add(v_user)
            db.session.commit()
        
        # Assign to event
        if not VolunteerAssignment.query.filter_by(event_id=event_id, volunteer_id=v_user.id).first():
            assignment = VolunteerAssignment(event_id=event_id, volunteer_id=v_user.id)
            db.session.add(assignment)
            db.session.commit()
            flash('Volunteer assigned successfully.', 'success')
        else:
            flash('Already assigned.', 'warning')
            
        return redirect(url_for('manage_event', event_id=event_id))
    
    return render_template('organizer/add_volunteer.html', event=event)

@app.route('/organizer/event/<int:event_id>/remove-volunteer/<int:assignment_id>', methods=['POST'])
@login_required
@role_required('organizer')
def remove_volunteer(event_id, assignment_id):
    assign = VolunteerAssignment.query.get_or_404(assignment_id)
    db.session.delete(assign)
    db.session.commit()
    flash('Volunteer removed.', 'success')
    return redirect(url_for('manage_event', event_id=event_id))

@app.route('/organizer/event/<int:event_id>/delete', methods=['POST'])
@login_required
@role_required('organizer')
def organizer_delete_event(event_id):
    event = Event.query.get_or_404(event_id)
    
    # Security Check
    if event.organizer_id != current_user.id:
        flash('Unauthorized deletion attempt.', 'error')
        return redirect(url_for('organizer_dashboard'))
    
    # Delete associated files
    if event.poster_filename:
        try: os.remove(os.path.join(Config.POSTERS_FOLDER, event.poster_filename))
        except: pass
        
    for photo in event.photos:
        try: os.remove(os.path.join(Config.PHOTOS_FOLDER, photo.filename))
        except: pass
        
    db.session.delete(event)
    db.session.commit()
    flash(f'Event "{event.name}" has been deleted.', 'success')
    return redirect(url_for('organizer_dashboard'))

# ==============================================================================
#                              VOLUNTEER ROUTES
# ==============================================================================

@app.route('/volunteer/dashboard')
@login_required
@role_required('volunteer')
def volunteer_dashboard():
    assignments = VolunteerAssignment.query.filter_by(volunteer_id=current_user.id).all()
    # Only show events that are currently approved/live
    events = [a.event for a in assignments if a.event.status == 'approved']
    return render_template('volunteer/dashboard.html', events=events)

@app.route('/volunteer/event/<int:event_id>/upload', methods=['POST'])
@login_required
@role_required('volunteer')
def volunteer_upload_photos(event_id):
    files = request.files.getlist('photos')
    for file in files:
        if file and allowed_file(file.filename):
            fn = secure_filename(f"{secrets.token_hex(8)}_{file.filename}")
            path = os.path.join(Config.PHOTOS_FOLDER, fn)
            file.save(path)
            
            # AI ML Tagging
            tags = analyze_image(path) if ML_CLASSIFIER_AVAILABLE else "live,event"
            
            photo = Photo(filename=fn, tags=tags, event_id=event_id, uploaded_by=current_user.id)
            db.session.add(photo)
    db.session.commit()
    flash('Photos uploaded to live feed.', 'success')
    return redirect(url_for('volunteer_dashboard'))

@app.route('/volunteer/event/<int:event_id>/camera-upload', methods=['POST'])
@login_required
@role_required('volunteer')
def volunteer_camera_upload(event_id):
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False}), 400
        
    fn, path = save_base64_image(data['image'], Config.PHOTOS_FOLDER, "camera")
    if fn:
        # AI ML Tagging
        tags = analyze_image(path) if ML_CLASSIFIER_AVAILABLE else "camera,live"
        
        photo = Photo(filename=fn, tags=tags, event_id=event_id, uploaded_by=current_user.id)
        db.session.add(photo)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Snapshot captured!'})
    return jsonify({'success': False}), 500

# ==============================================================================
#                                VIEWER ROUTES
# ==============================================================================

@app.route('/viewer/dashboard')
@login_required
@role_required('viewer')
def viewer_dashboard():
    # Show events that are either approved (live) or completed (archive)
    all_events = Event.query.filter(Event.status.in_(['approved', 'completed'])).order_by(Event.event_date.desc()).all()
    
    access = EventAccess.query.filter_by(viewer_email=current_user.email).all()
    unlocked_ids = [a.event_id for a in access]
    
    return render_template('viewer/dashboard.html', events=all_events, unlocked_ids=unlocked_ids)

@app.route('/viewer/unlock-event', methods=['POST'])
@login_required
@role_required('viewer')
def unlock_event():
    code = request.form.get('access_code').strip().upper()
    event = Event.query.filter_by(access_code=code).first()
    
    if event:
        # Check if already unlocked
        existing = EventAccess.query.filter_by(event_id=event.id, viewer_email=current_user.email).first()
        if not existing:
            db.session.add(EventAccess(event_id=event.id, viewer_email=current_user.email))
            db.session.commit()
        return redirect(url_for('event_gallery', event_id=event.id))
    
    flash('Invalid access code. Please try again.', 'error')
    return redirect(url_for('viewer_dashboard'))

@app.route('/viewer/event/<int:event_id>')
@login_required
def event_gallery(event_id):
    event = Event.query.get_or_404(event_id)
    # Check access for private events
    if event.event_type == 'private':
        if not EventAccess.query.filter_by(event_id=event_id, viewer_email=current_user.email).first():
            flash('This is a private gallery. Enter code to view.', 'warning')
            return redirect(url_for('viewer_dashboard'))
            
    photos = Photo.query.filter_by(event_id=event_id).order_by(Photo.uploaded_at.desc()).all()
    return render_template('viewer/event_gallery.html', event=event, photos=photos)

@app.route('/viewer/event/<int:event_id>/search', methods=['POST'])
@login_required
def search_photos(event_id):
    event = Event.query.get_or_404(event_id)
    keyword = request.form.get('keyword', '').strip().lower()
    
    # Perform search on AI-generated tags
    if not keyword:
        photos = Photo.query.filter_by(event_id=event_id).all()
    else:
        photos = Photo.query.filter(Photo.event_id == event_id, Photo.tags.like(f'%{keyword}%')).all()
        
    return render_template('viewer/event_gallery.html', event=event, photos=photos, search_keyword=keyword)

@app.route('/viewer/event/<int:event_id>/find-my-photos', methods=['GET', 'POST'])
@login_required
@role_required('viewer')
def find_my_photos(event_id):
    event = Event.query.get_or_404(event_id)
    matched = []
    
    if request.method == 'POST':
        path = None
        # Handle Camera Capture (JSON)
        if request.is_json:
            data = request.get_json()
            _, path = save_base64_image(data['image'], Config.SELFIES_FOLDER, "selfie")
        # Handle File Upload
        elif 'selfie' in request.files:
            file = request.files['selfie']
            if file and allowed_file(file.filename):
                fn = secure_filename(f"selfie_{secrets.token_hex(4)}_{file.filename}")
                path = os.path.join(Config.SELFIES_FOLDER, fn)
                file.save(path)
        
        if path:
            if FACE_RECOGNITION_AVAILABLE:
                matched = find_matching_photos(path, event_id)
                # Cleanup selfie to save space
                try: os.remove(path)
                except: pass
                
                if request.is_json:
                    return jsonify({'success': True, 'count': len(matched)})
                
                if not matched:
                    flash('No matching photos found. Try a clearer selfie.', 'info')
            else:
                flash('Face recognition is currently offline.', 'warning')
                
    return render_template('viewer/find_my_photos.html', 
                         event=event, 
                         matched_photos=matched, 
                         face_recognition_available=FACE_RECOGNITION_AVAILABLE)

from reel_engine import generate_reel

# --- DELETE INDIVIDUAL PHOTO ---
@app.route('/organizer/photo/<int:photo_id>/delete', methods=['POST'])
@login_required
@role_required('organizer')
def delete_photo(photo_id):
    photo = Photo.query.get_or_404(photo_id)
    event = db.session.get(Event, photo.event_id)

    # Security: Ensure only the organizer who owns the event can delete
    if event.organizer_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('organizer_dashboard'))

    # Delete physical file
    try:
        file_path = os.path.join(Config.PHOTOS_FOLDER, photo.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {e}")

    db.session.delete(photo)
    db.session.commit()
    flash('Photo removed from gallery.', 'success')
    return redirect(url_for('manage_event', event_id=event.id))


# --- DELETE ALL PHOTOS (ONLY FOR APPROVED EVENTS) ---
@app.route('/organizer/event/<int:event_id>/clear-all', methods=['POST'])
@login_required
@role_required('organizer')
def organizer_clear_all_photos(event_id):
    event = Event.query.get_or_404(event_id)

    # Security Check
    if event.organizer_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('organizer_dashboard'))

    # Condition: Only allow if event is approved
    if event.status != 'approved':
        flash('You can only clear the gallery after admin approval.', 'error')
        return redirect(url_for('manage_event', event_id=event.id))

    photos = Photo.query.filter_by(event_id=event_id).all()
    
    # Delete all physical files
    for photo in photos:
        try:
            path = os.path.join(Config.PHOTOS_FOLDER, photo.filename)
            if os.path.exists(path):
                os.remove(path)
        except:
            continue

    # Delete DB records
    Photo.query.filter_by(event_id=event_id).delete()
    db.session.commit()
    
    flash('All photos have been permanently deleted.', 'success')
    return redirect(url_for('manage_event', event_id=event.id))

@app.route('/viewer/event/<int:event_id>/generate-reel')
@login_required
def create_event_reel(event_id):
    flash("🎥 Generating your AI Reel... This takes about 30 seconds. Please wait.", "info")
    
    filename, error = generate_reel(event_id, db)
    
    if error:
        flash(error, "error")
        return redirect(url_for('event_gallery', event_id=event_id))
    
    # Redirect to the file directly or a preview page
    return send_file(os.path.join(Config.UPLOAD_FOLDER, filename), as_attachment=True)

@app.route('/viewer/photo/<int:photo_id>/download')
@login_required
def download_photo(photo_id):
    photo = Photo.query.get_or_404(photo_id)
    path = os.path.join(Config.PHOTOS_FOLDER, photo.filename)
    return send_file(path, as_attachment=True, download_name=photo.original_filename or photo.filename)

@app.route('/viewer/event/<int:event_id>/download-all')
@login_required
def download_all_photos(event_id):
    event = Event.query.get_or_404(event_id)
    photos = Photo.query.filter_by(event_id=event_id).all()
    
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in photos:
            path = os.path.join(Config.PHOTOS_FOLDER, p.filename)
            if os.path.exists(path):
                zf.write(path, p.filename)
    
    buf.seek(0)
    return send_file(buf, mimetype='application/zip', as_attachment=True, download_name=f'{event.name}_all_photos.zip')

@app.route('/viewer/download-matched', methods=['POST'])
@login_required
def download_matched_photos():
    ids = request.form.getlist('photo_ids')
    photos = Photo.query.filter(Photo.id.in_(ids)).all()
    
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in photos:
            path = os.path.join(Config.PHOTOS_FOLDER, p.filename)
            if os.path.exists(path):
                zf.write(path, p.filename)
                
    buf.seek(0)
    return send_file(buf, mimetype='application/zip', as_attachment=True, download_name='my_found_photos.zip')

# ==============================================================================
#                                SYSTEM INIT
# ==============================================================================



def init_db():
    with app.app_context():
        db.create_all()
        # Create Default Master Admin
        if not User.query.filter_by(email='admin@eventsnap.com').first():
            admin = User(
                email='admin@eventsnap.com', 
                name='System Admin', 
                role='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("📦 Database initialized with Admin account.")

socketio = SocketIO(app, cors_allowed_origins="*", manage_session=False)

# --- LIVE STREAMING SOCKETS ---

# ==================== SOCKET.IO EVENTS ====================

# ==================== SOCKET.IO DEBUGGED LOGIC ====================

# ==================== SOCKET.IO EVENTS ====================

# ==================== SOCKET.IO EVENTS ====================

from datetime import datetime
from flask_socketio import join_room, leave_room, emit

# Global variables for tracking
active_streams = {}
audio_stats = {}
audio_headers = {}  # Cache the first chunk (header) for each event

@socketio.on('connect')
def handle_connect():
    print("🟢 [SOCKET] A client connected")

@socketio.on('join_event')
def on_join(data):
    event_id = str(data['event_id'])
    room = event_id
    join_room(room)
    print(f"👤 [SOCKET] User joined Room: {room}")
    
    # If there's an active audio stream, send the header chunk first!
    if event_id in audio_headers:
        print(f"📡 [AUDIO] Sending cached header to new user in room {room}")
        header_data = audio_headers[event_id]
        emit('new_audio', header_data, room=request.sid)

@socketio.on('leave_event')
def on_leave(data):
    room = str(data['event_id'])
    leave_room(room)
    print(f"👋 [SOCKET] User left Room: {room}")

@socketio.on('start_stream')
def handle_start_stream(data):
    event_id = str(data['event_id'])
    room = event_id
    
    # Clear old header
    if event_id in audio_headers:
        del audio_headers[event_id]
        
    # Update DB
    with app.app_context():
        event = db.session.get(Event, int(event_id))
        if event:
            event.is_live = True
            db.session.commit()
            print(f"🔴 [SOCKET] Stream STARTED for Event {event_id} in Room: {room}")
            
            # Broadcast to everyone in the room
            emit('stream_started', {'event_id': event_id}, room=room, broadcast=True)

@socketio.on('stream_frame')
def handle_stream_frame(data):
    room = str(data['event_id'])  # Consistent
    emit('new_frame', {'image': data['image']}, room=room, include_self=False)

@socketio.on('stream_audio')
def handle_stream_audio(data):
    """Stream audio chunks to viewers - FIXED VERSION"""
    try:
        # Validate event_id
        if not data or 'event_id' not in data:
            print("⚠️ [AUDIO] Missing event_id")
            return {'status': 'error', 'message': 'Missing event_id'}
        
        event_id = str(data['event_id'])
        room = str(event_id)  # FIXED: Use same room naming as video!
        
        # Get audio data
        audio_data = data.get('audio')
        if not audio_data:
            print(f"⚠️ [AUDIO] Empty audio for event {event_id}")
            return {'status': 'error', 'message': 'No audio data'}
        
        # Process audio data format
        mime_type = data.get('mimeType', 'audio/webm')
        
        # Ensure proper data URI format
        if isinstance(audio_data, str) and not audio_data.startswith('data:'):
            audio_data = f'data:{mime_type};base64,{audio_data}'
        
        # Update statistics
        if event_id not in audio_stats:
            audio_stats[event_id] = {
                'started_at': datetime.utcnow(),
                'chunks_sent': 0,
                'total_bytes': 0
            }
            
        audio_stats[event_id]['chunks_sent'] += 1
        audio_stats[event_id]['total_bytes'] += len(audio_data)
        
        # CACHE HEADER: If this is the first chunk, save it!
        if audio_stats[event_id]['chunks_sent'] == 1:
            print(f"💾 [AUDIO] Caching header for event {event_id}")
            audio_headers[event_id] = {
                'audio': audio_data,
                'mimeType': mime_type,
                'timestamp': data.get('timestamp', datetime.utcnow().timestamp()),
                'event_id': event_id,
                'is_header': True # Flag it
            }
        
        # Log every 50th chunk
        chunk_count = audio_stats[event_id]['chunks_sent']
        if chunk_count % 50 == 1: 
            print(f"🎵 [AUDIO] Event: {event_id}, Chunks sent: {chunk_count}, Room: {room}")
        
        # Emit to all viewers in the same room
        emit('new_audio', {
            'audio': audio_data,
            'mimeType': mime_type,
            'timestamp': data.get('timestamp', datetime.utcnow().timestamp()),
            'event_id': event_id
        }, room=room, include_self=False)
        
        return {'status': 'success'}
        
    except Exception as e:
        print(f"❌ [AUDIO] Stream error: {str(e)}")
        return {'status': 'error', 'message': str(e)}
@app.route('/debug/rooms/<event_id>')
def debug_rooms(event_id):
    """Debug endpoint to check who's in a room"""
    room = str(event_id)
    room_info = {
        'room': room,
        'clients': [],
        'audio_stats': audio_stats.get(str(event_id), {}),
        'is_streaming': str(event_id) in active_streams
    }
    return jsonify(room_info)

@socketio.on('stream_audio_pcm')
def handle_stream_audio_pcm(data):
    """Stream RAW PCM audio chunks (Web Audio API)"""
    try:
        if not data or 'event_id' not in data:
            return
            
        event_id = str(data['event_id'])
        room = event_id
        
        # Log occasionally
        if event_id not in audio_stats:
            audio_stats[event_id] = {'chunks': 0}
        
        audio_stats[event_id]['chunks'] += 1
        if audio_stats[event_id]['chunks'] % 50 == 1:
            print(f"🎤 [PCM] Relay chunk #{audio_stats[event_id]['chunks']} for {event_id}")

        # Broadcast to room
        emit('new_audio_pcm', data, room=room, include_self=False)

    except Exception as e:
        print(f"❌ [PCM] Error: {e}")

@socketio.on('stop_stream')
def handle_stop_stream(data):
    event_id = str(data['event_id'])
    
    # Clear header
    if event_id in audio_headers:
        del audio_headers[event_id]
        
    # Rest of logic...
    room = str(event_id)
    
    # Clean up audio stats if exists
    if str(event_id) in audio_stats:
        stats = audio_stats[str(event_id)]
        duration = (datetime.utcnow() - stats.get('started_at', datetime.utcnow())).total_seconds()
        print(f"📊 [AUDIO] Final stats - Event: {event_id}, Duration: {duration:.1f}s, Total chunks: {stats.get('chunks_sent', 0)}")
        del audio_stats[str(event_id)]
    
    with app.app_context():
        event = db.session.get(Event, int(event_id))
        if event:
            event.is_live = False
            db.session.commit()
            print(f"⏹️ [SOCKET] Stream STOPPED for Event {event_id}")
            emit('stream_stopped', {'event_id': event_id}, room=room, broadcast=True)

@socketio.on('send_reaction')
def handle_reaction(data):
    room = str(data['event_id'])
    emit('new_reaction', {'emoji': data['emoji']}, room=room, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    print("🔴 [SOCKET] A client disconnected")

# Remove duplicate if __name__ == '__main__' blocks!
# Keep only one:
if __name__ == '__main__':
    init_db()
    print("\n" + "="*50)
    print("🚀 EVENT SNAP SERVER RUNNING")
    print(f"📍 URL: http://127.0.0.1:5000")
    print("="*50 + "\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)