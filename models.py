from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import secrets

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False) # admin, organizer, volunteer, viewer
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    organized_events = db.relationship('Event', backref='organizer', lazy=True, foreign_keys='Event.organizer_id')
    volunteer_assignments = db.relationship('VolunteerAssignment', backref='volunteer', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Event(db.Model):
    __tablename__ = 'events'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    venue = db.Column(db.String(200))
    event_date = db.Column(db.DateTime, nullable=False)
    event_type = db.Column(db.String(20), default='public') # public, private
    access_code = db.Column(db.String(10), unique=True)
    poster_filename = db.Column(db.String(255))
    status = db.Column(db.String(20), default='pending') # pending, approved, completed, rejected
    organizer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    is_live = db.Column(db.Boolean, default=False) # NEW: For Live Streaming
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    photos = db.relationship('Photo', backref='event', lazy=True, cascade='all, delete-orphan')
    volunteers = db.relationship('VolunteerAssignment', backref='event', lazy=True, cascade='all, delete-orphan')
    
    def generate_access_code(self):
        self.access_code = secrets.token_hex(4).upper()
        return self.access_code


class Photo(db.Model):
    __tablename__ = 'photos'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255))
    tags = db.Column(db.String(500)) 
    event_id = db.Column(db.Integer, db.ForeignKey('events.id'), nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)


class VolunteerAssignment(db.Model):
    __tablename__ = 'volunteer_assignments'
    
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey('events.id'), nullable=False)
    volunteer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)


class EventAccess(db.Model):
    __tablename__ = 'event_access'
    
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey('events.id'), nullable=False)
    viewer_email = db.Column(db.String(120), nullable=False)
    accessed_at = db.Column(db.DateTime, default=datetime.utcnow)