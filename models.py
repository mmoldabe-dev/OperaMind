"""
OperaMind - Модели базы данных (расширенная версия)
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    
    conversations = db.relationship('Conversation', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    duration = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Статус обработки
    status = db.Column(db.String(50), default='pending')  # pending/transcribing/analyzing/completed/error
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    analysis = db.relationship('Analysis', backref='conversation', uselist=False, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Conversation {self.filename}>'


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    
    # Транскрипция
    transcript = db.Column(db.Text)
    
    # Основной анализ
    topic = db.Column(db.String(100))
    category = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    urgency = db.Column(db.String(50))
    keywords = db.Column(db.Text)  # JSON
    summary = db.Column(db.Text)
    
    # Расширенный анализ
    detailed_analysis = db.Column(db.Text)  # Подробный анализ
    operator_quality = db.Column(db.String(50))  # Оценка работы оператора
    recommendations = db.Column(db.Text)  # Рекомендации
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Analysis for Conversation {self.conversation_id}>'