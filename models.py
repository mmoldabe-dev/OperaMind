"""
OperaMind - Модели базы данных (расширенная версия с ролями)
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from enum import Enum

db = SQLAlchemy()

class UserRole(Enum):
    USER = "user"
    OPERATOR = "operator"
    ADMIN = "admin"

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    role = db.Column(db.Enum(UserRole), default=UserRole.OPERATOR, nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    
    full_name = db.Column(db.String(200))
    department = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    
    conversations = db.relationship('Conversation', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    @property
    def is_administrator(self):
        return self.role == UserRole.ADMIN or self.is_admin
    
    @property
    def is_admin_property(self):
        return self.is_administrator
    
    @property
    def is_operator(self):
        return self.role == UserRole.OPERATOR
    
    @property
    def role_display(self):
        role_names = {
            UserRole.USER: "Пользователь",
            UserRole.OPERATOR: "Оператор",
            UserRole.ADMIN: "Администратор"
        }
        return role_names.get(self.role, "Неизвестно")
    
    def can_access_admin_panel(self):
        return self.is_administrator
    
    def can_view_all_conversations(self):
        return self.is_administrator
    
    def can_manage_users(self):
        return self.is_administrator
    
    def __repr__(self):
        return f'<User {self.username} ({self.role_display})>'


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    duration = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='pending')
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    analysis = db.relationship('Analysis', backref='conversation', uselist=False, cascade='all, delete-orphan')
    
    @property
    def status_display(self):
        """Человекочитаемый статус"""
        status_names = {
            'queued': '📋 В очереди',
            'pending': 'Ожидает обработки',
            'transcribing': '📝 Транскрипция',
            'analyzing': '🤖 Анализ',
            'completed': '✅ Завершено',
            'error': '❌ Ошибка'
        }
        return status_names.get(self.status, 'Неизвестно')
    
    @property
    def file_size_display(self):
        """Размер файла в читаемом формате"""
        if not self.file_size:
            return "Неизвестно"
        
        if self.file_size < 1024:
            return f"{self.file_size} Б"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} КБ"
        else:
            return f"{self.file_size / (1024 * 1024):.1f} МБ"
    
    def __repr__(self):
        return f'<Conversation {self.filename}>'


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    
    transcript = db.Column(db.Text)
    
    topic = db.Column(db.String(100))
    category = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    urgency = db.Column(db.String(50))
    keywords = db.Column(db.Text)
    summary = db.Column(db.Text)
    
    detailed_analysis = db.Column(db.Text)
    operator_quality = db.Column(db.String(50))
    recommendations = db.Column(db.Text)
    
    quality_score = db.Column(db.Float)
    response_time_rating = db.Column(db.String(20))
    politeness_rating = db.Column(db.String(20))
    problem_solved = db.Column(db.Boolean)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    @property
    def sentiment_display(self):
        """Человекочитаемая тональность"""
        sentiment_names = {
            # Английские варианты
            'positive': 'Позитивная',
            'negative': 'Негативная', 
            'neutral': 'Нейтральная',
            'mixed': 'Смешанная',
            # Русские варианты (из Gemini)
            'позитивный': 'Позитивная',
            'негативный': 'Негативная',
            'нейтральный': 'Нейтральная',
            'смешанный': 'Смешанная'
        }
        return sentiment_names.get(self.sentiment, 'Неопределено')
    
    @property
    def urgency_display(self):
        """Человекочитаемая срочность"""
        urgency_names = {
            # Английские варианты
            'low': 'Низкая',
            'medium': 'Средняя',
            'high': 'Высокая',
            'critical': 'Критическая',
            # Русские варианты (из Gemini)
            'низкая': 'Низкая',
            'средняя': 'Средняя',
            'высокая': 'Высокая',
            'критическая': 'Критическая'
        }
        return urgency_names.get(self.urgency, 'Неопределено')
    
    def __repr__(self):
        return f'<Analysis for Conversation {self.conversation_id}>'


class SystemSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text)
    description = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='audit_logs')
    
    def __repr__(self):
        return f'<AuditLog {self.user.username}: {self.action}>'