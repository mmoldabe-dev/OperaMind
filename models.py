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
    USER = "user"           # Обычный пользователь
    OPERATOR = "operator"   # Оператор call-центра
    ADMIN = "admin"         # Администратор системы

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Роли и права доступа
    role = db.Column(db.Enum(UserRole), default=UserRole.OPERATOR, nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # Обратная совместимость
    
    # Дополнительная информация
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
        """Проверка на администратора"""
        return self.role == UserRole.ADMIN or self.is_admin
    
    @property
    def is_admin_property(self):
        """Алиас для совместимости с Flask-Login"""
        return self.is_administrator
    
    @property
    def is_operator(self):
        """Проверка на оператора"""
        return self.role == UserRole.OPERATOR
    
    @property
    def role_display(self):
        """Человекочитаемое название роли"""
        role_names = {
            UserRole.USER: "Пользователь",
            UserRole.OPERATOR: "Оператор",
            UserRole.ADMIN: "Администратор"
        }
        return role_names.get(self.role, "Неизвестно")
    
    def can_access_admin_panel(self):
        """Может ли пользователь получить доступ к админ-панели"""
        return self.is_administrator
    
    def can_view_all_conversations(self):
        """Может ли пользователь видеть все разговоры"""
        return self.is_administrator
    
    def can_manage_users(self):
        """Может ли пользователь управлять пользователями"""
        return self.is_administrator
    
    def __repr__(self):
        return f'<User {self.username} ({self.role_display})>'


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    duration = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Статус обработки
    status = db.Column(db.String(50), default='pending')  # pending/transcribing/analyzing/completed/error
    
    # Метаданные файла
    file_size = db.Column(db.Integer)  # Размер в байтах
    mime_type = db.Column(db.String(100))
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    analysis = db.relationship('Analysis', backref='conversation', uselist=False, cascade='all, delete-orphan')
    
    @property
    def status_display(self):
        """Человекочитаемый статус"""
        status_names = {
            'pending': 'Ожидает обработки',
            'transcribing': 'Транскрипция',
            'analyzing': 'Анализ',
            'completed': 'Завершено',
            'error': 'Ошибка'
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
    
    # Метрики качества
    quality_score = db.Column(db.Float)  # Общая оценка от 0 до 10
    response_time_rating = db.Column(db.String(20))  # Быстро/Средне/Медленно
    politeness_rating = db.Column(db.String(20))  # Вежливо/Нейтрально/Грубо
    problem_solved = db.Column(db.Boolean)  # Решена ли проблема клиента
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    @property
    def sentiment_display(self):
        """Человекочитаемая тональность"""
        sentiment_names = {
            'positive': 'Позитивная',
            'negative': 'Негативная', 
            'neutral': 'Нейтральная',
            'mixed': 'Смешанная'
        }
        return sentiment_names.get(self.sentiment, 'Неопределено')
    
    @property
    def urgency_display(self):
        """Человекочитаемая срочность"""
        urgency_names = {
            'low': 'Низкая',
            'medium': 'Средняя',
            'high': 'Высокая',
            'critical': 'Критическая'
        }
        return urgency_names.get(self.urgency, 'Неопределено')
    
    def __repr__(self):
        return f'<Analysis for Conversation {self.conversation_id}>'


# Дополнительные модели для расширенной функциональности

class SystemSettings(db.Model):
    """Настройки системы"""
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text)
    description = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(db.Model):
    """Логи действий пользователей"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)  # login, upload, view, etc.
    details = db.Column(db.Text)  # JSON with additional info
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='audit_logs')
    
    def __repr__(self):
        return f'<AuditLog {self.user.username}: {self.action}>'