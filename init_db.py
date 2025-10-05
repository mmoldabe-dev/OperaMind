"""
OperaMind - Инициализация базы данных
Запустите этот файл ОДИН РАЗ перед первым запуском приложения
"""

from app import app, db
from models import User

def init_database():
    """Создание базы данных и демо-пользователя"""
    with app.app_context():
        # Создание всех таблиц
        db.create_all()
        print('📦 Создание таблиц базы данных...')
        
        # Проверка существования пользователя
        existing_user = User.query.filter_by(username='operator').first()
        
        if not existing_user:
            # Создание демо-пользователя
            demo_user = User(username='operator')
            demo_user.set_password('demo123')
            
            db.session.add(demo_user)
            db.session.commit()
            
            print('✅ База данных успешно инициализирована!')
            print('✅ Создан пользователь:')
            print('   Логин: operator')
            print('   Пароль: demo123')
        else:
            print('ℹ️  База данных уже существует')
            print('ℹ️  Используйте логин: operator / пароль: demo123')

if __name__ == '__main__':
    init_database()