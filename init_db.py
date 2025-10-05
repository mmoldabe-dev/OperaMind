"""
OperaMind - Инициализация БД v2
С поддержкой новых полей
"""

from app import app, db
from models import User

def init_database():
    with app.app_context():
        # Удаляем старую БД если есть (для чистой установки)
        # db.drop_all()  # Раскомментируйте для полного сброса
        
        # Создаём все таблицы
        db.create_all()
        print('📦 Создание таблиц...')
        
        # Проверка пользователя
        existing_user = User.query.filter_by(username='operator').first()
        
        if not existing_user:
            demo_user = User(username='operator')
            demo_user.set_password('demo123')
            
            db.session.add(demo_user)
            db.session.commit()
            
            print('✅ База данных инициализирована!')
            print('✅ Пользователь создан:')
            print('   📧 Логин: operator')
            print('   🔑 Пароль: demo123')
        else:
            print('ℹ️  База данных уже существует')
            print('ℹ️  Логин: operator / Пароль: demo123')
        
        # Создание администратора
        existing_admin = User.query.filter_by(username='admin').first()
        if not existing_admin:
            admin_user = User(username='admin', is_admin=True)
            admin_user.set_password('admin123')
            db.session.add(admin_user)
            db.session.commit()
            print('✅ Администратор создан:')
            print('   📧 Логин: admin')
            print('   🔑 Пароль: admin123')
        else:
            print('ℹ️  Админ уже существует: admin / admin123')
        
        print('\n📋 НОВЫЕ ВОЗМОЖНОСТИ:')
        print('  • Раздельная транскрипция и анализ')
        print('  • Поддержка TXT файлов')
        print('  • Подробный анализ разговоров')
        print('  • Оценка работы оператора')
        print('  • Рекомендации по улучшению')
        print('  • Скачивание отчётов')

if __name__ == '__main__':
    init_database()