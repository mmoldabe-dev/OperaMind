"""
OperaMind - Инициализация БД v3
С поддержкой ролевой системы
"""

from app import app, db
from models import User, UserRole

def init_database():
    with app.app_context():
        # Создаём все таблицы
        db.create_all()
        print('📦 Создание таблиц...')
        
        # Создание администратора
        existing_admin = User.query.filter_by(username='admin').first()
        if not existing_admin:
            admin_user = User(
                username='admin', 
                is_admin=True,
                role=UserRole.ADMIN,
                full_name='Системный администратор',
                email='admin@operamind.com'
            )
            admin_user.set_password('admin123')
            db.session.add(admin_user)
            print('✅ Администратор создан:')
            print('   📧 Логин: admin')
            print('   🔑 Пароль: admin123')
        else:
            print('ℹ️  Админ уже существует: admin / admin123')
        
        # Создание оператора
        existing_operator = User.query.filter_by(username='operator').first()
        if not existing_operator:
            operator_user = User(
                username='operator',
                role=UserRole.OPERATOR,
                full_name='Демо Оператор',
                email='operator@operamind.com',
                department='Call Center'
            )
            operator_user.set_password('demo123')
            db.session.add(operator_user)
            print('✅ Оператор создан:')
            print('   📧 Логин: operator')
            print('   🔑 Пароль: demo123')
        else:
            print('ℹ️  Оператор уже существует: operator / demo123')
        
        # Создание обычного пользователя
        existing_user = User.query.filter_by(username='user').first()
        if not existing_user:
            regular_user = User(
                username='user',
                role=UserRole.USER,
                full_name='Тестовый пользователь',
                email='user@operamind.com'
            )
            regular_user.set_password('user123')
            db.session.add(regular_user)
            print('✅ Пользователь создан:')
            print('   📧 Логин: user')
            print('   🔑 Пароль: user123')
        else:
            print('ℹ️  Пользователь уже существует: user / user123')
        
        db.session.commit()
        
        print('\n🎯 СИСТЕМА РОЛЕЙ:')
        print('  👑 ADMIN - Полный доступ к системе')
        print('  📞 OPERATOR - Анализ собственных звонков')
        print('  👤 USER - Базовые функции')
        
        print('\n� НОВЫЕ ВОЗМОЖНОСТИ:')
        print('  • Современный дизайн с единой системой стилей')
        print('  • Whisper/Vosk/Google транскрипция')
        print('  • Chunked обработка для лучшего качества')
        print('  • Дедупликация и оптимизация')
        print('  • Ролевая система доступа')
        print('  • Аудит логи действий')
        print('  • Адаптивный интерфейс')

if __name__ == '__main__':
    init_database()