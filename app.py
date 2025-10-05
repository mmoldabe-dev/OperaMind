import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from models import db, User, Conversation, Analysis
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///operator_analysis.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB максимальный размер файла

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a'}

# Инициализация базы данных
db.init_app(app)

# Инициализация Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Пожалуйста, войдите для доступа к этой странице'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Создание папки для загрузок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Вход выполнен успешно!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Неверный логин или пароль', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Вы вышли из системы', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('Файл не выбран', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('Файл не выбран', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        # Создание записи в базе данных
        conversation = Conversation(
            filename=filename,
            filepath=filepath,
            user_id=current_user.id
        )
        db.session.add(conversation)
        db.session.commit()
        
        flash(f'Файл {file.filename} успешно загружен!', 'success')
        return redirect(url_for('history'))
    else:
        flash('Недопустимый формат файла. Разрешены: mp3, wav, m4a', 'error')
        return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    conversations = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.upload_date.desc()).all()
    return render_template('history.html', conversations=conversations)

@app.route('/delete/<int:conversation_id>', methods=['POST'])
@login_required
def delete_conversation(conversation_id):
    conversation = Conversation.query.get_or_404(conversation_id)
    
    # Проверка прав доступа
    if conversation.user_id != current_user.id:
        flash('У вас нет прав для удаления этого разговора', 'error')
        return redirect(url_for('history'))
    
    # Удаление файла
    if os.path.exists(conversation.filepath):
        os.remove(conversation.filepath)
    
    # Удаление из БД
    db.session.delete(conversation)
    db.session.commit()
    
    flash('Разговор успешно удален', 'success')
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True)