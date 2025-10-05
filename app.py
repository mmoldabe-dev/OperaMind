"""
OperaMind - Главное приложение v2
Раздельная транскрипция и анализ
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv
import json
from io import BytesIO

from models import db, User, Conversation, Analysis
from transcriber import transcribe_audio_file, transcribe_from_text
from analyzer import analyze_transcript
from stats import stats_bp
from stats_user import stats_user_bp
from admin import admin_bp

load_dotenv()

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'operamind-secret-key-2025')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///operamind.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'txt'}

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@app.template_filter('from_json')
def from_json_filter(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return []
    return value if isinstance(value, list) else []

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            flash('Добро пожаловать в OperaMind!', 'success')
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
        
        # Создание записи
        conversation = Conversation(
            filename=filename,
            filepath=filepath,
            user_id=current_user.id,
            status='pending'
        )
        db.session.add(conversation)
        db.session.commit()
        
        flash(f'✅ Файл загружен!', 'success')
        
        # ЭТАП 1: ТРАНСКРИПЦИЯ
        print("\n" + "="*60)
        print("ЭТАП 1: ТРАНСКРИПЦИЯ")
        print("="*60)
        
        conversation.status = 'transcribing'
        db.session.commit()
        
        try:
            # Определяем тип файла
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    structured_transcript = transcribe_from_text(f.read())
                print("✅ Текст загружен из файла")
            else:
                structured_transcript = transcribe_audio_file(filepath)

            if not structured_transcript:
                raise Exception("Транскрипция не выполнена")

            # Попытка извлечь полный плоский текст
            flat_text = None
            try:
                import json as _json
                data_tr = _json.loads(structured_transcript)
                if isinstance(data_tr, dict) and 'text' in data_tr:
                    flat_text = data_tr.get('text')
                else:
                    flat_text = structured_transcript
            except Exception:
                flat_text = structured_transcript
            
            # ЭТАП 2: АНАЛИЗ
            print("\n" + "="*60)
            print("ЭТАП 2: АНАЛИЗ ТЕКСТА")
            print("="*60)
            
            conversation.status = 'analyzing'
            db.session.commit()
            
            analysis_result = analyze_transcript(flat_text)
            
            # Сохранение результатов
            analysis = Analysis(
                conversation_id=conversation.id,
                transcript=structured_transcript,  # сохраняем структурированный JSON
                topic=analysis_result.get('topic'),
                category=analysis_result.get('category'),
                sentiment=analysis_result.get('sentiment'),
                urgency=analysis_result.get('urgency'),
                keywords=json.dumps(analysis_result.get('keywords', []), ensure_ascii=False),
                summary=analysis_result.get('summary'),
                detailed_analysis=analysis_result.get('detailed_analysis'),
                operator_quality=analysis_result.get('operator_quality'),
                recommendations=analysis_result.get('recommendations')
            )
            
            conversation.status = 'completed'
            db.session.add(analysis)
            db.session.commit()
            
            flash('🎉 Обработка завершена!', 'success')
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            conversation.status = 'error'
            db.session.commit()
            flash(f'❌ Ошибка обработки: {str(e)}', 'error')
        
        return redirect(url_for('conversation_detail', conversation_id=conversation.id))
    else:
        flash('❌ Недопустимый формат. Разрешены: MP3, WAV, M4A, OGG, TXT', 'error')
        return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    conversations = Conversation.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Conversation.upload_date.desc()
    ).all()
    
    return render_template('history.html', conversations=conversations)

@app.route('/conversation/<int:conversation_id>')
@login_required
def conversation_detail(conversation_id):
    conversation = Conversation.query.get_or_404(conversation_id)
    # Доступ: владелец или админ
    if conversation.user_id != current_user.id and not getattr(current_user, 'is_admin', False):
        flash('❌ Нет доступа', 'error')
        return redirect(url_for('history'))
    return render_template('conversation_detail.html', conversation=conversation)

@app.route('/download/<int:conversation_id>')
@login_required
def download_report(conversation_id):
    """Скачать отчёт по разговору"""
    conversation = Conversation.query.get_or_404(conversation_id)
    
    if conversation.user_id != current_user.id:
        flash('❌ Нет доступа', 'error')
        return redirect(url_for('history'))
    
    if not conversation.analysis:
        flash('❌ Анализ не выполнен', 'error')
        return redirect(url_for('conversation_detail', conversation_id=conversation_id))
    
    # Формируем отчёт
    a = conversation.analysis
    keywords = json.loads(a.keywords) if a.keywords else []
    
    # Попытка преобразовать структурированную транскрипцию
    pretty_transcript = a.transcript
    try:
        data_t = json.loads(a.transcript)
        if isinstance(data_t, dict) and 'segments' in data_t:
            lines = []
            for seg in data_t.get('segments', []):
                speaker = seg.get('speaker', 'unknown')
                role = 'Клиент' if speaker == 'client' else ('Оператор' if speaker == 'operator' else 'Неизвестно')
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                text_line = seg.get('text', '')
                lines.append(f"[{start:06.2f}–{end:06.2f}] {role}: {text_line}")
            pretty_transcript = "\n".join(lines)
    except Exception:
        pass

    report = f"""
ОТЧЁТ ПО РАЗГОВОРУ
{'='*70}

ФАЙЛ: {conversation.filename}
ДАТА: {conversation.upload_date.strftime('%d.%m.%Y %H:%M')}

{'='*70}
КРАТКАЯ ИНФОРМАЦИЯ
{'='*70}

Тема: {a.topic}
Категория: {a.category}
Тональность: {a.sentiment}
Срочность: {a.urgency}
Качество работы оператора: {a.operator_quality}

{'='*70}
КЛЮЧЕВЫЕ СЛОВА
{'='*70}

{', '.join(keywords)}

{'='*70}
КРАТКОЕ СОДЕРЖАНИЕ
{'='*70}

{a.summary}

{'='*70}
ПОДРОБНЫЙ АНАЛИЗ
{'='*70}

{a.detailed_analysis}

{'='*70}
РЕКОМЕНДАЦИИ
{'='*70}

{a.recommendations}

{'='*70}
ПОЛНАЯ ТРАНСКРИПЦИЯ (ДИАЛОГ)
{'='*70}

{pretty_transcript}

{'='*70}
Сгенерировано системой OperaMind
"""
    
    # Создаём файл в памяти
    output = BytesIO()
    output.write(report.encode('utf-8'))
    output.seek(0)
    
    return send_file(
        output,
        as_attachment=True,
        download_name=f'report_{conversation.id}_{datetime.now().strftime("%Y%m%d")}.txt',
        mimetype='text/plain'
    )

@app.route('/delete/<int:conversation_id>', methods=['POST'])
@login_required
def delete_conversation(conversation_id):
    conversation = Conversation.query.get_or_404(conversation_id)
    
    if conversation.user_id != current_user.id:
        flash('❌ Нет доступа', 'error')
        return redirect(url_for('history'))
    
    if os.path.exists(conversation.filepath):
        os.remove(conversation.filepath)
    
    db.session.delete(conversation)
    db.session.commit()
    
    flash('🗑️ Разговор удалён', 'success')
    return redirect(url_for('history'))

app.register_blueprint(stats_bp)
app.register_blueprint(stats_user_bp)
app.register_blueprint(admin_bp)

if __name__ == '__main__':
    app.run(debug=True)