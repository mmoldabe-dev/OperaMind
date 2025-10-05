"""
OperaMind - Главное приложение v2
Раздельная транскрипция и анализ
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv
import json
from io import BytesIO
from sqlalchemy import func

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
        
        # Получаем размер файла
        file_size = os.path.getsize(filepath)
        
        # Создание записи
        conversation = Conversation(
            filename=filename,
            filepath=filepath,
            user_id=current_user.id,
            status='pending',
            file_size=file_size
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

            # Извлекаем длительность из метаданных
            try:
                data_tr = json.loads(structured_transcript)
                if isinstance(data_tr, dict) and 'meta' in data_tr:
                    conversation.duration = int(data_tr['meta'].get('duration_sec', 0))
            except:
                pass

            # Попытка извлечь полный плоский текст
            flat_text = None
            try:
                data_tr = json.loads(structured_transcript)
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
                transcript=structured_transcript,
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
    # Пагинация
    page = request.args.get('page', 1, type=int)
    per_page = 25
    
    # Фильтры
    status_filter = request.args.get('status', '')
    search_query = request.args.get('search', '')
    
    query = Conversation.query.filter_by(user_id=current_user.id)
    
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    if search_query:
        query = query.filter(Conversation.filename.ilike(f'%{search_query}%'))
    
    conversations = query.order_by(Conversation.upload_date.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
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
    
    # Проверка доступа
    if conversation.user_id != current_user.id and not getattr(current_user, 'is_admin', False):
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
                lines.append(f"[{start:06.2f}—{end:06.2f}] {role}: {text_line}")
            pretty_transcript = "\n".join(lines)
    except Exception:
        pass

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ОТЧЁТ ПО РАЗГОВОРУ                            ║
║                      OperaMind v2.0                              ║
╚══════════════════════════════════════════════════════════════════╝

ФАЙЛ: {conversation.filename}
ДАТА: {conversation.upload_date.strftime('%d.%m.%Y %H:%M')}
ПОЛЬЗОВАТЕЛЬ: {conversation.user.username}
ДЛИТЕЛЬНОСТЬ: {conversation.duration if conversation.duration else 'N/A'} сек

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
{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
"""
    
    # Создаём файл в памяти
    output = BytesIO()
    output.write(report.encode('utf-8-sig'))  # BOM для правильного отображения в Windows
    output.seek(0)
    
    filename = f'report_{conversation.id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype='text/plain; charset=utf-8'
    )

@app.route('/delete/<int:conversation_id>', methods=['POST'])
@login_required
def delete_conversation(conversation_id):
    conversation = Conversation.query.get_or_404(conversation_id)
    
    if conversation.user_id != current_user.id and not getattr(current_user, 'is_admin', False):
        flash('❌ Нет доступа', 'error')
        return redirect(url_for('history'))
    
    if os.path.exists(conversation.filepath):
        os.remove(conversation.filepath)
    
    db.session.delete(conversation)
    db.session.commit()
    
    flash('🗑️ Разговор удалён', 'success')
    return redirect(url_for('history'))

# ======= НОВЫЕ ФУНКЦИИ ИЗ ROADMAP =======

@app.route('/stats')
@login_required
def user_stats():
    """Персональная статистика пользователя"""
    user_id = current_user.id
    
    # Общая статистика
    total = Conversation.query.filter_by(user_id=user_id).count()
    completed = Conversation.query.filter_by(user_id=user_id, status='completed').count()
    errors = Conversation.query.filter_by(user_id=user_id, status='error').count()
    processing = total - completed - errors
    
    # Средняя длительность
    avg_duration = db.session.query(func.avg(Conversation.duration)).filter_by(user_id=user_id).scalar() or 0
    
    # Тональность
    sentiments = db.session.query(Analysis.sentiment, func.count()).join(Conversation).filter(
        Conversation.user_id == user_id
    ).group_by(Analysis.sentiment).all()
    sentiment_stats = {s or 'неопределено': c for s, c in sentiments}
    
    # Срочность
    urgencies = db.session.query(Analysis.urgency, func.count()).join(Conversation).filter(
        Conversation.user_id == user_id
    ).group_by(Analysis.urgency).all()
    urgency_stats = {u or 'неопределено': c for u, c in urgencies}
    
    # Категории
    categories = db.session.query(Analysis.category, func.count()).join(Conversation).filter(
        Conversation.user_id == user_id
    ).group_by(Analysis.category).all()
    category_stats = {c or 'неопределено': n for c, n in categories}
    
    # Топ ключевые слова
    kw_rows = db.session.query(Analysis.keywords).join(Conversation).filter(
        Conversation.user_id == user_id,
        Analysis.keywords.isnot(None)
    ).all()
    
    from collections import Counter
    all_keywords = []
    for kw_json, in kw_rows:
        try:
            all_keywords.extend(json.loads(kw_json))
        except:
            pass
    
    top_keywords = [w for w, _ in Counter(all_keywords).most_common(10)]
    
    # Активность по дням недели
    from collections import defaultdict
    import calendar
    
    calls_by_weekday = defaultdict(int)
    convs = Conversation.query.filter_by(user_id=user_id).all()
    for conv in convs:
        wd = conv.upload_date.weekday()
        calls_by_weekday[wd] += 1
    
    weekday_labels = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    weekday_data = [calls_by_weekday.get(i, 0) for i in range(7)]
    
    return render_template('user_stats.html',
        total=total,
        completed=completed,
        errors=errors,
        processing=processing,
        avg_duration=avg_duration,
        sentiments=sentiment_stats,
        urgencies=urgency_stats,
        categories=category_stats,
        top_keywords=top_keywords,
        weekday_labels=weekday_labels,
        weekday_data=weekday_data
    )

@app.route('/admin/analytics')
@login_required
def admin_analytics():
    """Глобальная аналитика для администратора"""
    if not getattr(current_user, 'is_admin', False):
        flash('Нет доступа', 'error')
        return redirect(url_for('index'))
    
    # Общая статистика
    total = Conversation.query.count()
    completed = Conversation.query.filter_by(status='completed').count()
    errors = Conversation.query.filter_by(status='error').count()
    processing = total - completed - errors
    
    # Средняя длительность
    avg_duration = db.session.query(func.avg(Conversation.duration)).scalar() or 0
    
    # Статистика по пользователям
    user_stats = db.session.query(
        User.username,
        func.count(Conversation.id).label('total'),
        func.sum(func.cast(Conversation.status == 'completed', db.Integer)).label('completed'),
        func.sum(func.cast(Conversation.status == 'error', db.Integer)).label('errors')
    ).join(Conversation).group_by(User.id).all()
    
    # Тональность (глобально)
    sentiments = db.session.query(Analysis.sentiment, func.count()).group_by(Analysis.sentiment).all()
    sentiment_stats = {s or 'неопределено': c for s, c in sentiments}
    
    # Срочность (глобально)
    urgencies = db.session.query(Analysis.urgency, func.count()).group_by(Analysis.urgency).all()
    urgency_stats = {u or 'неопределено': c for u, c in urgencies}
    
    # Категории (глобально)
    categories = db.session.query(Analysis.category, func.count()).group_by(Analysis.category).all()
    category_stats = {c or 'неопределено': n for c, n in categories}
    
    # Топ ключевые слова (глобально)
    kw_rows = db.session.query(Analysis.keywords).filter(Analysis.keywords.isnot(None)).all()
    
    from collections import Counter
    all_keywords = []
    for kw_json, in kw_rows:
        try:
            all_keywords.extend(json.loads(kw_json))
        except:
            pass
    
    top_keywords = [w for w, _ in Counter(all_keywords).most_common(15)]
    
    # Активность по дням
    from collections import defaultdict
    import calendar
    
    calls_by_weekday = defaultdict(int)
    convs = Conversation.query.all()
    for conv in convs:
        wd = conv.upload_date.weekday()
        calls_by_weekday[wd] += 1
    
    weekday_labels = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    weekday_data = [calls_by_weekday.get(i, 0) for i in range(7)]
    
    return render_template('admin_analytics.html',
        total=total,
        completed=completed,
        errors=errors,
        processing=processing,
        avg_duration=avg_duration,
        user_stats=user_stats,
        sentiments=sentiment_stats,
        urgencies=urgency_stats,
        categories=category_stats,
        top_keywords=top_keywords,
        weekday_labels=weekday_labels,
        weekday_data=weekday_data
    )

app.register_blueprint(stats_bp)
app.register_blueprint(stats_user_bp)
app.register_blueprint(admin_bp)

if __name__ == '__main__':
    app.run(debug=True)