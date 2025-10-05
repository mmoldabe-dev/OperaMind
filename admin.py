from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from sqlalchemy import func
from models import db, User, Conversation, Analysis, UserRole

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/admin/users')
@login_required
def admin_users():
    if not getattr(current_user, 'is_admin', False):
        flash('Нет доступа', 'error')
        return redirect(url_for('index'))
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users)

@admin_bp.route('/admin/users', methods=['POST'])
@login_required
def admin_users_update():
    if not getattr(current_user, 'is_admin', False):
        flash('Нет доступа', 'error')
        return redirect(url_for('index'))
    user_id = request.form.get('user_id')
    role = request.form.get('role')
    is_active = request.form.get('is_active') == '1'
    user = User.query.filter_by(id=user_id).first()
    if not user:
        flash('Пользователь не найден', 'error')
        return redirect(url_for('admin.admin_users'))
    if user.id == current_user.id:
        flash('Нельзя менять собственную запись здесь', 'warning')
        return redirect(url_for('admin.admin_users'))
    # Обновление роли
    try:
        if role in ['USER', 'OPERATOR', 'ADMIN']:
            # Сопоставление с Enum
            mapping = {
                'USER': UserRole.USER,
                'OPERATOR': UserRole.OPERATOR,
                'ADMIN': UserRole.ADMIN
            }
            user.role = mapping.get(role, user.role)
            # Для обратной совместимости со старым is_admin
            user.is_admin = (role == 'ADMIN')
        user.is_active = is_active
        db.session.commit()
        flash('Изменения сохранены', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Ошибка сохранения: {e}', 'error')
    return redirect(url_for('admin.admin_users'))

@admin_bp.route('/admin/user/create', methods=['POST'])
@login_required
def admin_user_create():
    if not getattr(current_user, 'is_admin', False):
        flash('Нет доступа', 'error')
        return redirect(url_for('index'))
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    email = request.form.get('email', '').strip() or None
    role = request.form.get('role', 'OPERATOR')
    if not username or not password:
        flash('Логин и пароль обязательны', 'error')
        return redirect(url_for('admin.admin_users'))
    if User.query.filter_by(username=username).first():
        flash('Пользователь с таким логином уже существует', 'error')
        return redirect(url_for('admin.admin_users'))
    try:
        mapping = {'USER': UserRole.USER, 'OPERATOR': UserRole.OPERATOR, 'ADMIN': UserRole.ADMIN}
        new_user = User(username=username, email=email, role=mapping.get(role, UserRole.OPERATOR), is_admin=(role=='ADMIN'))
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Пользователь создан', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Ошибка создания: {e}', 'error')
    return redirect(url_for('admin.admin_users'))

@admin_bp.route('/admin/user/<int:user_id>/history')
@login_required
def admin_user_history(user_id):
    if not getattr(current_user, 'is_admin', False):
        flash('Нет доступа', 'error')
        return redirect(url_for('index'))
    user = User.query.get_or_404(user_id)
    conversations = Conversation.query.filter_by(user_id=user.id).order_by(Conversation.upload_date.desc()).all()
    return render_template('admin_user_history.html', target_user=user, conversations=conversations)

@admin_bp.route('/admin/user/<int:user_id>/stats')
@login_required
def admin_user_stats(user_id):
    if not getattr(current_user, 'is_admin', False):
        flash('Нет доступа', 'error')
        return redirect(url_for('index'))
    user = User.query.get_or_404(user_id)
    # Агрегаты по пользователю
    total = Conversation.query.filter_by(user_id=user.id).count()
    completed = Conversation.query.filter_by(user_id=user.id, status='completed').count()
    errors = Conversation.query.filter_by(user_id=user.id, status='error').count()
    avg_duration = db.session.query(func.avg(Conversation.duration)).filter_by(user_id=user.id).scalar() or 0
    # Тональность
    sentiment_rows = db.session.query(Analysis.sentiment, func.count()).join(Conversation).filter(Conversation.user_id == user.id).group_by(Analysis.sentiment).all()
    sentiments = {s or 'unknown': c for s, c in sentiment_rows}
    # Срочность
    urgency_rows = db.session.query(Analysis.urgency, func.count()).join(Conversation).filter(Conversation.user_id == user.id).group_by(Analysis.urgency).all()
    urgencies = {u or 'unknown': c for u, c in urgency_rows}
    # Ключевые слова
    kw_rows = db.session.query(Analysis.keywords).join(Conversation).filter(Conversation.user_id == user.id).all()
    from collections import Counter
    import json as _json
    all_kw = []
    for kw_json, in kw_rows:
        try:
            all_kw.extend(_json.loads(kw_json))
        except Exception:
            pass
    top_keywords = [w for w, _ in Counter(all_kw).most_common(10)]
    return render_template('admin_user_stats.html', target_user=user,
                           total=total, completed=completed, errors=errors,
                           avg_duration=avg_duration, sentiments=sentiments,
                           urgencies=urgencies, top_keywords=top_keywords)

@admin_bp.route('/admin/history')
@login_required
def admin_history():
    if not getattr(current_user, 'is_admin', False):
        flash('Нет доступа', 'error')
        return redirect(url_for('index'))
    username = request.args.get('username', '').strip()
    query = Conversation.query.join(User)
    if username:
        query = query.filter(User.username.ilike(f"%{username}%"))
    conversations = query.order_by(Conversation.upload_date.desc()).all()

    # Считаем реальные данные для графика по дням недели
    from collections import Counter
    import calendar
    calls_by_weekday = Counter()
    for conv in conversations:
        wd = conv.upload_date.weekday()  # 0=Пн
        calls_by_weekday[wd] += 1
    labels = [calendar.day_abbr[i] for i in range(7)]
    data = [calls_by_weekday.get(i, 0) for i in range(7)]

    return render_template('admin_history.html', conversations=conversations, chart_labels=labels, chart_data=data)
