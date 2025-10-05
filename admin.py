from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from models import db, User, Conversation

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/admin/users')
@login_required
def admin_users():
    if not getattr(current_user, 'is_admin', False):
        flash('Нет доступа', 'error')
        return redirect(url_for('index'))
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users)

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
