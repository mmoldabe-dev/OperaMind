from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from models import db, Conversation, Analysis
from sqlalchemy import func
from datetime import datetime, timedelta

stats_bp = Blueprint('stats', __name__)

@stats_bp.route('/api/stats/summary')
@login_required
def stats_summary():
    # Количество звонков
    total_calls = Conversation.query.filter_by(user_id=current_user.id).count()
    # Средняя длительность (сек)
    avg_duration = db.session.query(func.avg(Conversation.duration)).filter_by(user_id=current_user.id).scalar() or 0
    # Количество звонков по дням за последние 14 дней
    today = datetime.utcnow().date()
    days = [today - timedelta(days=i) for i in range(13, -1, -1)]
    calls_by_day = {d.strftime('%Y-%m-%d'): 0 for d in days}
    rows = db.session.query(func.date(Conversation.upload_date), func.count()).filter(
        Conversation.user_id == current_user.id,
        Conversation.upload_date >= days[0]
    ).group_by(func.date(Conversation.upload_date)).all()
    for d, cnt in rows:
        calls_by_day[str(d)] = cnt
    # Тональность
    sentiments = db.session.query(Analysis.sentiment, func.count()).join(Conversation).filter(
        Conversation.user_id == current_user.id
    ).group_by(Analysis.sentiment).all()
    sentiment_stats = {s or 'unknown': c for s, c in sentiments}
    # Категории
    categories = db.session.query(Analysis.category, func.count()).join(Conversation).filter(
        Conversation.user_id == current_user.id
    ).group_by(Analysis.category).all()
    category_stats = {c or 'unknown': n for c, n in categories}
    return jsonify({
        'total_calls': total_calls,
        'avg_duration_sec': round(avg_duration, 1),
        'calls_by_day': calls_by_day,
        'sentiments': sentiment_stats,
        'categories': category_stats
    })
