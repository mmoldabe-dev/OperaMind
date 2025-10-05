from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from models import db, Conversation, Analysis
from sqlalchemy import func
from datetime import datetime, timedelta

stats_user_bp = Blueprint('stats_user', __name__)

@stats_user_bp.route('/api/stats/user/<int:user_id>')
@login_required
def stats_user(user_id):
    # Только для себя или админа (расширить при необходимости)
    if user_id != current_user.id:
        return jsonify({'error': 'Нет доступа'}), 403
    # Количество звонков
    total_calls = Conversation.query.filter_by(user_id=user_id).count()
    # Средняя длительность
    avg_duration = db.session.query(func.avg(Conversation.duration)).filter_by(user_id=user_id).scalar() or 0
    # По дням
    today = datetime.utcnow().date()
    days = [today - timedelta(days=i) for i in range(13, -1, -1)]
    calls_by_day = {d.strftime('%Y-%m-%d'): 0 for d in days}
    rows = db.session.query(func.date(Conversation.upload_date), func.count()).filter(
        Conversation.user_id == user_id,
        Conversation.upload_date >= days[0]
    ).group_by(func.date(Conversation.upload_date)).all()
    for d, cnt in rows:
        calls_by_day[str(d)] = cnt
    # Тональность
    sentiments = db.session.query(Analysis.sentiment, func.count()).join(Conversation).filter(
        Conversation.user_id == user_id
    ).group_by(Analysis.sentiment).all()
    sentiment_stats = {s or 'unknown': c for s, c in sentiments}
    # Категории
    categories = db.session.query(Analysis.category, func.count()).join(Conversation).filter(
        Conversation.user_id == user_id
    ).group_by(Analysis.category).all()
    category_stats = {c or 'unknown': n for c, n in categories}
    # ТОП-5 ключевых слов
    keywords = db.session.query(Analysis.keywords).join(Conversation).filter(
        Conversation.user_id == user_id
    ).all()
    from collections import Counter
    all_words = []
    for kw_json, in keywords:
        try:
            import json as _json
            all_words.extend(_json.loads(kw_json))
        except Exception:
            pass
    top_keywords = [w for w, _ in Counter(all_words).most_common(5)]
    # Средняя оценка оператора
    op_qualities = db.session.query(Analysis.operator_quality, func.count()).join(Conversation).filter(
        Conversation.user_id == user_id
    ).group_by(Analysis.operator_quality).all()
    op_quality_stats = {q or 'unknown': n for q, n in op_qualities}
    return jsonify({
        'total_calls': total_calls,
        'avg_duration_sec': round(avg_duration, 1),
        'calls_by_day': calls_by_day,
        'sentiments': sentiment_stats,
        'categories': category_stats,
        'top_keywords': top_keywords,
        'operator_quality': op_quality_stats
    })
