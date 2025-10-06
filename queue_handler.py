"""
Система очередей для фоновой обработки аудио
Использует threading для параллельной обработки
"""

import threading
import queue
import time
from datetime import datetime
from models import db, Conversation, Analysis
from transcriber import transcribe_audio_file, transcribe_from_text
from analyzer import analyze_transcript
import json
import os

# Глобальная очередь задач
task_queue = queue.Queue()
processing_threads = []
queue_lock = threading.Lock()

# Статистика
queue_stats = {
    'total_processed': 0,
    'total_errors': 0,
    'currently_processing': 0,
    'queue_size': 0
}


class ProcessingTask:
    """Задача на обработку"""
    def __init__(self, conversation_id, filepath, filename, is_text=False):
        self.conversation_id = conversation_id
        self.filepath = filepath
        self.filename = filename
        self.is_text = is_text
        self.created_at = datetime.utcnow()
        self.status = 'queued'
        self.error = None


def process_task(task, app):
    """Обработка одной задачи"""
    with app.app_context():
        try:
            print(f"\n{'='*70}")
            print(f"🔄 ОБРАБОТКА: {task.filename}")
            print(f"{'='*70}")
            
            conversation = db.session.get(Conversation, task.conversation_id)
            if not conversation:
                raise Exception("Разговор не найден в БД")
            
            # ЭТАП 1: ТРАНСКРИПЦИЯ
            print("\n📝 ЭТАП 1: ТРАНСКРИПЦИЯ")
            conversation.status = 'transcribing'
            db.session.commit()
            
            if task.is_text:
                with open(task.filepath, 'r', encoding='utf-8') as f:
                    structured_transcript = transcribe_from_text(f.read())
                print("   ✓ Текст загружен")
            else:
                structured_transcript = transcribe_audio_file(task.filepath)
            
            if not structured_transcript:
                raise Exception("Транскрипция не выполнена")
            
            # Извлекаем длительность
            try:
                data_tr = json.loads(structured_transcript)
                if isinstance(data_tr, dict) and 'meta' in data_tr:
                    conversation.duration = int(data_tr['meta'].get('duration_sec', 0))
            except:
                pass
            
            # Извлекаем плоский текст
            flat_text = None
            try:
                data_tr = json.loads(structured_transcript)
                if isinstance(data_tr, dict) and 'text' in data_tr:
                    flat_text = data_tr.get('text')
                else:
                    flat_text = structured_transcript
            except:
                flat_text = structured_transcript
            
            # ЭТАП 2: АНАЛИЗ
            print("\n🤖 ЭТАП 2: АНАЛИЗ")
            conversation.status = 'analyzing'
            db.session.commit()
            
            analysis_result = analyze_transcript(flat_text)
            
            # Сохранение
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
            
            print(f"\n✅ ЗАВЕРШЕНО: {task.filename}")
            print(f"{'='*70}\n")
            
            with queue_lock:
                queue_stats['total_processed'] += 1
            
        except Exception as e:
            print(f"\n❌ ОШИБКА: {task.filename}")
            print(f"   {str(e)}")
            print(f"{'='*70}\n")
            
            with app.app_context():
                conversation = db.session.get(Conversation, task.conversation_id)
                if conversation:
                    conversation.status = 'error'
                    db.session.commit()
            
            task.error = str(e)
            with queue_lock:
                queue_stats['total_errors'] += 1


def worker_thread(app):
    """Рабочий поток для обработки очереди"""
    print("🔵 Рабочий поток запущен")
    
    while True:
        try:
            # Ждем задачу из очереди
            task = task_queue.get(timeout=1)
            
            with queue_lock:
                queue_stats['currently_processing'] += 1
                queue_stats['queue_size'] = task_queue.qsize()
            
            # Обрабатываем
            process_task(task, app)
            
            # Завершаем
            task_queue.task_done()
            
            with queue_lock:
                queue_stats['currently_processing'] -= 1
                queue_stats['queue_size'] = task_queue.qsize()
                
        except queue.Empty:
            # Очередь пуста, ждем
            with queue_lock:
                queue_stats['queue_size'] = task_queue.qsize()
            continue
        except Exception as e:
            print(f"❌ Ошибка в рабочем потоке: {e}")
            with queue_lock:
                queue_stats['currently_processing'] -= 1


def init_queue_system(app, num_workers=None):
    """Инициализация системы очередей"""
    if num_workers is None:
        # По умолчанию 2 параллельных обработчика
        num_workers = int(os.getenv('QUEUE_WORKERS', '2'))
    
    print(f"\n{'='*70}")
    print(f"🚀 ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ОЧЕРЕДЕЙ")
    print(f"{'='*70}")
    print(f"   Параллельных обработчиков: {num_workers}")
    print(f"{'='*70}\n")
    
    # Создаем рабочие потоки
    for i in range(num_workers):
        thread = threading.Thread(
            target=worker_thread,
            args=(app,),
            daemon=True,
            name=f"QueueWorker-{i+1}"
        )
        thread.start()
        processing_threads.append(thread)
    
    print(f"✅ Система очередей запущена: {num_workers} потоков\n")


def add_to_queue(conversation_id, filepath, filename, is_text=False):
    """Добавить задачу в очередь"""
    task = ProcessingTask(conversation_id, filepath, filename, is_text)
    task_queue.put(task)
    
    with queue_lock:
        queue_stats['queue_size'] = task_queue.qsize()
    
    print(f"➕ Добавлено в очередь: {filename} (позиция: {task_queue.qsize()})")
    return task


def get_queue_status():
    """Получить статус очереди"""
    with queue_lock:
        return {
            'queue_size': task_queue.qsize(),
            'currently_processing': queue_stats['currently_processing'],
            'total_processed': queue_stats['total_processed'],
            'total_errors': queue_stats['total_errors'],
            'workers_active': len([t for t in processing_threads if t.is_alive()])
        }


def wait_for_queue():
    """Дождаться завершения всех задач"""
    task_queue.join()