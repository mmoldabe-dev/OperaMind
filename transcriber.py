import os
import json
import math
import re
import speech_recognition as sr
from pydub import AudioSegment
from pathlib import Path

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    _VOSK_AVAILABLE = True
except Exception:
    _VOSK_AVAILABLE = False

# Опциональные зависимости для кластеризации
try:
    import numpy as np
    import librosa
    from sklearn.cluster import KMeans
    _CLUSTER_AVAILABLE = True
except Exception:
    _CLUSTER_AVAILABLE = False


def _split_sentences(text: str):
    # Простейшее разбиение на предложения
    raw = re.split(r'([.!?])', text)
    sentences = []
    current = ''
    for part in raw:
        if not part:
            continue
        current += part
        if part in '.!?':
            cleaned = current.strip()
            if cleaned:
                sentences.append(cleaned)
            current = ''
    # Хвост
    tail = current.strip()
    if tail:
        sentences.append(tail)
    return [s for s in (seg.strip() for seg in sentences) if s]


def _approx_words(sentence_text: str, start_time: float, end_time: float):
    words = [w for w in re.split(r'\s+', sentence_text.strip()) if w]
    if not words:
        return []
    duration = max(end_time - start_time, 0.0001)
    per = duration / len(words)
    out = []
    t = start_time
    for w in words:
        out.append({
            'w': w,
            'start': round(t, 3),
            'end': round(t + per, 3)
        })
        t += per
    # Коррекция последнего
    if out:
        out[-1]['end'] = round(end_time, 3)
    return out


def transcribe_audio_file(filepath: str):
    """Расширенная транскрипция с попыткой Vosk.

    Алгоритм:
      1. Пытаемся использовать Vosk (если доступен и есть модель VOSK_MODEL_PATH).
      2. Если Vosk недоступен — fallback на Google SpeechRecognition (цельный текст + эвристика).
    """
    print(f"📤 Загружаю: {filepath}")
    audio = AudioSegment.from_file(filepath)
    original_ms = len(audio)
    audio_proc = audio.set_frame_rate(16000).set_channels(1)
    duration_min = original_ms / 60000
    print(f"⏱️ Длительность: {duration_min:.1f} мин")

    # --- Попытка Vosk ---
    model_path = os.getenv('VOSK_MODEL_PATH')
    if _VOSK_AVAILABLE and model_path and Path(model_path).exists():
        try:
            print("🔊 Использую Vosk модель")
            temp_wav = "temp_vosk.wav"
            audio_proc.export(temp_wav, format="wav")

            from wave import open as wave_open
            import wave, json as _json
            wf = wave_open(temp_wav, "rb")
            rec = KaldiRecognizer(VoskModel(model_path), wf.getframerate())
            rec.SetWords(True)

            import json as _json2
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    results.append(_json2.loads(rec.Result()))
            results.append(_json2.loads(rec.FinalResult()))
            wf.close()
            os.remove(temp_wav)

            # Собираем ВСЕ слова в один список, не теряя filler words
            raw_words = []  # [{'word':, 'start':, 'end':}
            for res in results:
                if isinstance(res, dict):
                    for w in res.get('result', []) or []:
                        if 'word' in w:
                            raw_words.append({
                                'w': w.get('word'),
                                'start': float(w.get('start', 0.0)),
                                'end': float(w.get('end', 0.0))
                            })

            if not raw_words:
                raise RuntimeError("Пустой результат Vosk")

            # Сегментация по паузам > 0.8s
            pause_threshold = float(os.getenv('PAUSE_THRESHOLD', '0.8'))
            grouped = []  # список сегментов с полями words, start, end
            current = [raw_words[0]]
            for prev, cur in zip(raw_words, raw_words[1:]):
                gap = cur['start'] - prev['end']
                if gap > pause_threshold:
                    grouped.append(current)
                    current = [cur]
                else:
                    current.append(cur)
            if current:
                grouped.append(current)

            # Извлечение текстов сегментов
            segments = []
            full_words_flat = []
            for idx, g in enumerate(grouped):
                start_t = g[0]['start']
                end_t = g[-1]['end']
                text_seg = ' '.join(w['w'] for w in g)
                full_words_flat.extend(w['w'] for w in g)
                segments.append({
                    'index': idx,
                    'speaker': 'unknown',  # временно, определим позже
                    'start': round(start_t, 3),
                    'end': round(end_t, 3),
                    'text': text_seg,
                    'words': [
                        {
                            'w': w['w'],
                            'start': round(w['start'], 3),
                            'end': round(w['end'], 3)
                        } for w in g
                    ],
                    'gap_from_prev': 0.0 if idx == 0 else round(start_t - segments[-1]['end'], 3)
                })

            # Кластеризация спикеров (если включено и доступны зависимости)
            clustering_enabled = os.getenv('SPEAKER_CLUSTERING', '1') == '1'
            clustering_info = {}
            if clustering_enabled and _CLUSTER_AVAILABLE:
                try:
                    print("🧪 Кластеризация спикеров (MFCC + KMeans)...")
                    # Загружаем аудио одной волной для индексации
                    # ВАЖНО: sample_rate, чтобы не перекрыть модуль speech_recognition (sr)
                    y, sample_rate = librosa.load(filepath, sr=16000, mono=True)
                    feat_vectors = []
                    for seg in segments:
                        s = max(int(seg['start'] * sample_rate), 0)
                        e = min(int(seg['end'] * sample_rate), len(y))
                        chunk = y[s:e]
                        if len(chunk) < sample_rate * 0.1:  # слишком короткий
                            chunk = np.pad(chunk, (0, int(sample_rate*0.1)-len(chunk))) if len(chunk) > 0 else np.zeros(int(sample_rate*0.1))
                        mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=13)
                        vec = np.mean(mfcc, axis=1)
                        feat_vectors.append(vec)
                    X = np.stack(feat_vectors, axis=0)
                    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                    labels = kmeans.fit_predict(X)
                    # Эвристика: выберем кластер с сегментом, содержащим приветствие, как оператор
                    greeting_words = {"здравствуйте","добрый","привет","оператор","чем","могу"}
                    cluster_scores = {0:0,1:0}
                    for lab, seg in zip(labels, segments):
                        tokens = set(seg['text'].lower().split())
                        if tokens & greeting_words:
                            cluster_scores[lab] += 1
                    if cluster_scores[0] == cluster_scores[1]:
                        # тайбрейк: кластер с более ранним первым сегментом -> оператор
                        first0 = min(seg['start'] for seg,l in zip(segments, labels) if l==0)
                        first1 = min(seg['start'] for seg,l in zip(segments, labels) if l==1)
                        operator_cluster = 0 if first0 <= first1 else 1
                    else:
                        operator_cluster = 0 if cluster_scores[0] > cluster_scores[1] else 1
                    # Присваиваем роли
                    for seg, lab in zip(segments, labels):
                        seg['speaker'] = 'operator' if lab == operator_cluster else 'client'
                    clustering_info = {
                        'method': 'kmeans-mfcc',
                        'operator_cluster': int(operator_cluster),
                        'cluster_scores': cluster_scores
                    }
                    print("✅ Кластеризация завершена")
                except Exception as ce:
                    print(f"⚠️ Кластеризация не удалась: {ce}. Использую чередование.")
                    for idx, seg in enumerate(segments):
                        seg['speaker'] = 'client' if idx % 2 == 0 else 'operator'
                    clustering_info = {
                        'method': 'fallback-alternation',
                        'error': str(ce)[:120]
                    }
            else:
                # Fallback чередование
                for idx, seg in enumerate(segments):
                    seg['speaker'] = 'client' if idx % 2 == 0 else 'operator'
                clustering_info = {
                    'method': 'alternation',
                    'note': 'Clustering off or dependencies missing'
                }

            flat_text = ' '.join(full_words_flat).strip()
            structured = {
                'text': flat_text,
                'segments': segments,
                'meta': {
                    'duration_sec': round(original_ms / 1000.0, 3),
                    'method': 'vosk',
                    'pause_threshold': pause_threshold,
                    'speaker_clustering': clustering_info
                }
            }
            print(f"✅ Vosk готово: {len(flat_text)} символов, сегментов: {len(segments)}")
            return json.dumps(structured, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Vosk не удалось: {e}. Перехожу к Google SR")

    # --- Fallback Google ---
    try:
        recognizer = sr.Recognizer()
        temp_wav = "temp_transcribe.wav"
        audio_proc.export(temp_wav, format="wav")
        with sr.AudioFile(temp_wav) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            full_text = recognizer.recognize_google(audio_data, language='ru-RU')
        os.remove(temp_wav)
        full_text = full_text.strip()
        print(f"✅ Google SR готово: {len(full_text)} символов")
        sentences = _split_sentences(full_text) or [full_text]
        total_chars = sum(len(s) for s in sentences) or 1
        current_start = 0.0
        segments = []
        for idx, sent in enumerate(sentences):
            proportion = len(sent) / total_chars
            seg_duration = proportion * (original_ms / 1000.0)
            start = current_start
            end = start + seg_duration
            speaker = 'client' if idx % 2 == 0 else 'operator'
            words = _approx_words(sent, start, end)
            gap_from_prev = 0.0 if idx == 0 else round(start - segments[-1]['end'], 3)
            segments.append({
                'index': idx,
                'speaker': speaker,
                'start': round(start, 3),
                'end': round(end, 3),
                'text': sent,
                'words': words,
                'gap_from_prev': gap_from_prev
            })
            current_start = end
        structured = {
            'text': full_text,
            'segments': segments,
            'meta': {
                'duration_sec': round(original_ms / 1000.0, 3),
                'method': 'google_speech + heuristic segmentation',
                'note': 'Роли чередуются; подключите VOSK_MODEL_PATH для Vosk'
            }
        }
        return json.dumps(structured, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Ошибка Google SR: {e}")
        return None


def transcribe_from_text(text: str):
    """Оборачиваем текст в ту же JSON-структуру для унификации."""
    cleaned = text.strip()
    sentences = _split_sentences(cleaned) or [cleaned]
    total_chars = sum(len(s) for s in sentences) or 1
    # Простая равномерная шкала времени
    total_duration = len(cleaned.split()) * 0.4  # 0.4 сек на слово эвристика
    current = 0.0
    segments = []
    for idx, sent in enumerate(sentences):
        proportion = len(sent) / total_chars
        seg_duration = proportion * total_duration
        start = current
        end = start + seg_duration
        words = _approx_words(sent, start, end)
        segments.append({
            'index': idx,
            'speaker': 'client' if idx % 2 == 0 else 'operator',
            'start': round(start, 3),
            'end': round(end, 3),
            'text': sent,
            'words': words,
            'gap_from_prev': 0.0 if idx == 0 else round(start - segments[-1]['end'], 3)
        })
        current = end
    structured = {
        'text': cleaned,
        'segments': segments,
        'meta': {
            'duration_sec': round(total_duration, 3),
            'method': 'text import + heuristic segmentation'
        }
    }
    return json.dumps(structured, ensure_ascii=False)