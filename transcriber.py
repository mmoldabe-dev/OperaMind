import os
import json
import math
import re
import speech_recognition as sr
from pydub import AudioSegment
from pathlib import Path
import concurrent.futures

# Опционально Whisper (faster-whisper)
_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except Exception:
    _WHISPER_AVAILABLE = False

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
            1. (Опционально) Whisper (ENABLE_WHISPER=1, если установлен faster-whisper) для максимально точной транскрипции.
            2. Затем пытаемся Vosk (если доступен и есть модель VOSK_MODEL_PATH).
            3. Если ни Whisper ни Vosk — fallback Google SpeechRecognition (chunked + эвристика).
    """
    print(f"📤 Загружаю: {filepath}")
    audio = AudioSegment.from_file(filepath)
    original_ms = len(audio)
    audio_proc = audio.set_frame_rate(16000).set_channels(1)
    duration_min = original_ms / 60000
    print(f"⏱️ Длительность: {duration_min:.1f} мин")

    # --- Попытка Whisper (если включено) ---
    enable_whisper = os.getenv('ENABLE_WHISPER', '0') == '1'
    whisper_model_name = os.getenv('WHISPER_MODEL', 'medium')
    if enable_whisper and _WHISPER_AVAILABLE:
        try:
            print(f"🟣 Whisper backend: модель={whisper_model_name}")
            # faster-whisper отдаёт уже сегменты с таймкодами и словами (если word_timestamps=True)
            model = WhisperModel(whisper_model_name, device=os.getenv('WHISPER_DEVICE','cpu'), compute_type=os.getenv('WHISPER_COMPUTE','int8'))
            segments_iter, info = model.transcribe(str(filepath), language='ru', word_timestamps=True)
            segments = []
            all_words_flat = []
            idx = 0
            for seg in segments_iter:
                words = []
                for w in seg.words or []:
                    words.append({
                        'w': w.word.strip(),
                        'start': round(w.start,3),
                        'end': round(w.end,3)
                    })
                    all_words_flat.append(w.word.strip())
                text_clean = seg.text.strip()
                if not text_clean:
                    continue
                segments.append({
                    'index': idx,
                    'speaker': 'unknown',
                    'segment_type': 'speech',
                    'start': round(seg.start,3),
                    'end': round(seg.end,3),
                    'text': text_clean,
                    'words': words,
                    'gap_from_prev': 0.0 if idx==0 else round(seg.start - segments[-1]['end'],3)
                })
                idx += 1
            # Простое чередование ролей (можно позже кластеризовать — reused logic с Vosk)
            for i, seg in enumerate(segments):
                seg['speaker'] = 'client' if i % 2 == 0 else 'operator'
            flat_text = ' '.join(all_words_flat).strip()
            structured = {
                'text': flat_text,
                'segments': segments,
                'meta': {
                    'duration_sec': round(original_ms / 1000.0, 3),
                    'method': 'whisper',
                    'model': whisper_model_name
                }
            }
            print(f"✅ Whisper готово: {len(flat_text)} символов, сегментов: {len(segments)}")
            return json.dumps(structured, ensure_ascii=False)
        except Exception as we:
            print(f"⚠️ Whisper не удалось: {we}. Пробую Vosk/Google")

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

            # Сегментация по паузам > 0.8s + вставка событий, если включено FULL_AUDIO_EVENTS
            pause_threshold = float(os.getenv('PAUSE_THRESHOLD', '0.8'))
            full_audio_events = os.getenv('FULL_AUDIO_EVENTS', '0') == '1'
            grouped = []  # список сегментов речи (слов), events будет позже
            current = [raw_words[0]]
            gaps = []  # [(gap_start, gap_end, gap_duration)]
            for prev, cur in zip(raw_words, raw_words[1:]):
                gap = cur['start'] - prev['end']
                if gap > pause_threshold:
                    grouped.append(current)
                    gaps.append((prev['end'], cur['start'], gap))
                    current = [cur]
                else:
                    current.append(cur)
            if current:
                grouped.append(current)

            # Извлечение текстов сегментов
            segments = []
            full_words_flat = []
            seg_counter = 0
            def add_speech_segment(g):
                nonlocal seg_counter
                start_t = g[0]['start']
                end_t = g[-1]['end']
                text_seg = ' '.join(w['w'] for w in g)
                full_words_flat.extend(w['w'] for w in g)
                segments.append({
                    'index': seg_counter,
                    'speaker': 'unknown',
                    'segment_type': 'speech',
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
                    'gap_from_prev': 0.0 if seg_counter == 0 else round(start_t - segments[-1]['end'], 3)
                })
                seg_counter += 1
            def add_gap_segment(gap_start, gap_end, gap_duration):
                nonlocal seg_counter
                if not full_audio_events:
                    return
                segments.append({
                    'index': seg_counter,
                    'speaker': 'none',
                    'segment_type': 'silence',
                    'start': round(gap_start, 3),
                    'end': round(gap_end, 3),
                    'text': f'[silence {gap_duration:.2f}s]',
                    'words': [],
                    'gap_from_prev': 0.0 if seg_counter == 0 else round(gap_start - segments[-1]['end'], 3)
                })
                seg_counter += 1

            # Перебираем группы и межгрупповые паузы
            for i, g in enumerate(grouped):
                add_speech_segment(g)
                if i < len(gaps):
                    gs, ge, gd = gaps[i]
                    add_gap_segment(gs, ge, gd)

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

    # --- Fallback Google (с опцией chunked) ---
    try:
        enable_chunked = os.getenv('ENABLE_CHUNKED_GOOGLE', '1') == '1'
        chunk_seconds = float(os.getenv('GOOGLE_CHUNK_SECONDS', '10'))  # Уменьшил для лучшего качества
        chunk_overlap = float(os.getenv('GOOGLE_CHUNK_OVERLAP', '3'))   # Увеличил overlap
        full_audio_events = os.getenv('FULL_AUDIO_EVENTS', '0') == '1'
        detect_silence = os.getenv('DETECT_SILENCE_GOOGLE', '1') == '1'

        recognizer = sr.Recognizer()
        temp_wav = "temp_transcribe.wav"
        audio_proc.export(temp_wav, format="wav")

        from wave import open as wave_open
        wf = wave_open(temp_wav, 'rb')
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()
        total_duration = total_frames / sample_rate
        wf.close()

        words_master = []  # [{'w':text,'start':s,'end':e}]
        raw_segments = []  # предварительные распознанные куски

        def recognize_chunk(idx_chunk, ms_start, ms_end, start_pos, end_pos):
            piece = audio_proc[ms_start:ms_end]
            piece_wav = f"temp_chunk_{idx_chunk}.wav"
            piece.export(piece_wav, format='wav')
            local_recognizer = sr.Recognizer()
            with sr.AudioFile(piece_wav) as source:
                local_recognizer.adjust_for_ambient_noise(source, duration=0.1)
                local_recognizer.energy_threshold = 300
                local_recognizer.dynamic_energy_threshold = False
                local_recognizer.pause_threshold = 0.3
                audio_data = local_recognizer.record(source)
                try:
                    chunk_text = local_recognizer.recognize_google(audio_data, language='ru-RU', show_all=False)
                except sr.RequestError:
                    try:
                        local_recognizer.energy_threshold = 500
                        local_recognizer.pause_threshold = 0.8
                        chunk_text = local_recognizer.recognize_google(audio_data, language='ru-RU')
                    except Exception:
                        chunk_text = ''
                except sr.UnknownValueError:
                    chunk_text = '[неразборчиво]'
                except Exception:
                    chunk_text = ''
            os.remove(piece_wav)
            return {
                'idx': idx_chunk,
                'start': start_pos,
                'end': end_pos,
                'text': chunk_text.strip()
            }

        if enable_chunked and total_duration > chunk_seconds:
            print(f"🔁 Chunked Google SR: total={total_duration:.1f}s chunk={chunk_seconds}s overlap={chunk_overlap}s")
            step = chunk_seconds - chunk_overlap
            if step <= 0:
                print("⚠️ Некорректные параметры: OVERLAP >= CHUNK. Корректирую шаг.")
                step = max(chunk_seconds * 0.8, 1.0)
            start_pos = 0.0
            idx_chunk = 0
            max_iter = int(math.ceil(total_duration / step) + 5)
            iter_count = 0
            chunk_jobs = []
            while start_pos < total_duration and iter_count < max_iter:
                end_pos = min(start_pos + chunk_seconds, total_duration)
                ms_start = int(start_pos * 1000)
                ms_end = int(end_pos * 1000)
                chunk_jobs.append((idx_chunk, ms_start, ms_end, start_pos, end_pos))
                if end_pos >= total_duration:
                    break
                start_pos += step
                idx_chunk += 1
                iter_count += 1
            if iter_count >= max_iter:
                print("⚠️ Достигнут предохранительный лимит итераций chunked цикла — возможна аномалия параметров.")
            # Параллельная обработка чанков
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(chunk_jobs))) as executor:
                future_to_idx = {executor.submit(recognize_chunk, *job): job[0] for job in chunk_jobs}
                for future in concurrent.futures.as_completed(future_to_idx):
                    res = future.result()
                    results.append(res)
            # Собрать в исходном порядке
            results.sort(key=lambda x: x['idx'])
            for res in results:
                if res['text']:
                    raw_segments.append({'start': res['start'], 'end': res['end'], 'text': res['text']})
        else:
            with sr.AudioFile(temp_wav) as source:
                # Более агрессивная настройка для сохранения слов
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                recognizer.energy_threshold = 300
                recognizer.dynamic_energy_threshold = False
                recognizer.pause_threshold = 0.3
                audio_data = recognizer.record(source)
                whole_text = ''
                try:
                    whole_text = recognizer.recognize_google(audio_data, language='ru-RU', show_all=False)
                except Exception as ce:
                    print(f"❌ Google SR ошибка: {ce}")
                whole_text = whole_text.strip()
            if whole_text:
                raw_segments.append({'start': 0.0, 'end': total_duration, 'text': whole_text})

        os.remove(temp_wav)

        if not raw_segments:
            raise RuntimeError('Google SR не вернул текст')

        # Дедупликация overlap между chunk'ами
        deduped_segments = []
        if len(raw_segments) > 1:
            deduped_segments.append(raw_segments[0])
            for i in range(1, len(raw_segments)):
                prev_text = raw_segments[i-1]['text'].strip()
                curr_text = raw_segments[i]['text'].strip()
                
                # Ищем overlap последних слов предыдущего и первых текущего
                prev_words = prev_text.split()
                curr_words = curr_text.split()
                
                overlap_len = 0
                for j in range(1, min(len(prev_words), len(curr_words)) + 1):
                    if prev_words[-j:] == curr_words[:j]:
                        overlap_len = j
                
                # Удаляем overlap из текущего chunk'а
                if overlap_len > 0:
                    dedupe_curr = ' '.join(curr_words[overlap_len:])
                    print(f"🔄 Удален overlap {overlap_len} слов: {' '.join(curr_words[:overlap_len])}")
                else:
                    dedupe_curr = curr_text
                
                if dedupe_curr.strip():
                    deduped_segments.append({
                        'start': raw_segments[i]['start'],
                        'end': raw_segments[i]['end'], 
                        'text': dedupe_curr
                    })
        else:
            deduped_segments = raw_segments

        # Склейка текста после дедупликации
        full_text = ' '.join(seg['text'] for seg in deduped_segments).strip()

        # Разбиение на сегменты с улучшенным распределением времени
        final_segments = []
        idx = 0
        for rs in deduped_segments:
            # Разбиваем на предложения внутри кусочка для лучшего time mapping
            local_sents = _split_sentences(rs['text']) or [rs['text']]
            span = rs['end'] - rs['start']
            total_chars = sum(len(s) for s in local_sents) or 1
            cursor = rs['start']
            for s_text in local_sents:
                portion = len(s_text) / total_chars
                seg_duration = portion * span
                seg_start = cursor
                seg_end = seg_start + seg_duration
                words = _approx_words(s_text, seg_start, seg_end)
                final_segments.append({
                    'index': idx,
                    'speaker': 'unknown',
                    'segment_type': 'speech',
                    'start': round(seg_start,3),
                    'end': round(seg_end,3),
                    'text': s_text,
                    'words': words,
                    'gap_from_prev': 0.0 if idx==0 else round(seg_start - final_segments[-1]['end'],3)
                })
                idx += 1
                cursor = seg_end

        # Вставка сегментов тишины между предложениями если включено
        if full_audio_events and detect_silence:
            enriched = []
            for i, seg in enumerate(final_segments):
                enriched.append(seg)
                if i < len(final_segments)-1:
                    gap = final_segments[i+1]['start'] - seg['end']
                    if gap > 0.5:  # порог фиксированный для fallback
                        enriched.append({
                            'index': idx,
                            'speaker': 'none',
                            'segment_type': 'silence',
                            'start': seg['end'],
                            'end': final_segments[i+1]['start'],
                            'text': f'[silence {gap:.2f}s]',
                            'words': [],
                            'gap_from_prev': round(seg['end'] - seg['end'],3)
                        })
                        idx += 1
            final_segments = enriched

        # Простое чередование ролей (или оставить unknown)
        for i, seg in enumerate(final_segments):
            if seg['speaker'] == 'unknown':
                seg['speaker'] = 'client' if i % 2 == 0 else 'operator'

        structured = {
            'text': full_text,
            'segments': final_segments,
            'meta': {
                'duration_sec': round(original_ms / 1000.0, 3),
                'method': 'google_speech_chunked' if enable_chunked else 'google_speech_single',
                'chunked': enable_chunked,
                'chunk_seconds': chunk_seconds,
                'overlap_seconds': chunk_overlap,
                'full_audio_events': full_audio_events
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