import os
import json
import math
import re
import speech_recognition as sr
from pydub import AudioSegment
from pathlib import Path
import concurrent.futures
from multiprocessing import cpu_count, Pool
import time

# Whisper
_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except Exception:
    pass

# Vosk
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    _VOSK_AVAILABLE = True
except Exception:
    _VOSK_AVAILABLE = False

# Кластеризация
try:
    import numpy as np
    import librosa
    from sklearn.cluster import KMeans
    _CLUSTER_AVAILABLE = True
except Exception:
    _CLUSTER_AVAILABLE = False


# ===================== ЯЗЫКОВЫЕ НАСТРОЙКИ =====================

LANGUAGE_CONFIG = {
    'ru': {
        'name': 'Русский',
        'whisper_code': 'ru',
        'google_code': 'ru-RU',
        'vosk_model_env': 'VOSK_MODEL_PATH_RU',
        'greeting_words': {'здравствуйте', 'добрый', 'привет', 'оператор', 'чем', 'могу', 'слушаю', 'алло'}
    },
    'kk': {
        'name': 'Қазақ',
        'whisper_code': 'kk',
        'google_code': 'kk-KZ',
        'vosk_model_env': 'VOSK_MODEL_PATH_KK',
        'greeting_words': {'сәлеметсіз', 'сәлем', 'қайырлы', 'күн', 'оператор', 'көмек', 'тыңдаймын', 'алло'}
    }
}

def detect_language(text_sample):
    """Определение языка по характерным буквам и словам"""
    if not text_sample:
        return os.getenv('DEFAULT_LANGUAGE', 'ru')
    
    sample_lower = text_sample.lower()
    
    # Казахские специфичные буквы
    kk_specific = 'әіңғүұқөһ'
    kk_char_count = sum(1 for c in sample_lower if c in kk_specific)
    
    # Сильный индикатор казахского
    if kk_char_count > 2:
        return 'kk'
    
    # Ключевые слова
    kk_keywords = {'сәлем', 'қалай', 'жақсы', 'рахмет', 'кешіріңіз', 'өтінемін', 'көмектесіңіз', 'бар', 'жоқ'}
    ru_keywords = {'здравствуйте', 'привет', 'как', 'хорошо', 'спасибо', 'извините', 'пожалуйста', 'помогите', 'есть', 'нет'}
    
    kk_score = sum(1 for word in kk_keywords if word in sample_lower)
    ru_score = sum(1 for word in ru_keywords if word in sample_lower)
    
    if kk_score > ru_score:
        return 'kk'
    elif ru_score > kk_score:
        return 'ru'
    
    # По умолчанию русский
    return 'ru'


# ===================== УТИЛИТЫ =====================

def _split_sentences(text: str):
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
    tail = current.strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _approx_words(sentence_text: str, start_time: float, end_time: float):
    words = [w for w in re.split(r'\s+', sentence_text.strip()) if w]
    if not words:
        return []
    duration = max(end_time - start_time, 0.0001)
    per = duration / len(words)
    out = []
    t = start_time
    for w in words:
        out.append({'w': w, 'start': round(t, 3), 'end': round(t + per, 3)})
        t += per
    if out:
        out[-1]['end'] = round(end_time, 3)
    return out


# ===================== WHISPER (МАКСИМАЛЬНОЕ КАЧЕСТВО) =====================

def transcribe_whisper(filepath: str, audio_ms: int):
    """Whisper с максимальным качеством без компромиссов"""
    try:
        # Максимальная модель для точности
        model_name = os.getenv('WHISPER_MODEL', 'large-v3')
        device = os.getenv('WHISPER_DEVICE', 'cpu')
        
        # Для точности используем float32, не int8
        compute = 'float32' if device == 'cpu' else 'float16'
        
        print(f"🟣 Whisper MAXIMUM QUALITY: model={model_name}, compute={compute}")
        
        # Загрузка с максимальными потоками
        num_workers = cpu_count()
        model = WhisperModel(
            model_name, 
            device=device, 
            compute_type=compute,
            cpu_threads=num_workers,
            num_workers=num_workers
        )
        
        lang_mode = os.getenv('LANGUAGE_MODE', 'multi')
        
        print(f"   💪 Использую {num_workers} потоков CPU")
        
        # Максимальные параметры качества
        segments_iter, info = model.transcribe(
            str(filepath),
            language=None if lang_mode == 'multi' else LANGUAGE_CONFIG.get(lang_mode, {}).get('whisper_code', 'ru'),
            word_timestamps=True,
            beam_size=10,          # Увеличено с 5 до 10
            best_of=10,            # Увеличено с 5 до 10
            temperature=0.0,       # Детерминированный вывод
            patience=2.0,          # Терпение для beam search
            length_penalty=1.0,
            repetition_penalty=1.01,
            no_repeat_ngram_size=3,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,  # Контекст предыдущего текста
            initial_prompt="Это разговор оператора call-центра с клиентом на русском или казахском языке.",
            vad_filter=True,       # Voice Activity Detection
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                max_speech_duration_s=float('inf'),
                min_silence_duration_ms=500,
                speech_pad_ms=400
            )
        )
        
        detected_lang = getattr(info, 'language', 'ru')
        lang_prob = getattr(info, 'language_probability', 0.0)
        print(f"   ✓ Язык: {detected_lang} (вероятность: {lang_prob:.2%})")
        
        segments = []
        all_words_flat = []
        idx = 0
        
        for seg in segments_iter:
            words = []
            for w in seg.words or []:
                word_text = w.word.strip()
                if word_text:  # Пропускаем пустые
                    words.append({
                        'w': word_text,
                        'start': round(w.start, 3),
                        'end': round(w.end, 3),
                        'probability': round(getattr(w, 'probability', 1.0), 3)
                    })
                    all_words_flat.append(word_text)
            
            text_clean = seg.text.strip()
            if not text_clean:
                continue
            
            segments.append({
                'index': idx,
                'speaker': 'unknown',
                'segment_type': 'speech',
                'start': round(seg.start, 3),
                'end': round(seg.end, 3),
                'text': text_clean,
                'words': words,
                'confidence': round(getattr(seg, 'avg_logprob', 0.0), 3),
                'gap_from_prev': 0.0 if idx == 0 else round(seg.start - segments[-1]['end'], 3)
            })
            idx += 1
        
        # Умное определение спикеров с кластеризацией
        if _CLUSTER_AVAILABLE and len(segments) > 2:
            try:
                print("   🎯 Кластеризация спикеров...")
                y, sr_audio = librosa.load(filepath, sr=16000, mono=True)
                
                features = []
                valid_segments = []
                
                for seg in segments:
                    start_sample = int(seg['start'] * sr_audio)
                    end_sample = int(seg['end'] * sr_audio)
                    
                    if end_sample > start_sample and end_sample <= len(y):
                        chunk = y[start_sample:end_sample]
                        
                        if len(chunk) > sr_audio * 0.1:  # Минимум 0.1 сек
                            # Извлекаем MFCC + дополнительные признаки
                            mfcc = librosa.feature.mfcc(y=chunk, sr=sr_audio, n_mfcc=20)
                            chroma = librosa.feature.chroma_stft(y=chunk, sr=sr_audio)
                            spectral = librosa.feature.spectral_centroid(y=chunk, sr=sr_audio)
                            
                            feature_vec = np.concatenate([
                                np.mean(mfcc, axis=1),
                                np.std(mfcc, axis=1),
                                np.mean(chroma, axis=1),
                                np.mean(spectral)
                            ])
                            
                            features.append(feature_vec)
                            valid_segments.append(seg)
                
                if len(features) >= 2:
                    X = np.vstack(features)
                    kmeans = KMeans(n_clusters=2, n_init=20, random_state=42, max_iter=500)
                    labels = kmeans.fit_predict(X)
                    
                    # Определяем оператора по приветствиям
                    lang_config = LANGUAGE_CONFIG.get(detected_lang, LANGUAGE_CONFIG['ru'])
                    cluster_scores = {0: 0, 1: 0}
                    cluster_first_time = {0: float('inf'), 1: float('inf')}
                    
                    for seg, label in zip(valid_segments, labels):
                        tokens = set(seg['text'].lower().split())
                        if tokens & lang_config['greeting_words']:
                            cluster_scores[label] += 2
                        cluster_first_time[label] = min(cluster_first_time[label], seg['start'])
                    
                    # Оператор: больше приветствий ИЛИ говорит первым
                    if cluster_scores[0] == cluster_scores[1]:
                        operator_cluster = 0 if cluster_first_time[0] < cluster_first_time[1] else 1
                    else:
                        operator_cluster = 0 if cluster_scores[0] > cluster_scores[1] else 1
                    
                    for seg, label in zip(valid_segments, labels):
                        seg['speaker'] = 'operator' if label == operator_cluster else 'client'
                    
                    print(f"   ✓ Кластеризация: оператор=кластер_{operator_cluster}")
            except Exception as e:
                print(f"   ⚠️ Кластеризация не удалась: {e}")
                # Fallback: чередование
                for i, seg in enumerate(segments):
                    seg['speaker'] = 'client' if i % 2 == 0 else 'operator'
        else:
            # Простое чередование
            for i, seg in enumerate(segments):
                seg['speaker'] = 'client' if i % 2 == 0 else 'operator'
        
        flat_text = ' '.join(all_words_flat).strip()
        
        return {
            'text': flat_text,
            'segments': segments,
            'meta': {
                'duration_sec': round(audio_ms / 1000.0, 3),
                'method': 'whisper_max_quality',
                'model': model_name,
                'language': detected_lang,
                'language_probability': lang_prob,
                'cpu_threads': num_workers
            }
        }
    except Exception as e:
        print(f"⚠️ Whisper ошибка: {e}")
        return None


# ===================== VOSK (КАЧЕСТВО + СКОРОСТЬ) =====================

def transcribe_vosk(filepath: str, audio_proc: AudioSegment, audio_ms: int, detected_lang: str = 'ru'):
    """Vosk с максимальной точностью"""
    try:
        lang_config = LANGUAGE_CONFIG.get(detected_lang, LANGUAGE_CONFIG['ru'])
        model_path = os.getenv(lang_config['vosk_model_env'])
        
        if not model_path or not Path(model_path).exists():
            print(f"   ⚠️ Vosk модель для {lang_config['name']} не найдена: {model_path}")
            return None
        
        print(f"🔊 Vosk HIGH QUALITY: язык={lang_config['name']}")
        
        temp_wav = "temp_vosk.wav"
        audio_proc.export(temp_wav, format="wav")
        
        from wave import open as wave_open
        wf = wave_open(temp_wav, "rb")
        
        model = VoskModel(model_path)
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        rec.SetMaxAlternatives(3)  # Альтернативные варианты для точности
        rec.SetPartialWords(True)
        
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(json.loads(rec.Result()))
        results.append(json.loads(rec.FinalResult()))
        wf.close()
        os.remove(temp_wav)
        
        # Собираем слова с уверенностью
        raw_words = []
        for res in results:
            if isinstance(res, dict):
                for w in res.get('result', []) or []:
                    if 'word' in w:
                        raw_words.append({
                            'w': w.get('word'),
                            'start': float(w.get('start', 0.0)),
                            'end': float(w.get('end', 0.0)),
                            'conf': float(w.get('conf', 1.0))
                        })
        
        if not raw_words:
            raise RuntimeError("Пустой результат Vosk")
        
        # Адаптивная сегментация по паузам
        pause_threshold = float(os.getenv('PAUSE_THRESHOLD', '0.6'))
        grouped = []
        current = [raw_words[0]]
        
        for prev, cur in zip(raw_words, raw_words[1:]):
            gap = cur['start'] - prev['end']
            # Адаптивный порог: длинные паузы = новый сегмент
            adaptive_threshold = pause_threshold
            if gap > adaptive_threshold:
                grouped.append(current)
                current = [cur]
            else:
                current.append(cur)
        if current:
            grouped.append(current)
        
        # Формирование сегментов
        segments = []
        full_words_flat = []
        
        for idx, g in enumerate(grouped):
            start_t = g[0]['start']
            end_t = g[-1]['end']
            text_seg = ' '.join(w['w'] for w in g)
            full_words_flat.extend(w['w'] for w in g)
            avg_conf = sum(w['conf'] for w in g) / len(g)
            
            segments.append({
                'index': idx,
                'speaker': 'unknown',
                'segment_type': 'speech',
                'start': round(start_t, 3),
                'end': round(end_t, 3),
                'text': text_seg,
                'words': [{'w': w['w'], 'start': round(w['start'], 3), 'end': round(w['end'], 3), 'confidence': round(w['conf'], 3)} for w in g],
                'confidence': round(avg_conf, 3),
                'gap_from_prev': 0.0 if idx == 0 else round(start_t - segments[-1]['end'], 3)
            })
        
        # Умное определение спикеров
        for i, seg in enumerate(segments):
            tokens = set(seg['text'].lower().split())
            if tokens & lang_config['greeting_words']:
                seg['speaker'] = 'operator'
            else:
                seg['speaker'] = 'client' if i % 2 == 0 else 'operator'
        
        flat_text = ' '.join(full_words_flat).strip()
        
        return {
            'text': flat_text,
            'segments': segments,
            'meta': {
                'duration_sec': round(audio_ms / 1000.0, 3),
                'method': 'vosk_high_quality',
                'language': detected_lang,
                'model_path': model_path,
                'pause_threshold': pause_threshold
            }
        }
    except Exception as e:
        print(f"⚠️ Vosk ошибка: {e}")
        return None


# ===================== GOOGLE SR (АГРЕССИВНАЯ ПАРАЛЛЕЛИЗАЦИЯ) =====================

def transcribe_google_aggressive(filepath: str, audio_proc: AudioSegment, audio_ms: int, detected_lang: str = 'ru'):
    """Google SR с максимальной параллелизацией и двуязычной поддержкой"""
    try:
        lang_config = LANGUAGE_CONFIG.get(detected_lang, LANGUAGE_CONFIG['ru'])
        google_lang = lang_config['google_code']
        
        # Агрессивная нарезка для качества
        chunk_seconds = float(os.getenv('GOOGLE_CHUNK_SECONDS', '6'))
        chunk_overlap = float(os.getenv('GOOGLE_CHUNK_OVERLAP', '3'))
        
        print(f"🔁 Google SR AGGRESSIVE: язык={lang_config['name']}")
        
        temp_wav = "temp_transcribe.wav"
        audio_proc.export(temp_wav, format="wav")
        
        from wave import open as wave_open
        wf = wave_open(temp_wav, 'rb')
        sample_rate = wf.getframerate()
        total_frames = wf.getnframes()
        total_duration = total_frames / sample_rate
        wf.close()
        
        def recognize_chunk_multilang(idx_chunk, ms_start, ms_end, start_pos, end_pos, primary_lang, secondary_lang):
            """Распознавание с попыткой двух языков для смешанных диалогов"""
            piece = audio_proc[ms_start:ms_end]
            piece_wav = f"temp_chunk_{idx_chunk}_{os.getpid()}.wav"
            piece.export(piece_wav, format='wav')
            
            local_recognizer = sr.Recognizer()
            with sr.AudioFile(piece_wav) as source:
                # Агрессивная настройка для качества
                local_recognizer.adjust_for_ambient_noise(source, duration=0.05)
                local_recognizer.energy_threshold = 200
                local_recognizer.dynamic_energy_threshold = False
                local_recognizer.pause_threshold = 0.2
                audio_data = local_recognizer.record(source)
                
                chunk_text = ''
                lang_used = primary_lang
                
                # Пробуем основной язык
                try:
                    chunk_text = local_recognizer.recognize_google(audio_data, language=primary_lang)
                except sr.UnknownValueError:
                    # Если не распознал, пробуем второй язык
                    try:
                        chunk_text = local_recognizer.recognize_google(audio_data, language=secondary_lang)
                        lang_used = secondary_lang
                    except Exception:
                        chunk_text = '[неразборчиво]'
                except Exception:
                    chunk_text = ''
            
            try:
                os.remove(piece_wav)
            except:
                pass
            
            return {
                'idx': idx_chunk, 
                'start': start_pos, 
                'end': end_pos, 
                'text': chunk_text.strip(),
                'lang': lang_used
            }
        
        # Подготовка чанков
        step = max(chunk_seconds - chunk_overlap, 1.0)
        chunk_jobs = []
        start_pos = 0.0
        idx_chunk = 0
        
        while start_pos < total_duration:
            end_pos = min(start_pos + chunk_seconds, total_duration)
            ms_start = int(start_pos * 1000)
            ms_end = int(end_pos * 1000)
            chunk_jobs.append((idx_chunk, ms_start, ms_end, start_pos, end_pos))
            if end_pos >= total_duration:
                break
            start_pos += step
            idx_chunk += 1
        
        # МАКСИМАЛЬНАЯ параллелизация
        max_workers = min(cpu_count() * 2, len(chunk_jobs), 32)  # В 2 раза больше ядер
        print(f"   💪 АГРЕССИВНАЯ параллелизация: {max_workers} потоков")
        
        # Определяем второй язык для fallback
        secondary_lang = 'kk-KZ' if detected_lang == 'ru' else 'ru-RU'
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(recognize_chunk_multilang, *job, google_lang, secondary_lang): job[0] 
                for job in chunk_jobs
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        results.sort(key=lambda x: x['idx'])
        os.remove(temp_wav)
        
        # Умная дедупликация с учётом языка
        raw_segments = []
        for res in results:
            if res['text'] and res['text'] != '[неразборчиво]':
                raw_segments.append({
                    'start': res['start'], 
                    'end': res['end'], 
                    'text': res['text'],
                    'lang': res['lang']
                })
        
        if not raw_segments:
            raise RuntimeError('Google SR не распознал текст')
        
        deduped_segments = []
        if raw_segments:
            deduped_segments.append(raw_segments[0])
            for i in range(1, len(raw_segments)):
                prev_words = raw_segments[i-1]['text'].split()
                curr_words = raw_segments[i]['text'].split()
                
                # Умная дедупликация
                overlap_len = 0
                for j in range(1, min(len(prev_words), len(curr_words), 8) + 1):
                    if prev_words[-j:] == curr_words[:j]:
                        overlap_len = j
                
                dedupe_text = ' '.join(curr_words[overlap_len:]) if overlap_len > 0 else raw_segments[i]['text']
                if dedupe_text.strip():
                    deduped_segments.append({
                        'start': raw_segments[i]['start'],
                        'end': raw_segments[i]['end'],
                        'text': dedupe_text,
                        'lang': raw_segments[i]['lang']
                    })
        
        full_text = ' '.join(seg['text'] for seg in deduped_segments).strip()
        
        # Финальные сегменты с улучшенным таймингом
        final_segments = []
        idx = 0
        
        for rs in deduped_segments:
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
                    'speaker': 'client' if idx % 2 == 0 else 'operator',
                    'segment_type': 'speech',
                    'start': round(seg_start, 3),
                    'end': round(seg_end, 3),
                    'text': s_text,
                    'words': words,
                    'gap_from_prev': 0.0 if idx == 0 else round(seg_start - final_segments[-1]['end'], 3)
                })
                idx += 1
                cursor = seg_end
        
        return {
            'text': full_text,
            'segments': final_segments,
            'meta': {
                'duration_sec': round(audio_ms / 1000.0, 3),
                'method': 'google_speech_aggressive',
                'language': detected_lang,
                'workers': max_workers,
                'multilang': True
            }
        }
    except Exception as e:
        print(f"❌ Google SR ошибка: {e}")
        return None


# ===================== ГЛАВНАЯ ФУНКЦИЯ =====================

def transcribe_audio_file(filepath: str):
    """МАКСИМАЛЬНОЕ КАЧЕСТВО без компромиссов"""
    print(f"\n{'='*70}")
    print(f"📤 ТРАНСКРИПЦИЯ: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    audio = AudioSegment.from_file(filepath)
    original_ms = len(audio)
    audio_proc = audio.set_frame_rate(16000).set_channels(1)
    duration_min = original_ms / 60000
    
    print(f"⏱️  Длительность: {duration_min:.1f} мин")
    print(f"💻 Доступно CPU ядер: {cpu_count()}")
    
    # Приоритет: Whisper > Vosk > Google
    enable_whisper = os.getenv('ENABLE_WHISPER', '1') == '1'
    enable_vosk = os.getenv('ENABLE_VOSK', '0') == '1'
    
    result = None
    
    # 1. WHISPER - максимальная точность
    if enable_whisper and _WHISPER_AVAILABLE:
        result = transcribe_whisper(filepath, original_ms)
        if result:
            elapsed = time.time() - start_time
            print(f"\n✅ ГОТОВО за {elapsed:.1f}с (Whisper)")
            print(f"{'='*70}\n")
            return json.dumps(result, ensure_ascii=False)
    
    # 2. Определение языка для fallback методов
    sample_audio = audio_proc[:15000]  # Первые 15 секунд
    sample_wav = "temp_lang_detect.wav"
    sample_audio.export(sample_wav, format="wav")
    
    detected_lang = 'ru'
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(sample_wav) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.1)
            audio_data = recognizer.record(source)
            try:
                sample_text = recognizer.recognize_google(audio_data, language='ru-RU')
                detected_lang = detect_language(sample_text)
            except:
                # Пробуем казахский
                try:
                    sample_text = recognizer.recognize_google(audio_data, language='kk-KZ')
                    detected_lang = 'kk'
                except:
                    pass
        print(f"🔍 Определён язык: {LANGUAGE_CONFIG[detected_lang]['name']}")
    except Exception as e:
        print(f"⚠️ Автоопределение языка не удалось: {e}")
    finally:
        if os.path.exists(sample_wav):
            os.remove(sample_wav)
    
    # 3. VOSK - быстро и точно
    if enable_vosk and _VOSK_AVAILABLE and not result:
        result = transcribe_vosk(filepath, audio_proc, original_ms, detected_lang)
        if result:
            elapsed = time.time() - start_time
            print(f"\n✅ ГОТОВО за {elapsed:.1f}с (Vosk)")
            print(f"{'='*70}\n")
            return json.dumps(result, ensure_ascii=False)
    
    # 4. GOOGLE SR - агрессивная параллелизация
    if not result:
        result = transcribe_google_aggressive(filepath, audio_proc, original_ms, detected_lang)
        if result:
            elapsed = time.time() - start_time
            print(f"\n✅ ГОТОВО за {elapsed:.1f}с (Google)")
            print(f"{'='*70}\n")
            return json.dumps(result, ensure_ascii=False)
    
    raise RuntimeError("Все методы транскрипции не удались")


def transcribe_from_text(text: str):
    """Импорт текста с автоопределением языка"""
    cleaned = text.strip()
    detected_lang = detect_language(cleaned[:500])
    
    print(f"📄 Импорт текста: язык={LANGUAGE_CONFIG[detected_lang]['name']}")
    
    sentences = _split_sentences(cleaned) or [cleaned]
    total_chars = sum(len(s) for s in sentences) or 1
    total_duration = len(cleaned.split()) * 0.4
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
            'segment_type': 'speech',
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
            'method': 'text_import',
            'language': detected_lang
        }
    }
    return json.dumps(structured, ensure_ascii=False)