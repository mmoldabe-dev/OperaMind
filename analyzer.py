"""
OperaMind - Модуль анализа разговоров
Работает с актуальными моделями Gemini 2.x
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()

def analyze_transcript(transcript):
    """
    Анализирует готовый текст разговора
    
    Args:
        transcript (str): Текст разговора
        
    Returns:
        dict: Полный анализ разговора
    """
    try:
        print("🤖 Анализирую разговор через Gemini...")
        
        # Конфигурация API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise Exception("GEMINI_API_KEY не найден в .env файле")
        
        genai.configure(api_key=api_key)
        
        prompt = f"""Проанализируй этот разговор оператора с клиентом.

Текст разговора:
{transcript}

Верни ТОЛЬКО JSON в таком формате:
{{
    "topic": "краткая тема (1-3 слова)",
    "category": "одна из: техническая/биллинг/продажи/жалоба/консультация/другое",
    "sentiment": "один из: негативный/нейтральный/позитивный",
    "urgency": "один из: низкая/средняя/высокая/критическая",
    "keywords": ["ключ1", "ключ2", "ключ3", "ключ4", "ключ5"],
    "summary": "краткое содержание в 2-3 предложениях",
    "detailed_analysis": "подробный анализ разговора: качество обслуживания, проблемы клиента, как решена ситуация",
    "operator_quality": "оценка работы оператора (отлично/хорошо/удовлетворительно/плохо)",
    "recommendations": "рекомендации для улучшения обслуживания"
}}"""
        
        # АКТУАЛЬНЫЕ модели Gemini 2.x (из вашего теста)
        models_priority = [
            'models/gemini-2.0-flash',           # Основная - быстрая
            'models/gemini-2.5-flash',           # Новая версия
            'models/gemini-flash-latest',        # Последняя стабильная
            'models/gemini-2.0-flash-lite',      # Лёгкая версия
            'models/gemini-2.5-flash-lite',      # Новая лёгкая
        ]
        
        last_error = None
        
        for model_name in models_priority:
            try:
                print(f"   Пробую модель: {model_name}")
                
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                
                result_text = response.text.strip()
                
                # Очистка от markdown
                if result_text.startswith('```'):
                    parts = result_text.split('```')
                    if len(parts) >= 2:
                        result_text = parts[1]
                        if result_text.startswith('json'):
                            result_text = result_text[4:]
                
                result_text = result_text.strip()
                analysis = json.loads(result_text)
                
                print(f"✅ Анализ завершён! (модель: {model_name})")
                return analysis
                
            except Exception as e:
                error_str = str(e)
                last_error = e
                
                # Если превышена квота - пробуем следующую модель
                if "429" in error_str or "quota" in error_str.lower():
                    print(f"   ⚠️  Квота исчерпана для {model_name}, пробую другую...")
                    time.sleep(2)  # Задержка перед следующей попыткой
                    continue
                else:
                    print(f"   ❌ {model_name}: {error_str[:100]}")
                    continue
        
        # Если все модели недоступны
        raise Exception(f"Все модели недоступны. Последняя ошибка: {last_error}")
        
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка парсинга JSON: {e}")
        return create_fallback_analysis()
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return create_fallback_analysis()


def create_fallback_analysis():
    """Базовый анализ при ошибке"""
    return {
        "topic": "Не определено",
        "category": "другое",
        "sentiment": "нейтральный",
        "urgency": "средняя",
        "keywords": ["разговор", "клиент", "оператор"],
        "summary": "Анализ не выполнен. Требуется ручная проверка.",
        "detailed_analysis": "Автоматический анализ недоступен.",
        "operator_quality": "не оценено",
        "recommendations": "Проверьте разговор вручную."
    }