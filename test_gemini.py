"""
Тестовый скрипт для проверки Gemini API
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def test_gemini_api():
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY не найден в .env файле!")
        return
    
    print(f"✅ API ключ найден: {api_key[:10]}...")
    
    try:
        genai.configure(api_key=api_key)
        print("\n📋 Доступные модели для generateContent:\n")
        
        available = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                available.append(model.name)
                print(f"   ✓ {model.name}")
        
        if not available:
            print("   ❌ Нет доступных моделей!")
            return
        
        # Тестируем первую доступную модель
        print(f"\n🧪 Тестирую модель: {available[0]}\n")
        
        model = genai.GenerativeModel(available[0])
        response = model.generate_content("Скажи 'привет' на русском")
        
        print(f"✅ Модель работает!")
        print(f"📝 Ответ: {response.text}\n")
        
        print("="*60)
        print("✅ ВСЁ РАБОТАЕТ! Используйте эту модель в analyzer.py:")
        print(f"   model = genai.GenerativeModel('{available[0]}')")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n🔍 Возможные причины:")
        print("   1. Неверный API ключ")
        print("   2. API ключ не активирован на https://makersuite.google.com/")
        print("   3. Превышена квота запросов")
        print("   4. Нужно обновить библиотеку: pip install --upgrade google-generativeai")

if __name__ == '__main__':
    test_gemini_api()