import logging
import aiohttp
import asyncio
import os
import re
import aiosqlite
from datetime import datetime
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InputFile
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
import edge_tts

BOT_TOKEN = os.getenv("BOT_TOKEN")
LLM_API_KEY = os.getenv("LLM_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DB_PATH = "teacher.db"

LEVEL_THRESHOLDS = {
    "A1": 25,
    "A2": 50,
    "B1": 100
}

# Расширенный список моделей
MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

awaiting_question = set()

# ============ БАЗА ДАННЫХ ============

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                level TEXT DEFAULT 'A1',
                started_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                topic TEXT,
                created_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                question TEXT,
                answer TEXT,
                created_at TEXT
            )
        """)
        await db.commit()

async def get_user(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT * FROM users WHERE user_id=?", (user_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                await db.execute(
                    "INSERT INTO users (user_id, level, started_at) VALUES (?, ?, ?)",
                    (user_id, "A1", datetime.now().isoformat())
                )
                await db.commit()
                return {"user_id": user_id, "level": "A1"}
            return {"user_id": row[0], "level": row[1]}

async def save_lesson(user_id: int, topic: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO lessons (user_id, topic, created_at) VALUES (?, ?, ?)",
            (user_id, topic, datetime.now().isoformat())
        )
        await db.commit()

async def save_question(user_id: int, question: str, answer: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO questions (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
            (user_id, question, answer, datetime.now().isoformat())
        )
        await db.commit()

async def get_user_stats(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM lessons WHERE user_id=?", (user_id,)
        ) as cur:
            lesson_count = (await cur.fetchone())[0] or 0
        
        async with db.execute(
            "SELECT topic FROM lessons WHERE user_id=? ORDER BY id DESC LIMIT 10",
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()
            recent_topics = [r[0] for r in rows]
        
        user = await get_user(user_id)
        return {
            "level": user["level"],
            "lesson_count": lesson_count,
            "recent_topics": recent_topics
        }

async def check_level_up(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        user = await get_user(user_id)
        current_level = user["level"]
        
        async with db.execute(
            "SELECT COUNT(*) FROM lessons WHERE user_id=?", (user_id,)
        ) as cur:
            count = (await cur.fetchone())[0]
        
        next_level = None
        if current_level == "A1" and count >= LEVEL_THRESHOLDS["A1"]:
            next_level = "A2"
        elif current_level == "A2" and count >= LEVEL_THRESHOLDS["A2"]:
            next_level = "B1"
        
        if next_level:
            await db.execute(
                "UPDATE users SET level=? WHERE user_id=?",
                (next_level, user_id)
            )
            await db.commit()
            return next_level
        return None

# ============ ИИ ============

async def ask_llm(prompt: str, temperature: float = 0.7) -> str:
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    errors = []
    for model in MODELS:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        try:
            logging.info(f"Пробую модель: {model}")
            async with aiohttp.ClientSession() as session:
                async with session.post(API_URL, headers=headers, json=data,
                                       timeout=aiohttp.ClientTimeout(total=45)) as resp:
                    result = await resp.json()
                    
                    if "choices" in result:
                        logging.info(f"✅ Успех через {model}")
                        return result["choices"][0]["message"]["content"]
                    
                    error_code = result.get("error", {}).get("code", 0)
                    error_msg = result.get("error", {}).get("message", "Unknown")
                    logging.warning(f"❌ {model}: код {error_code} - {error_msg}")
                    
                    if error_code in [429, 502, 503]:
                        errors.append(f"{model}: {error_msg}")
                        continue
                    
                    errors.append(f"{model}: {error_msg}")
                    continue
                    
        except asyncio.TimeoutError:
            logging.warning(f"⏱️ {model}: timeout")
            errors.append(f"{model}: timeout")
            continue
        except Exception as e:
            logging.warning(f"⚠️ {model}: {e}")
            errors.append(f"{model}: {str(e)}")
            continue
    
    logging.error(f"Все модели недоступны. Ошибки: {errors}")
    return "⚠️ Все модели недоступны. Попробуйте через минуту."

async def generate_audio(text: str, filename: str) -> str:
    path = f"audio/{filename}.mp3"
    if not os.path.exists("audio"):
        os.makedirs("audio")
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural", rate="-30%")
    await communicate.save(path)
    return path

def format_grammar(text: str) -> str:
    patterns = [
        r'\b(Hello|Hi|Goodbye|Bye|Hey)\b',
        r'\b(Good morning|Good afternoon|Good evening|Good night)\b',
        r'\b(I|you|he|she|it|we|they)\b',
        r'\b(am|is|are|was|were|be|been)\b',
        r'\b(have|has|had|do|does|did)\b',
        r'\b(a|an|the)\b',
        r'\b(this|that|these|those)\b',
        r'\b(my|your|his|her|its|our|their)\b',
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, lambda m: f'**{m.group(0)}**', result, flags=re.IGNORECASE)
    return result

def get_main_menu():
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("📚 Следующий урок"), KeyboardButton("❓ Задать вопрос"))
    kb.add(KeyboardButton("📊 Мой прогресс"))
    return kb

# ============ ЛОГИКА УРОКОВ ============

async def determine_next_topic(user_id: int) -> str:
    stats = await get_user_stats(user_id)
    recent = stats["recent_topics"]
    level = stats["level"]
    count = stats["lesson_count"]
    
    if level == "B1":
        prompt = f"""Ты методист английского. Ученик уровня B1 (средний) прошёл {count} уроков.
Последние темы: {', '.join(recent) if recent else 'нет'}.

Определи следующую тему для B1:
- Разговорные ситуации (путешествия, работа, хобби)
- Условные предложения (if...)
- Сложные времена (Present Perfect, Future)

Ответь ТОЛЬКО названием темы (одним словом или короткой фразой на русском)."""
    elif level == "A2":
        prompt = f"""Ты методист английского. Ученик уровня A2 (базовый) прошёл {count} уроков.
Последние темы: {', '.join(recent) if recent else 'нет'}.

Определи следующую тему для A2:
- Короткие диалоги (магазин, кафе, транспорт)
- Past Simple, Future Simple
- Вопросы и ответы
- Ситуации из жизни

Ответь ТОЛЬКО названием темы (одним словом или короткой фразой на русском)."""
    else:
        prompt = f"""Ты методист английского для детей и начинающих. Ученик уровня A1 (новичок) прошёл {count} уроков.
Последние темы: {', '.join(recent) if recent else 'нет'}.

Ученику предстоит пройти {LEVEL_THRESHOLDS['A1']} базовых уроков перед переходом на A2.

Определи следующую тему от простого к сложному (25 шагов):
- Приветствие, знакомство, семья
- Числа, цвета, дни недели
- Еда, напитки, фрукты
- Дом, комната, мебель
- Одежда, части тела
- Школа, предметы
- Животные, природа
- Погода, времена года
- Хобби, спорт
- Глаголы движения
- Present Simple (я/ты/мы/они)
- Present Simple (он/она/оно)
- Отрицания и вопросы
- Предлоги места (in, on, under)
- Время (часы)

НЕ ПОВТОРЯЙ темы из списка последних уроков.
Ответь ТОЛЬКО названием темы (одним словом или короткой фразой на русском)."""
    
    topic = await ask_llm(prompt, temperature=0.3)
    return topic.replace('"', '').replace("'", '').strip()

async def generate_lesson(topic: str, user_id: int) -> list:
    stats = await get_user_stats(user_id)
    level = stats["level"]
    recent = stats["recent_topics"]
    count = stats["lesson_count"]
    
    if level in ["A2", "B1"]:
        prompt = f"""Ты учитель английского. Ученик уровня {level}, прошёл {count} уроков.
Недавно было: {', '.join(recent[:5]) if recent else 'ничего'}.
Создай урок на тему "{topic}".

Для {level}:
- Дай 5 коротких ДИАЛОГОВ (2 реплики в каждом) или ситуаций
- Используй Past Simple, Future, вопросы
- Переводы точные, транскрипция русскими буквами в скобках

ФОРМАТ (строго):
📚 Тема: {topic}

1.
🔹 [русская фраза]
🔸 [english] ([транскрипция])
🔹 [русская реплика собеседника]
🔸 [english] ([транскрипция])

(5 пар)

💡 Грамматика: объяснение на русском, 2-3 предложения."""
    else:
        prompt = f"""Ты учитель английского для детей и начинающих. Ученик уровня A1, прошёл {count} из {LEVEL_THRESHOLDS['A1']} уроков базового курса.
Недавно было: {', '.join(recent[:5]) if recent else 'ничего'}.
Создай урок на тему "{topic}".

ВАЖНО ДЛЯ ДЕТЕЙ:
- Используй простые, понятные слова
- Фразы должны быть короткими (3-5 слов)
- Избегай сложных грамматических конструкций
- Переводы должны быть естественными для ребёнка

ПРАВИЛА:
- Реальные слова, точные переводы
- Транскрипция РУССКИМИ буквами в скобках
- Постепенно усложняй от урока к уроку

ФОРМАТ (строго):
📚 Тема: {topic}

1.
🔹 [русская фраза]
🔸 [english] ([транскрипция])

2.
🔹 [русская фраза]
🔸 [english] ([транскрипция])

(5 фраз)

💡 Грамматика: объяснение простыми словами на русском, 2-3 предложения."""
    
    lesson_text = await ask_llm(prompt)
    blocks = []
    
    if lesson_text.startswith("Ошибка") or lesson_text.startswith("⚠️"):
        return [{"type": "text", "content": lesson_text}]
    
    lines = lesson_text.split('\n')
    current_text = ""
    current_audio = None
    grammar_text = ""
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        if stripped.startswith('📚'):
            if current_text:
                blocks.append({"type": "text", "content": current_text})
                current_text = ""
            # Пропускаем строку темы - она уже показана в "Готовлю урок"
        
        elif stripped.startswith('💡'):
            if current_text:
                blocks.append({"type": "text", "content": current_text})
                current_text = ""
            grammar_text = stripped
        
        elif grammar_text and not stripped[0].isdigit():
            grammar_text += "\n" + stripped
        
        elif len(stripped) > 0 and stripped[0].isdigit() and stripped.endswith('.'):
            if grammar_text:
                blocks.append({"type": "text", "content": format_grammar(grammar_text)})
                grammar_text = ""
            if current_text:
                blocks.append({"type": "text", "content": current_text})
            current_text = stripped
        
        elif stripped.startswith('🔹'):
            current_text += "\n" + stripped
        
        elif stripped.startswith('🔸'):
            current_text += "\n" + stripped
            en_part = stripped.replace('🔸', '').strip()
            if '(' in en_part:
                current_audio = en_part.split('(')[0].strip()
            else:
                current_audio = en_part
            blocks.append({"type": "phrase", "text": current_text, "audio": current_audio})
            current_text = ""
            current_audio = None
        
        else:
            if current_text:
                current_text += "\n" + stripped
            else:
                blocks.append({"type": "text", "content": stripped})
    
    if grammar_text:
        blocks.append({"type": "text", "content": format_grammar(grammar_text)})
    if current_text:
        blocks.append({"type": "text", "content": current_text})
    
    return blocks

async def send_lesson(message_or_query, user_id: int):
    topic = await determine_next_topic(user_id)
    
    msg = message_or_query.message if hasattr(message_or_query, 'message') else message_or_query
    
    await msg.answer(f"🧠 Готовлю урок на тему: {topic}...")
    
    blocks = await generate_lesson(topic, user_id)
    
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            await msg.answer(block["content"], parse_mode="Markdown")
        elif block["type"] == "phrase":
            try:
                audio_path = await generate_audio(block["audio"], f"lesson_{user_id}_{i}")
                audio = InputFile(audio_path)
                await msg.answer_voice(
                    audio,
                    caption=block["text"]
                )
            except Exception as e:
                await msg.answer(f"{block['text']}\n⚠️ Ошибка озвучки: {e}")
        await asyncio.sleep(0.5)
    
    await save_lesson(user_id, topic)
    leveled_up = await check_level_up(user_id)
    
    if leveled_up:
        await msg.answer(
            f"🎉 **Поздравляю!** Вы перешли на уровень **{leveled_up}**!\n\n"
            "Уроки становятся интереснее: диалоги, ситуации, живая речь!",
            parse_mode="Markdown"
        )
    
    await msg.answer("✅ Урок завершён!")

# ============ ОБРАБОТЧИКИ ============

@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    await init_db()
    await get_user(message.from_user.id)
    await message.answer(
        "👋 Привет! Я твой автономный учитель английского.\n\n"
        "📚 **Программа обучения:**\n"
        f"• **A1 (новичок)** — {LEVEL_THRESHOLDS['A1']} уроков базы\n"
        f"• **A2 (базовый)** — диалоги и ситуации\n"
        f"• **B1 (средний)** — свободная речь\n\n"
        "Прогресс сохраняется автоматически.\n"
        "Подходит для детей и начинающих! 🎓\n\n"
        "Начинаем первый урок!",
        reply_markup=get_main_menu()
    )
    await send_lesson(message, message.from_user.id)

@dp.message_handler(lambda m: m.text == "📚 Следующий урок")
async def handle_next_lesson(message: types.Message):
    await send_lesson(message, message.from_user.id)

@dp.message_handler(lambda m: m.text == "📊 Мой прогресс")
async def handle_stats(message: types.Message):
    stats = await get_user_stats(message.from_user.id)
    level = stats["level"]
    count = stats["lesson_count"]
    
    text = f"📊 **Твой прогресс**\n\n"
    text += f"🎓 Уровень: **{level}**\n"
    text += f"📚 Пройдено уроков: **{count}**\n\n"
    
    if level == "A1":
        remaining = LEVEL_THRESHOLDS["A1"] - count
        progress = int((count / LEVEL_THRESHOLDS["A1"]) * 20)
        text += f"📈 До A2 осталось: **{remaining}** уроков\n"
        text += f"Прогресс: {'█' * progress}{'░' * (20 - progress)}\n\n"
    elif level == "A2":
        remaining = LEVEL_THRESHOLDS["A2"] - count
        text += f"📈 До B1 осталось: **{remaining}** уроков\n\n"
    
    text += f"🔥 Последние темы:\n"
    if stats["recent_topics"]:
        for i, t in enumerate(stats["recent_topics"][:5], 1):
            text += f"{i}. {t}\n"
    else:
        text += "Пока нет\n"
    
    await message.answer(text, parse_mode="Markdown", reply_markup=get_main_menu())

@dp.message_handler(lambda m: m.text == "❓ Задать вопрос")
async def handle_question_start(message: types.Message):
    awaiting_question.add(message.from_user.id)
    await message.answer(
        "💬 Напиши свой вопрос, и я объясню простыми словами.\n\n"
        "Например: «Как сказать 'я устал'?» или «Объясни a и the»"
    )

@dp.message_handler()
async def handle_question(message: types.Message):
    if message.from_user.id in awaiting_question:
        awaiting_question.discard(message.from_user.id)
        await message.answer("🤔 Думаю...")
        
        prompt = f"""Ты дружелюбный учитель английского для детей и начинающих.
Ученик задал вопрос: {message.text}

Ответь понятно и кратко НА РУССКОМ, простыми словами.
Если вопрос про английские слова — дай примеры с переводом и транскрипцией русскими буквами."""
        
        answer = await ask_llm(prompt)
        await message.answer(answer, reply_markup=get_main_menu())
        await save_question(message.from_user.id, message.text, answer)
    else:
        await message.answer("Используй кнопки меню 👇", reply_markup=get_main_menu())

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_db())
    executor.start_polling(dp, skip_updates=True)
