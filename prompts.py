import json

from validation import (
    ValidationReason,
    ResponseValidationError
)

SYSTEM_PROMPT = """
Ты генерируешь карточку товара для маркетплейса на основе входных данных.
На вход ты получаешь информацию о товаре в JSON формате.

Твоя задача:
Верни только JSON с полями: description, pros, cons, tags.
Поле description должно содержать краткое описание товара в 2-5 предложениях (строка от 120 до 700 символов).
Поле pros должно описывать "плюсы" товара (список из 2-5 непустых строк).
Поле cons должно описывать "минусы" товара (список из 1-3 непустых строк).
Поле tags должно содержать короткие теги для поиска товара (список из 3-8 непустых строк).

Требования:
1. Не добавляй новые поля.
2. Не придумывай факты, которых нет во входных данных.
3. Не раздувай текст: пиши емко и соблюдай указанные диапазоны. Лучше написать меньше, но по делу.
4. Поле description формируй только на основе title, category и attributes. Другие источники использовать запрещено.
5. Если отзыв в reviews противоречит attributes, не используй его как источник для других полей.
6. Каждый пункт pros и cons должен быть уникальным и коротким, в среднем 2-5 слов.
7. Не включай в pros или cons нейтральные факты.
8. В tags сочетай тип товара, назначение и ключевые характеристики; каждый тег должен быть полезен для поиска.

Пример обработки:
Вход:
{
  "product_id": "P-1001",
  "title": "Беспроводные наушники AirBass X2",
  "category": "Электроника / Наушники",
  "attributes": {
    "бренд": "AirBass",
    "тип": "внутриканальные",
    "время работы": "24 ч",
    "Bluetooth": "5.3",
    "влагозащита": "IPX4"
  },
  "reviews": [
    "Звук чистый, бас глубокий, за свои деньги отлично",
    "На морозе быстро садятся, зимой неудобно"
  ]
}

Выход:
{
  "description": "Беспроводные внутриканальные наушники AirBass X2 работают по Bluetooth 5.3 и защищены от брызг по стандарту IPX4. Заявленное время работы составляет до 24 часов.",
  "pros": ["Чистый звук", "Глубокий бас"],
  "cons": ["Быстро разряжаются на морозе"],
  "tags": ["наушники", "беспроводные", "внутриканальные", "Bluetooth 5.3", "IPX4"]
}

А теперь обработай этот вход:
"""

def build_messages(source_product: dict) -> list[dict]:
    """Формирует инструкцию для первой генерации карточки."""
    return [
        {
            "role" : "system",
            "content" : SYSTEM_PROMPT,
        },
        {
            "role" : "user",
            "content" : json.dumps(source_product, ensure_ascii=False)
        }
    ]


def build_retry_message(error: ResponseValidationError) -> str:
    """Формирует инструкцию для повторной генерации карточки."""
    fixes = []
    details = error.details or {}

    if ValidationReason.INVALID_JSON in error.reasons:
        fixes.append(
            "Верни один корректный JSON-объект без текста до или после него."
        )

    if ValidationReason.MISSING_FIELD in error.reasons:
        fixes.append(
            "JSON должен содержать все поля: description, pros, cons, tags."
        )

    if ValidationReason.WRONG_TYPE in error.reasons:
        fixes.append(
            "description должен быть строкой, "
            "pros, cons и tags - списками строк."
        )

    if ValidationReason.DESCRIPTION_LENGTH in error.reasons:
        length = details.get("description_length")

        if length is not None:
            if length < 120:
                fixes.append(
                    f"description содержит только {length} символов. "
                    "Необходимо не менее 120 символов. Добавь побольше деталей."
                )
            elif length > 700:
                fixes.append(
                    f"description содержит {length} символов - это слишком много. "
                    "Нельзя использовать более 700 символов. Сократи текст."
                )

    if ValidationReason.DESCRIPTION_SENTENCE_COUNT in error.reasons:
        sentence_count = details.get("description_sentence_count")

        if sentence_count is not None:
            if sentence_count < 2:
                fixes.append(
                    f"description содержит {sentence_count} предложений - "
                    "это слишком мало. Напиши еще одно предложение."
                )
            elif sentence_count > 5:
                fixes.append(
                    f"description содержит {sentence_count} предложений - "
                    "а нужно не более 5 предложений. Убери лишние предложения."
                )

    if ValidationReason.WRONG_LIST_LENGTH in error.reasons:
        targets = {
            "pros": (2, 5),
            "cons": (1, 3),
            "tags": (3, 8),
        }

        for field, (min_length, target_max) in targets.items():
            length = details.get(f"{field}_length")

            if length is None:
                continue

            if length < min_length:
                fixes.append(
                    f"{field} содержит {length} элементов — это слишком мало. "
                    f"Добавь только недостающие элементы: "
                    f"в итоге должно быть {min_length}-{target_max}."
                )

            elif length > target_max:
                fixes.append(
                    f"{field} содержит {length} элементов — это слишком много. "
                    f"Удаляй лишние элементы и оставь только "
                    f"{min_length}-{target_max} самых подходящих."
                )

    if ValidationReason.EMPTY_VALUE in error.reasons:
        fixes.append(
            "Не используй пустые строки или пустые элементы списков."
        )

    if ValidationReason.OUTPUT_TRUNCATED in error.reasons:
        fixes.append(
            "Предыдущий ответ был обрезан. "
            "Сделай ответ существенно короче."
        )

    if ValidationReason.OTHER in error.reasons:
        fixes.append(
            "Исправь ответ в соответствии с требованиями."
        )

    return (
        "Предыдущий ответ не прошёл валидацию. "
        "Сгенерируй весь JSON заново.\n"
        "КРИТИЧЕСКИ ВАЖНО: не увеличивай объём ответа без необходимости. "
        "Если чего-то слишком много, удаляй лишнее, а не добавляй новый текст.\n"
        "Исправь нарушения:\n- "
        + "\n- ".join(fixes)
    )