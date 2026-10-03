# Эксперимент temperature

Цель: выбрать устойчивое значение `temperature` для генерации валидных JSON-карточек.

Фиксировано:
- top_p = 0.9
- seed = 42
- max_attempts = 3

Проверяем:
- temperature = 0.1
- temperature = 0.2
- temperature = 0.4

Основная метрика: valid_rate.
Дополнительная: avg_attempts.
