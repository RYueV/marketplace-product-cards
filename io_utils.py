import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    """Загружает данные из JSON-файла."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    """Сохраняет данные в JSON-файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_jsonl(path: str | Path) -> list[dict]:
    """Загружает строки JSONL-файла в список словарей."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_jsonl(
    path: str | Path,
    items: list[dict],
) -> None:
    """Сохраняет список словарей в JSONL-файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


def append_jsonl(
    path: str | Path,
    item: dict,
) -> None:
    """Добавляет одну запись в JSONL-файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


def load_checkpoint(path: str | Path) -> set[str]:
    """Возвращает product_id уже обработанных товаров из checkpoint."""
    path = Path(path)

    if not path.exists():
        return set()

    return {
        item["product_id"]
        for item in load_jsonl(path)
    }


def save_checkpoint(
    path: str | Path,
    result: dict,
) -> None:
    """Добавляет результат обработки товара в checkpoint."""
    append_jsonl(path, result)


def load_results(path: str | Path) -> list[dict]:
    """Загружает сохранённые результаты генерации."""
    path = Path(path)

    if not path.exists():
        return []

    return load_jsonl(path)


def save_failure_diagnostic(
    diagnostic_dir: str | Path,
    source_product: dict,
    generation_params: dict,
    attempts_log: list[dict],
) -> None:
    """Сохраняет подробную диагностику карточки, не прошедшей все попытки."""
    diagnostic_dir = Path(diagnostic_dir)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{source_product['product_id']}.json"

    data = {
        "product_id": source_product["product_id"],
        "generation_params": generation_params,
        "attempts": attempts_log,
    }

    save_json(
        diagnostic_dir / filename,
        data,
    )