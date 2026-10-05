import re
import json
from enum import Enum
from pydantic import(
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    ValidationError,
)

#   ОПИСАНИЕ КОНТРАКТА ОТВЕТА С ПОМОЩЬЮ PYDANTIC

def count_sentences(text):
    """Число предложений в тексте. Точка внутри числа концом предложения не считается."""
    return len([s for s in re.split(r"[.!?]+(?!\d)", text) if s.strip()])


class ProductCard(BaseModel):
    """Результат обработки карточки."""

    model_config = ConfigDict(extra="forbid")

    product_id: str
    description: str = Field(min_length=120, max_length=700)
    pros: list[str] = Field(min_length=2, max_length=5)
    cons: list[str] = Field(min_length=1, max_length=3)
    tags: list[str] = Field(min_length=3, max_length=8)

    @field_validator("product_id", "description", mode="before")
    @classmethod
    def strip_string(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("pros", "cons", "tags", mode="before")
    @classmethod
    def strip_list_items(cls, value):
        if isinstance(value, list):
            return [
                item.strip() if isinstance(item, str) else item
                for item in value
            ]
        return value

    @field_validator("product_id", "description")
    @classmethod
    def string_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError(
                "Строка не должна быть пустой | поля: product_id, description."
            )
        return value

    @field_validator("pros", "cons", "tags")
    @classmethod
    def list_items_must_not_be_empty(
        cls,
        value: list[str],
    ) -> list[str]:
        if any(not item for item in value):
            raise ValueError(
                "Элемент списка не должен быть пустым | поля: pros, cons, tags."
            )
        return value

    @field_validator("description")
    @classmethod
    def description_sentence_count(cls, value: str) -> str:
        sentence_count = count_sentences(value)

        if not 2 <= sentence_count <= 5:
            raise ValueError(
                "description должно содержать от 2 до 5 предложений"
            )

        return value


#   ПАРСИНГ, ВАЛИДАЦИЯ И ПОЛИТИКА ОБРАБОТКИ ОШИБОК

class ValidationReason(str, Enum):
    """
    Категории ошибок в ответе модели.
    PRODUCT_ID_MISMATCH и EXTRA_FIELD обрабатываются детерминированно.
    """
    EMPTY_RESPONSE = "empty_response"
    INVALID_JSON = "invalid_json"
    MISSING_FIELD = "missing_field"
    WRONG_TYPE = "wrong_type"
    DESCRIPTION_LENGTH = "description_length"
    DESCRIPTION_SENTENCE_COUNT = "description_sentence_count"
    WRONG_LIST_LENGTH = "wrong_list_length"
    EMPTY_VALUE = "empty_value"
    LONG_INPUT = "long_input"
    OUTPUT_TRUNCATED = "output_truncated"
    OTHER = "other_validation_error"


class ResponseValidationError(ValueError):
    """Ошибка парсинга или проверки ответа модели."""
    def __init__(
            self,
            reasons: list[ValidationReason],
            message: str,
            details: dict | None = None,
    ):
        super().__init__(message)
        self.reasons = reasons
        self.details = details or {}


def get_validation_info(
    error: ValidationError,
) -> tuple[list[ValidationReason], dict]:
    """Преобразует ошибки Pydantic в категории ошибок карточки."""

    reasons = []
    details = {}

    for err in error.errors():
        error_type = err["type"]
        field = err["loc"][0] if err["loc"] else None
        message = err["msg"].lower()
        input_value = err.get("input")

        if error_type == "missing":
            reason = ValidationReason.MISSING_FIELD

        elif error_type in {"string_type", "list_type"}:
            reason = ValidationReason.WRONG_TYPE

        elif (
            field == "description"
            and error_type in {"string_too_short", "string_too_long"}
        ):
            reason = ValidationReason.DESCRIPTION_LENGTH

            if isinstance(input_value, str):
                details["description_length"] = len(input_value.strip())

        elif (
            field in {"pros", "cons", "tags"}
            and error_type in {"too_short", "too_long"}
        ):
            reason = ValidationReason.WRONG_LIST_LENGTH

            if isinstance(input_value, list):
                details[f"{field}_length"] = len(input_value)

        elif "пуст" in message:
            reason = ValidationReason.EMPTY_VALUE

        elif (
            field == "description"
            and "предложен" in message
        ):
            reason = ValidationReason.DESCRIPTION_SENTENCE_COUNT

            if isinstance(input_value, str):
                details["description_sentence_count"] = count_sentences(
                    input_value
                )

        else:
            reason = ValidationReason.OTHER

        reasons.append(reason)

    reasons = list(dict.fromkeys(reasons))

    return reasons, details


def repair_response(data: dict, source_product: dict) -> dict:
    """Вносит детерминированные исправления в сгенерированную карточку."""

    allowed_fields = {
        "product_id", "description", "pros", "cons", "tags"
    }

    repaired = {
        key : value
        for key, value in data.items()
        if key in allowed_fields
    }

    repaired["product_id"] = source_product["product_id"]

    return repaired


def parse_and_validate(content: str, source_product: dict) -> ProductCard:
    """Распарсить JSON-ответ, проверить контракт карточки и исправить ответ."""
    if not content or not content.strip():
        raise ResponseValidationError(
            reasons=[ValidationReason.EMPTY_RESPONSE],
            message="Модель вернула пустой ответ.",
        )

    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ResponseValidationError(
            reasons=[ValidationReason.INVALID_JSON],
            message="Ответ модели не является корректным JSON.",
        ) from error

    if not isinstance(data, dict):
        raise ResponseValidationError(
            reasons=[ValidationReason.WRONG_TYPE],
            message="Ответ модели должен быть JSON-объектом.",
        )

    repaired = repair_response(data, source_product)

    try:
        return ProductCard.model_validate(repaired)

    except ValidationError as error:
        reasons, details = get_validation_info(error)

        raise ResponseValidationError(
            reasons=reasons,
            message=str(error),
            details=details,
        ) from error