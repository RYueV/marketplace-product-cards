from pydantic import BaseModel
from openai import AsyncOpenAI, BadRequestError

from validation import (
    ProductCard,
    ValidationReason,
    ResponseValidationError,
    parse_and_validate
)
from prompts import (
    build_messages,
    build_retry_message
)
from io_utils import save_failure_diagnostic

class GenerationResult(BaseModel):
    """Результат генерации одной карточки."""
    product_id: str
    card: ProductCard | None
    valid: bool
    attempts: int
    errors: list[ValidationReason]
    input_tokens: int = 0
    output_tokens: int = 0


async def generate_one(
        client: AsyncOpenAI,
        model_name: str,
        source_product: dict,
        generation_params: dict,
        max_attempts: int = 2,
        diagnostic_dir: str | None = None,
) -> GenerationResult:
    """Генерирует и валидирует одну карточку с повторными попытками."""
    api_params = generation_params.copy()

    base_seed = api_params.pop("seed", 42)

    extra_body = {}
    for name in ("top_k", "repetition_penalty"):
        if name in api_params:
            extra_body[name] = api_params.pop(name)


    messages = build_messages(source_product)

    all_errors = []
    input_tokens = 0
    output_tokens = 0
    attempts_log = []

    for attempt in range(1, max_attempts + 1):
        attempt_seed = base_seed + attempt - 1

        attempt_log = {
            "attempt": attempt,
            "seed": attempt_seed,
            "messages": [message.copy() for message in messages],
        }

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                extra_body=extra_body,
                seed=attempt_seed,
                **api_params,
            )

        except BadRequestError as error:
            error_text = str(error)

            if "maximum context length" in error_text:
                attempt_log["api_error"] = error_text
                attempt_log["validation_reasons"] = [
                    ValidationReason.LONG_INPUT.value
                ]
                attempt_log["validation_details"] = {}

                attempts_log.append(attempt_log)

                if diagnostic_dir is not None:
                    save_failure_diagnostic(
                        diagnostic_dir=diagnostic_dir,
                        source_product=source_product,
                        generation_params=generation_params,
                        attempts_log=attempts_log,
                    )

                return GenerationResult(
                    product_id=source_product["product_id"],
                    card=None,
                    valid=False,
                    attempts=attempt,
                    errors=all_errors + [ValidationReason.LONG_INPUT],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            raise

        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        attempt_log["raw_response"] = content
        attempt_log["finish_reason"] = finish_reason

        if response.usage is not None:
            input_tokens += response.usage.prompt_tokens or 0
            output_tokens += response.usage.completion_tokens or 0

        try:
            if finish_reason == "length":
                raise ResponseValidationError(
                    reasons=[ValidationReason.OUTPUT_TRUNCATED],
                    message="Ответ модели был обрезан по max_tokens.",
                )

            card = parse_and_validate(
                content=content,
                source_product=source_product
            )

            return GenerationResult(
                product_id=source_product["product_id"],
                card=card,
                valid=True,
                attempts=attempt,
                errors=all_errors,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except ResponseValidationError as error:
            attempt_log["validation_reasons"] = [
                reason.value for reason in error.reasons
            ]

            attempt_log["validation_details"] = error.details
            attempts_log.append(attempt_log)

            all_errors.extend(error.reasons)

            if attempt == max_attempts:
                break

            if ValidationReason.EMPTY_RESPONSE in error.reasons:
                messages = build_messages(source_product)
                continue

            messages = build_messages(source_product)

            if ValidationReason.OUTPUT_TRUNCATED in error.reasons:
                messages.append(
                    {
                        "role": "user",
                        "content": build_retry_message(error),
                    }
                )
            else:
                messages.extend([
                    {
                        "role": "assistant",
                        "content": content,
                    },
                    {
                        "role": "user",
                        "content": build_retry_message(error),
                    },
                ])

    if diagnostic_dir is not None:
        save_failure_diagnostic(
            diagnostic_dir=diagnostic_dir,
            source_product=source_product,
            generation_params=generation_params,
            attempts_log=attempts_log,
        )

    return GenerationResult(
        product_id=source_product["product_id"],
        card=None,
        valid=False,
        attempts=max_attempts,
        errors=all_errors,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )