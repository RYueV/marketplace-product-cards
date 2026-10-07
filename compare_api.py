
import os
import asyncio
import random
import time
from collections import Counter

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from transformers import AutoTokenizer

from generation import GenerationResult, generate_one
from io_utils import load_jsonl, save_json
from long_input import prepare_product_input
from prompts import build_messages, build_retry_message
from validation import (
    ValidationReason,
    ResponseValidationError,
    parse_and_validate,
)

LOCAL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
API_MODEL_NAME = "apodex/apodex-1.1-mini:free"

GENERATION_PARAMS = {
    "temperature": 0.5,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_tokens": 250,
    "seed": 42,
}

MAX_ATTEMPTS = 2
MAX_API_ATTEMPTS = 4
API_SAMPLE_SIZE = 20
MAX_INPUT_TOKENS = 1500

API_INPUT_PRICE = 20
API_OUTPUT_PRICE = 60


def calculate_retry_delay(
        attempt: int,
        base_delay: float = 1.0,
        max_delay: float = 30.0
) -> float:
    """Рассчитывает задержку перед повторным API-запросом."""
    exponential_delay = base_delay * 2 ** (attempt - 1)
    jitter = random.uniform(0, base_delay)
    return min(exponential_delay + jitter, max_delay)


async def create_chat_completion_with_retry(
        client: AsyncOpenAI,
        model_name: str,
        messages: list[dict],
        generation_params: dict,
        max_api_attempts: int = MAX_API_ATTEMPTS
):
    """Отправляет API-запрос с повторами при временных ошибках."""
    api_params = generation_params.copy()
    seed = api_params.pop("seed", 42)
    repetition_penalty = api_params.pop("repetition_penalty", 1.1)

    for attempt in range(1, max_api_attempts + 1):
        try:
            return await client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                seed=seed,
                extra_body={
                    "repetition_penalty": repetition_penalty,
                    "reasoning": {"enabled": False},
                },
                **api_params,
            )

        except (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
        ):
            if attempt == max_api_attempts:
                raise

            delay = calculate_retry_delay(attempt)
            await asyncio.sleep(delay)

        except BadRequestError:
            raise


async def generate_one_api(
        client: AsyncOpenAI,
        model_name: str,
        source_product: dict,
        generation_params: dict,
        max_attempts: int = MAX_ATTEMPTS
) -> GenerationResult:
    """Генерирует и валидирует одну карточку через OpenRouter."""
    messages = build_messages(source_product)
    all_errors = []
    input_tokens = 0
    output_tokens = 0

    base_seed = generation_params.get("seed", 42)

    for attempt in range(1, max_attempts + 1):
        current_params = generation_params.copy()
        current_params["seed"] = base_seed + attempt - 1

        try:
            response = await create_chat_completion_with_retry(
                client=client,
                model_name=model_name,
                messages=messages,
                generation_params=current_params,
            )

        except BadRequestError:
            return GenerationResult(
                product_id=source_product["product_id"],
                card=None,
                valid=False,
                attempts=attempt,
                errors=all_errors + [ValidationReason.OTHER],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
        ):
            return GenerationResult(
                product_id=source_product["product_id"],
                card=None,
                valid=False,
                attempts=attempt,
                errors=all_errors + [ValidationReason.OTHER],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        if response.usage is not None:
            input_tokens += response.usage.prompt_tokens or 0
            output_tokens += response.usage.completion_tokens or 0

        try:
            if finish_reason == "length":
                raise ResponseValidationError(
                    reasons=[ValidationReason.OUTPUT_TRUNCATED],
                    message="Ответ модели был обрезан по max_tokens.",
                )

            card = parse_and_validate(content, source_product)

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
            all_errors.extend(error.reasons)

            if attempt == max_attempts:
                break

            messages = build_messages(source_product)

            if ValidationReason.EMPTY_RESPONSE in error.reasons:
                continue

            if ValidationReason.OUTPUT_TRUNCATED in error.reasons:
                messages.append({
                    "role": "user",
                    "content": build_retry_message(error),
                })
            else:
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": build_retry_message(error)},
                ])

    return GenerationResult(
        product_id=source_product["product_id"],
        card=None,
        valid=False,
        attempts=max_attempts,
        errors=all_errors,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def run_api_comparison(
        items: list[dict],
        client: AsyncOpenAI,
        model_name: str,
        generation_params: dict
):
    """Последовательно обрабатывает товары через OpenRouter."""
    results = []
    started = time.perf_counter()

    for index, product in enumerate(items, start=1):
        item_started = time.perf_counter()

        result = await generate_one_api(
            client=client,
            model_name=model_name,
            source_product=product,
            generation_params=generation_params,
        )

        latency = time.perf_counter() - item_started
        results.append((result, latency))

        print(
            f"{index}/{len(items)} {result.product_id}: "
            f"valid={result.valid}, "
            f"attempts={result.attempts}, "
            f"latency={latency:.2f} с"
        )

    return results, time.perf_counter() - started


async def run_local_comparison(
        items: list[dict],
        client: AsyncOpenAI
):
    """Последовательно обрабатывает те же товары через локальный vLLM."""
    results = []
    started = time.perf_counter()

    for product in items:
        item_started = time.perf_counter()

        result = await generate_one(
            client=client,
            model_name=LOCAL_MODEL_NAME,
            source_product=product,
            generation_params=GENERATION_PARAMS,
            max_attempts=MAX_ATTEMPTS,
        )

        results.append(
            (result, time.perf_counter() - item_started)
        )

    return results, time.perf_counter() - started


def calculate_comparison_metrics(
        run_results,
        elapsed_sec: float,
        model_name: str,
        api: bool = False
) -> dict:
    """Вычисляет метрики сравнения моделей."""
    results = [result for result, _ in run_results]
    latencies = [latency for _, latency in run_results]

    n = len(results)
    valid = sum(result.valid for result in results)
    input_tokens = sum(result.input_tokens for result in results)
    output_tokens = sum(result.output_tokens for result in results)

    metrics = {
        "model": model_name,
        "n": n,
        "valid": valid,
        "valid_rate": valid / n if n else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "avg_input_tokens": input_tokens / n if n else 0.0,
        "avg_output_tokens": output_tokens / n if n else 0.0,
        "elapsed_sec": elapsed_sec,
        "avg_latency": sum(latencies) / n if n else 0.0,
        "avg_attempts": (
            sum(result.attempts for result in results) / n
            if n else 0.0
        ),
    }

    if api:
        metrics["cost_per_1000"] = (
            (input_tokens * API_INPUT_PRICE
             + output_tokens * API_OUTPUT_PRICE)
            / 1_000_000 / n * 1000
            if n else 0.0
        )

        metrics["errors_by_reason"] = dict(Counter(
            error.value
            for result in results
            if not result.valid
            for error in set(result.errors)
        ))

    return metrics


async def main():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
    dev_data = load_jsonl("data/dev.jsonl")

    products = [
        prepare_product_input(
            product=product,
            tokenizer=tokenizer,
            max_input_tokens=MAX_INPUT_TOKENS,
        )
        for product in dev_data[:API_SAMPLE_SIZE]
    ]

    api_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=120,
        max_retries=0,
    )

    local_client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        timeout=120,
    )

    await generate_one(
        client=local_client,
        model_name=LOCAL_MODEL_NAME,
        source_product=products[0],
        generation_params=GENERATION_PARAMS,
    )

    api_results, api_elapsed = await run_api_comparison(
        items=products,
        client=api_client,
        model_name=API_MODEL_NAME,
        generation_params=GENERATION_PARAMS,
    )

    local_results, local_elapsed = await run_local_comparison(
        items=products,
        client=local_client,
    )

    api_metrics = calculate_comparison_metrics(
        api_results,
        api_elapsed,
        API_MODEL_NAME,
        api=True,
    )

    local_metrics = calculate_comparison_metrics(
        local_results,
        local_elapsed,
        LOCAL_MODEL_NAME,
    )

    save_json(
        "outputs/api_comparison_run.json",
        {
            "local": local_metrics,
            "api": api_metrics,
        },
    )

    print("Локальная модель:", local_metrics)
    print("OpenRouter:", api_metrics)


if __name__ == "__main__":
    asyncio.run(main())
