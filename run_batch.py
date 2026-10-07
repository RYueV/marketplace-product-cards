import asyncio
import time

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from generation import generate_one
from io_utils import load_jsonl, save_jsonl, save_json
from long_input import prepare_product_input
from check_metrics import calculate_run_stats, calculate_metrics


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

GENERATION_PARAMS = {
    "temperature": 0.5,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_tokens": 250,
    "seed": 42,
}

MAX_INPUT_TOKENS = 1500
MAX_ATTEMPTS = 2
CONCURRENCY = 48

BENCHMARK_PATH = "data/benchmark.jsonl"
PREDICTIONS_PATH = "outputs/predictions.jsonl"


async def generate_timed(
        product,
        client,
        model_name,
        generation_params
):
    """Генерирует одну карточку и измеряет её latency."""
    started = time.perf_counter()

    result = await generate_one(
        client=client,
        model_name=model_name,
        source_product=product,
        generation_params=generation_params,
        max_attempts=MAX_ATTEMPTS
    )

    latency = time.perf_counter() - started

    return result, latency


async def run_concurrent(items, concurrency, generate_one):
    """Отправляет запросы параллельно, не больше concurrency одновременно."""
    sem = asyncio.Semaphore(concurrency)

    async def wrapped(item):
        async with sem:
            return await generate_one(item)

    started = time.perf_counter()

    results = await asyncio.gather(
        *[wrapped(item) for item in items]
    )

    elapsed = time.perf_counter() - started

    return results, elapsed


async def run_benchmark(
        items,
        concurrency,
        client,
        model_name,
        generation_params
):
    """Запускает пакетный прогон."""

    async def generate(product):
        return await generate_timed(
            product=product,
            client=client,
            model_name=model_name,
            generation_params=generation_params
        )

    run_results, elapsed = await run_concurrent(
        items=items,
        concurrency=concurrency,
        generate_one=generate
    )

    run_stats = calculate_run_stats(
        run_results=run_results,
        elapsed_sec=elapsed,
        model_name=model_name,
        concurrency=concurrency,
        generation_params=generation_params
    )

    metrics = calculate_metrics(
        run_results=run_results,
        elapsed_sec=elapsed
    )

    report = {
        "run": run_stats,
        "metrics": metrics,
        "api_comparison": {},
    }

    save_json(
        "outputs/report.json",
        report
    )

    return run_results, elapsed


async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        timeout=120
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    benchmark = load_jsonl(BENCHMARK_PATH)

    prepared_benchmark = [
        prepare_product_input(
            product=product,
            tokenizer=tokenizer,
            max_input_tokens=MAX_INPUT_TOKENS
        )
        for product in benchmark
    ]

    run_results, elapsed = await run_benchmark(
        items=prepared_benchmark,
        concurrency=CONCURRENCY,
        client=client,
        model_name=MODEL_NAME,
        generation_params=GENERATION_PARAMS
    )

    predictions = []

    for result, _ in run_results:
        if result.valid:
            predictions.append(
                result.card.model_dump()
            )
        else:
            predictions.append(
                {
                    "product_id": result.product_id,
                    "valid": result.valid,
                    "attempts": result.attempts,
                    "errors": [
                        error.value
                        for error in result.errors
                    ],
                }
            )

    save_jsonl(
        PREDICTIONS_PATH,
        predictions
    )

    valid_count = sum(
        result.valid
        for result, _ in run_results
    )

    print(f"Обработано карточек: {len(run_results)}")
    print(f"Валидных карточек: {valid_count}")
    print(f"Общее время, с: {elapsed:.3f}")


if __name__ == "__main__":
    asyncio.run(main())