from collections import Counter


GPU_PRICE_PER_HOUR = 65
BENCHMARK_SIZE = 300


def calculate_run_stats(
        run_results,
        elapsed_sec: float,
        model_name: str,
        concurrency: int,
        generation_params: dict
) -> dict:
    """Собирает фактически измеренные данные пакетного прогона."""

    results = [result for result, _ in run_results]
    latencies = [latency for _, latency in run_results]

    return {
        "model": model_name,
        "concurrency": concurrency,
        "params": generation_params,
        "elapsed_sec": elapsed_sec,
        "input_tokens": sum(
            result.input_tokens
            for result in results
        ),
        "output_tokens": sum(
            result.output_tokens
            for result in results
        ),
        "n": len(results),
        "avg_latency": (
            sum(latencies) / len(latencies)
            if latencies
            else 0.0
        ),
        "avg_attempts": (
            sum(result.attempts for result in results) / len(results)
            if results
            else 0.0
        )
    }


def calculate_metrics(
    run_results,
    elapsed_sec: float,
    gpu_price_per_hour: float = GPU_PRICE_PER_HOUR,
    benchmark_size: int = BENCHMARK_SIZE
) -> dict:
    """Вычисляет метрики качества, скорости и стоимости."""

    results = [result for result, _ in run_results]

    n = len(results)

    valid = sum(
        result.valid
        for result in results
    )

    valid_rate = (
        valid / n
        if n
        else 0.0
    )

    throughput = (
        n / elapsed_sec
        if elapsed_sec > 0
        else 0.0
    )

    time_for_300_sec = (
        benchmark_size / throughput
        if throughput > 0
        else 0.0
    )

    cost_vllm_per_1000 = (
        (elapsed_sec / 3600 * gpu_price_per_hour)
        / n
        * 1000
        if n
        else 0.0
    )

    errors_by_reason = Counter(
        error.value
        for result in results
        if not result.valid
        for error in set(result.errors)
    )

    return {
        "valid": valid,
        "valid_rate": valid_rate,
        "throughput": throughput,
        "time_for_300_sec": time_for_300_sec,
        "cost_vllm_per_1000": cost_vllm_per_1000,
        "errors_by_reason": dict(errors_by_reason)
    }