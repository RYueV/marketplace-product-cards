# Marketplace Product Cards

Рабочий репозиторий проекта генерации карточек товаров.

## Текущий этап

Подготовлен notebook для локальной модели Qwen/Qwen2.5-1.5B-Instruct через vLLM и эксперимента по `temperature`.

## Запуск vLLM

```bash
nohup vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 \
  --gpu-memory-utilization 0.9 --enforce-eager --port 8000 > vllm.log 2>&1 &
```

Проверка:

```bash
curl -s http://localhost:8000/v1/models
```

Основной notebook: `notebooks/stage2_generation_experiments.ipynb`.

Финальные обязательные файлы проекта (`run_batch.py`, `check_metrics.py`, `compare_api.py`, `outputs/predictions.jsonl`, `outputs/report.json`) будут добавлены на следующих этапах.
