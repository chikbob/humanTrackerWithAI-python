# Research Benchmark Layer

Набор инструментов для экспериментальной части магистерской работы:

- сценарии сравнений detector/tracker;
- запуск benchmark-прогонов по видео;
- сохранение результатов в SQLite;
- экспорт таблиц в `CSV` и `Markdown` для вставки в пояснительную записку.

## Основные сценарии

- `detector_comparison` — сравнение `yolov8n/s/m` при фиксированном `ByteTrack`;
- `tracker_comparison` — сравнение `ByteTrack`, `BoT-SORT` и `detect_only` для одной модели;
- `latency_profile` — сравнение latency/FPS между tracking и detect-only для одной модели.

## Запуск benchmark-прогона

```bash
python scripts/run_benchmark_suite.py \
  --scenario detector_comparison \
  --source /path/to/video.mp4 \
  --frame-limit 120 \
  --warmup-frames 10
```

## Экспорт последних результатов

```bash
python scripts/export_benchmark_results.py
```

Результаты экспортируются в `docs/research/benchmarks/` в форматах `csv` и `md`.
