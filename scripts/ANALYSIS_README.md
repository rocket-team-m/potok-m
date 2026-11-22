# Training analysis plots

Графики для отчётов в стиле opencode.ai: near-black `#0c0c0c`, near-white `#fbfbfb`,
акцент `#6b8fff`, Inter-like sans, моно-индексы, тонкая сетка, без эмодзи.

```bash
cd ~/potok-m/ai
uv run python ../scripts/analyze_training.py --out ../scripts/analysis_out --theme dark
uv run python ../scripts/analyze_training.py --out ../scripts/analysis_out_light --theme light
```

Вход по умолчанию:
- `ai/out/sweep/results.csv` — epoch × seed sweep;
- `scripts/ab_report.json` — финальный closed-loop A/B.

Выход:
- `rmse_curve.svg/.png` — train/held-out по эпохам, mean ± std по 3 сидам;
- `ece_curve.svg/.png` — raw vs calibrated ECE и gate 0.05;
- `ab_comparison.svg/.png` — baseline / oracle / студенты по боковому RMS и апогею;
- `analysis.md` — числовая таблица и вердикт.

`--plot-only` в ratatui-инструменте (`ai/tui`) делает static SVG в том же языке;
этот matplotlib-скрипт нужен для подробного разбора логов и вставки в PDF/отчёты.
