# Движок обучения: TCN, LoopWM-lite, LoopWM

Статус: черновик. Версия 1.

## Назначение

Один CLI для всех моделей и один формат отчёта. Модели обучаются в цепочке:

```
LoopWM (полный, офлайн) → LoopWM-lite → LRS-RNN → edge
```

## CLI

```bash
PYTHONPATH=src uv run python -m potok_ai.engine \
  --model loopwm|loopwm-lite|lrs \
  --stage train|distill|eval \
  --data VALIDATION.bin \
  --out DIR --epochs N --seed N --device cpu
```

## Общий конфиг

- окно 50 кадров × 14 признаков;
- Conditioning: 6 чисел;
- split по `scenario_<id>` из v2 метаданных симулятора;
- метрики: RMSE по сценариям, held-out/train ratio, число параметров,
  MAC на вывод, размер экспорта;
- три сида на конфигурацию по умолчанию.

## Адаптеры моделей

| Имя | Реализация | Роль |
|---|---|---|
| `lrs` | `student.py` | бортовой студент, 1 кадр за тик |
| `loopwm-lite` | новый `loopwm_lite.py` | бортовой верификатор, 16 латентных токенов |
| `loopwm` | `loopwm.py` | привилегированный учитель, офлайн |

Каждый адаптер предоставляет `build`, `train_step`, `eval`, `export`, `load`.

## LoopWM-lite

- Perceiver-encoder: 50×14 → 16 латентных токенов ширины 64.
- Один parameter-shared Transformer-блок, 2 итерации с adaptive exit.
- Головы: latent через 3 и 6 кадров, поправка, confidence.
- Цели: латент полного LoopWM и reference correction.
- MAC: около 0,2–0,4 млн на вызов, оценка 0,3–0,6 мс на Pi.
- Вызывается на борту раз в 20 кадров или при низком confidence студента.

## Отчёт

Движок печатает одну таблицу на прогон:

| модель | RMSE train | RMSE held-out | ratio | params | MAC | bytes |

Плюс JSON в `--out/metrics.json` для сборки ablation-матрицы.

## Разделение репозиториев

Пока один репозиторий `potok-m` с каталогами `ai/`, `protocol/`, `sim/`, `edge/`.
Разделение на отдельные репозитории делать через `git subtree split` после
стабилизации API. Без одобрения владельца новые remote-репозитории не создаются.
