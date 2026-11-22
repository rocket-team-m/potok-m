# Поток-М: ИИ-стек

Обучение моделей вспомогательного контура управления ракеты «Поток-М» на Python и бортовой inference на Rust (Raspberry Pi) в одном репозитории.

> **Статус: прототип, не сертифицирован для полёта.** Всё, что есть сейчас, — CPU-тесты и сверка с эталоном формата на синтетических данных. Замеров на Raspberry Pi и полётного лога нет; детали — в разделе «Ограничения».

Python и Rust работают с одним и тем же форматом модели **v2** (`FORMAT.md`, magic `POTOKM1\0`, версия 2). Python — только обучение и экспорт весов, Rust — только бортовой inference и dispatch. Общий эталонный файл `tests/fixtures/tiny_model.bin` + `tests/fixtures/tiny_expected.npy` используется обеими сторонами для сверки (допуск 1e-6).

## Быстрый старт

### Python (обучение)

Требуется Python ≥ 3.12 и [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # CPU-зависимости (torch 2.15.0+cpu)
uv run pytest -q             # тесты
uv run ruff check src tests  # линтер
```

Единый CLI для всех моделей — `potok_ai.engine`:

```bash
PYTHONPATH=src uv run python -m potok_ai.engine --help

PYTHONPATH=src uv run python -m potok_ai.engine \
  --model loopwm-lite --stage train \
  --data <путь к v2-файлу> --out /tmp/potok-lite \
  --epochs 1 --seed 3 --device cpu

PYTHONPATH=src uv run python -m potok_ai.engine \
  --model loopwm-lite --stage eval \
  --data <путь к v2-файлу> --out /tmp/potok-lite --device cpu
```

- `--model`: `lrs`, `loopwm-lite`, `loopwm`.
- `--stage`: `train`, `distill`, `eval` (все три семьи моделей обучаются от учителя, поэтому `distill` — тот же путь, что `train`; `eval` только загружает экспорт и считает метрики).
- Результат: таблица метрик в stdout и `metrics.json` в `--out` (`rmse_train`, `rmse_held_out`, `ratio`, `params`, `mac`, `bytes`).

### Rust (бортовой inference)

```bash
cd edge
cargo test
cargo check
```

Дополнительно — генерация фикстуры и замер (на x86-64 это только ориентир, не замер на Pi):

```bash
cargo run --bin make-fixture -- --out /tmp/potok-model.bin
cargo run --release --bin bench -- --model /tmp/potok-model.bin --frames 2000
```

## Структура

| Путь | Что там |
|---|---|
| `src/potok_ai/` | Модели и обучение: `model.py` (SmartTeacher, TCN, TSCT), `loopwm.py`, `loopwm_lite.py`, `student.py` (LRS-RNN), `engine.py`, `train*.py`, `distill_student.py`, `export.py`, `data.py`, `optim.py`, `jepa.py` |
| `tests/` | pytest-проверки и эталонные фикстуры (`fixtures/`) |
| `FORMAT.md` | Авторитетное описание бинарного формата v2 и входного `validation.bin` |
| `BUILD_LOG.md`, `edge/BUILD_LOG.md` | Решения, оговорки и результаты прогонов |
| `edge/src/` | Runtime: `format_v2.rs`, `student_runtime.rs`, `speculative.rs`, `runtime.rs`, `uart.rs`, `frame.rs`, `simd.rs`, `lowlat.rs` |
| `edge/src/bin/` | `infer`, `bench`, `bench-spec`, `make-fixture` |
| `edge/tests/` | Интеграционные тесты Cargo, включая сверку с Python-фикстурами |

Тяжёлые чекпоинты и датасеты в git не лежат — они в Releases.

## Архитектура

- **Физические признаки.** Симулятор выдаёт окно из 14 физических каналов (8 датчиков + `alpha_deg`, `beta_deg`, `q_dyn_kpa`, `a_lat_resid`, `drag_est`, `static_margin_cal`) и 6 чисел состояния ракеты (`mass_kg, area_m2, cd0, static_margin_cal, thrust_peak_n, gyro_noise`). Итого 20 чисел на кадр. Разделение по сценариям ветра/порывов, не по «типам ракет».

- **Учитель LoopWM.** Привилегированная модель (`loopwm.py`; предшественник — `SmartTeacher` в `model.py`): один общий `LoopBlock` применяется итеративно, выход — две ограниченные поправки и confidence. Истинная поправка симулятора (ветер) участвует только в функции потерь, никогда не подаётся на вход. Обучение представлений идёт через JEPA-головы, предсказывающие латенты EMA-энкодера на 3/6/10 кадров вперёд; KoLeo не даёт представлениям схлопнуться. Число проходов определяется сходимостью обновления, максимум 6.

- **Студент LRS-RNN.** `StatefulTCN` (`student.py`) — компактная рекуррентная модель на один кадр: состояние каждого блока сохраняется между шагами и обновляется по правилу `h = 0.8·h_prev + 0.2·candidate`; линейные слои — тернарные TSCT. На выходе — две поправки (`tanh(g_max)·…`) и confidence. Сброс состояния — в начале эпизода и при смене кода фазы. Это та модель, которая едет на борту.

- **Дистилляция.** Учитель размечает окна, студент обучается на последнем кадре окна: MSE по поправкам плюс MSE по confidence к `exp(-|g_teacher - g_student| / conf_temp)`. Fast-JEPA и KoLeo улучшают само представление. Экспорт унифицирован: `export.py` пишет v2-файл с SHA256-хвостом, `Runtime::load` в Rust выбирает читатель по magic-байтам.

- **Спекулятивная проверка (`edge/src/speculative.rs`).** Студент считает каждый кадр; учитель вызывается только при низком confidence или по принудительному расписанию (каждые 20 кадров). Гарантия «не более одного вызова учителя за цикл» обеспечена структурой кода. Есть два профиля UART: `compat` (921600 бод, 70 Б/цикл) и `lowlat` (2000000 бод, 58 Б/цикл); выбор — флагом `--profile`, без автоопределения.

## Связанные репозитории

- Протокол обмена: [rocket-team-m/potok-m-protocol](https://github.com/rocket-team-m/potok-m-protocol)
- Симулятор и генератор валидационных файлов: [rocket-team-m/potok-m-sim](https://github.com/rocket-team-m/potok-m-sim)

Документация проекта: https://rocket-team-m.github.io/potok-m/

## Ограничения

- **Confidence не откалиброван.** Голова выдаёт сигмоидное значение, которое не является оценкой вероятности. Порог расхождения «учитель vs студент» в `3·g_max` при общих ограничениях голов недостижим, поэтому использованный в коде механизм проверки срабатывает только по принудительному расписанию.
- **Замеры на Raspberry Pi не проводились.** NEON-путь проверен компиляцией под aarch64, но не запускался; все числа по задержкам в `edge/BUILD_LOG.md` получены на x86-64 и являются ориентиром, не результатом для целевого устройства.
- **Полётного лога нет.** Обучение — только CPU-прогоны на синтетических/отладочных данных; симулятор ещё должен выпускать валидационные v2-файлы с реальными физическими признаками и флагами сценариев.

## Лицензия

MIT.
