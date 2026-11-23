# DATA01 — идеальный генератор обучающих данных (privileged teacher → student)

Статус: сделано. Документ: `ai/docs/IDEAL_GENERATOR.md` (180 строк, SPDX `GPL-3.0-or-later`).
Локальный коммит: `294722f` (`docs(ai): spec ideal v3 training-data generator`). Push не выполнялся.

## Что зафиксировано

- Разделение ролей v3: `z_priv` = слоты 0..21 POTOKTRJ v3 (22 → 8, нормализация только по train-сайдкару); метки = 22..25, из них авторитетная команда `[g_accel_y, −g_accel_x]` (24–25), `g_theta` (22–23) — только диагностика; student = 50×14 + 6 conditioning. Опоры: `sim/core/src/trajectory.rs:85-114`, `ai/src/potok_ai/privileged_teacher.py:51-70,215-270`, `ai/src/potok_ai/data_v3.py:6-14,41-60`.
- Кумулятивные стадии S0–S6 (nominal → gust → shear → compound → turbulence → sensor lag/faults → bounded adversary), отказ от плейсхолдера `scenario_id % 4`; темп 15 % calm replay + `U(0, c_stage)` с линейным рампом.
- Рандомизация 1-cos (амплитуда, момент, градиент через уже существующий `duration_factor_range`, оба знака), AR(1)/Dryden и von Kármán при согласованных `TI`/`L`, задержка датчика `lag_scale` 1.0–1.4 против OOD 1.5–2.0 с LOUD-проверкой непересечения; независимые поименованные DRAW-подпотоки.
- Удержанные ячейки 16–19 запечатаны; выбор модели leave-one-scenario-out по 0–15; DAgger и аугментация не могут коснуться оценки; раздельный eval seed, ≥3 seed на руку, печать окна.
- Provenance: `manifest.json` + `manifest.provenance.json` (config_hash, seed, split, hash'и файлов, git HEAD, protocol schema, golden episode) и построчная `Provenance` в DAgger; иных заявлений о происхождении нет.
- Метрики: относительное снижение бокового RMS `1 − assisted/zero` и относительный прирост апогея `assisted/zero − 1` из реальной траектории v3, с абсолютными метрами, n, тремя seed и oracle-положительным контролем. Ориентиры из `ab_report.md:23-43`: oracle −32.3 % / +129 %, текущие студенты ≈ −1.7 % / +7 %.
- Запреты: контаминация оценки, тихие подмены (нулевой truth/target, `g_theta` вместо команды), непромаркированная утечка привилегий, loss-only «привилегированный» учитель, сравнение на разных окнах, робастность по одному seed.

## Найденные расхождения (в документ включены)

- `ai/src/potok_ai/loopwm.py:1-10`, `train_loopwm.py:91-107`: истина идёт только в loss, `PrivilegedFeatureBuilder` в тренер не подключён — `SPEC_SMART.md:26-32` требует обратного.
- `ai/src/potok_ai/curriculum.py:23-29`: стадия по `id % 4` — плейсхолдер.
- `privileged_teacher.py:6-8`: порядок полей в докстринге расходится с `TRUTH_V3_TICK_NAMES` (`trajectory.rs:85-114`).
- `data.py:444-449`: window без authoritative target получает нулевую метку — источник тихой подмены.

## Проверка

Только документация: ни один исходный модуль, конфиг, тест или манифест не изменён; тесты не запускались (docs-only). Чужие незакоммиченные файлы в дереве не трогались. Сабагенты не использовались.
