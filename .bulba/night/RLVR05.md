# RLVR05 — RLVR-учитель на базе симулятора

**Дата:** 2026-11-23
**Repo:** `/home/sehaxe/potok-m/sim` (git root `/home/sehaxe/potok-m`)
**Статус:** done, committed locally, not pushed.
**Commit:** `5942503` — `docs(sim): add RLVR teacher design with verifiable reward contract`

## Deliverable

Создан ровно один файл в репозитории: `sim/docs/RLVR_TEACHER.md`
(198 строк, `SPDX-License-Identifier: GPL-3.0-or-later`). Исходного кода,
конфигов и workflow не касался. Субагентов не использовал.

## Coverage

| Требование | Где в документе |
|---|---|
| Прошивка в контуре → генератор данных и учитель | §1 (SIL 500 Гц ↔ 50 Гц, шов `run_episode_with_control{,_ctx}`, `--control-file`, физическое отображение руля), §2 (роли среды и верификатора). |
| Верифицируемая формула награды | §3: `L = lateral_rms_m`, `A = apogee_m`, `H`, `S`, жёсткий гейт `g(τ)`, `R = w·…`, парный базовый прогон `L₀(i)`, `A_ref = 3127.363 м`. |
| Контракт роллаута / provenance | §4: `(seed, episode, scenario, config_hash, split, версия форматов)`, `SplitMix64`→`StdRng`, rayon бит-идентичен, `manifest.json` без меток времени/PID, 14 LOUD-гейтов, сплит 0–15 / 16–19. |
| Обучение и дистилляция | §5: клонирование по v3-целям 4–5, privileged teacher → history-студент на on-policy данных, GRPO-подобные групповые преимущества, фальсификация обнулением truth. |
| Реализовано / планируется | §6: таблица; явно отмечено, что самой функции `R` и RL-раннера нет. |
| Честные ограничения | §7: нет полётного лога, нет платы (`hw::build_board()` — `todo!`), награда не настроена, аэро-плейсхолдеры и открытое противоречие TTX, верификатор = симулятор. |

## Первичные источники

Проверены напрямую по `arxiv.org/abs` в этот же день: **arXiv:2411.15124** (Tulu 3,
вводит термин RLVR), **arXiv:2501.12948** (DeepSeek-R1),
**arXiv:2402.03300** (DeepSeekMath/GRPO). Три статьи по sim-to-real/RMA
(2107.04034, 1703.06907, 2007.01434) взяты из уже проверенной таблицы
`ai/docs/IDEAL_DATA_RESEARCH.md` §2. Ни одного непроверенного ID нет.

## Constraints observed

- Только один файл в `sim/` создан; существующие untracked-файлы другой lane
  (`core/src/aero_derivatives.rs`, `core/src/flight_replay.rs`,
  `core/src/identification.rs` и др.) не трогались и не стейджились.
- `git add` по явному пути, без `git add .`; `git diff --cached --check` чистый;
  в коммите только один файл. Push не делался.
- Источники — только файлы репозитория и первичные публикации; чаты/переписка
  источником не указаны.

## Verification

- `git status --short` до коммита: в индексе только `sim/docs/RLVR_TEACHER.md`.
- `git diff --cached --check`: clean.
- Документ проверен на случайные не-русские вкрапления: 0 совпадений.
- Documentation-only: сборка и тесты не запускались, никаких результатов не заявлено.
