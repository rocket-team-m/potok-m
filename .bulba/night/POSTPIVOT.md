# POSTPIVOT — RLVR снят с роли по умолчанию

Дата: 2026-11-24. Источник решения: `.bulba/night/RES-RLVR.md` (RES-RLVR).
Сабагентов не задавал. Лицензия затронутых файлов — GPL-3.0-or-later (SPDX на месте).

## Решение

Для непрерывного низкоразмерного контроллера прямой GRPO-полиградиент коллапсирует
(в единственном найденном head-to-head: Ant/Walker2d/Humanoid 1012.77 / 129.60 / 86.93
против 6609.79 / 5964.93 / 9484.00 у PPO). По умолчанию остаётся on-policy
привилегированная дистилляция (DAgger/RMA-стиль); верификатор сохраняется как
инструмент выбора чекпоинта и фальсификации, а не источник градиента.

## Что изменено (только 4 файла задачи)

- `ai/src/potok_ai/train_teacher_v2.py`
  - `STAGE_ORDER = ("sft", "privileged", "dagger", "distill")`; `RLVR_STAGE = "rlvr"`;
    новое поле `TeacherV2Config.rlvr: bool = False`.
  - `--rlvr` / `--stage rlvr` — opt-in пятая стадия. До `distill` снимаются снимки
    учителя после `sft`/`privileged`/`dagger`; `_verifier_score` оценивает каждый
    снимок в `torch.no_grad()` тем же `verifiable_reward`, лучший восстанавливается
    в модель, пишется `teacher_selected.pt` и блок `reward` в `metrics.json`.
  - Удалён градиентный контур RLVR: нет `_build_optimizer`, `backward`, advantage,
    policy loss и anchor в этой стадии (`details.optimizer_steps == 0`).
  - `--stage rlvr` выполняет стадии учителя + выбор, без дистилляции.
  - Верхнеуровневый `entropy` берётся из последней teacher-стадии (`dagger`),
    потому что `rlvr` и `distill` энтропию не несут.
- `ai/tests/test_train_teacher_v2.py` — гейты дефолтного порядка, opt-in порядка,
  отсутствия оптимизатора/backward в RLVR (`verifiable_reward` требует
  `not torch.is_grad_enabled()`), сохранённой provenance и файла чекпоинта.
- `ai/docs/TRAINING_RECIPE.md`, `ai/README.md` — новый порядок и команды запуска.

## Сохранено без изменений

Формула и веса награды (`REWARD_FORMULA`, `REWARD_WEIGHTS`), жёсткий гейт
(финитные `u`, `|u| <= g_max`), мягкий член риска, child stream `engine01:rlvr`,
`VerifiableReward`/`reward_provenance`, split и fingerprint DAgger, гейты RMSE /
generalisation ratio / ECE / entropy, `module_sources`. Физика, пороги и константы
не тронуты.

## Проверка (до коммита)

- `.venv/bin/ruff check src tests` → `All checks passed!`
- `.venv/bin/ruff format --check` на затронутых файлах → clean.
- `.venv/bin/python -m pytest -q tests/test_train_teacher_v2.py` → **23 passed**.
- `.venv/bin/python -m pytest -q` → **350 passed in 114.61s**.

Коммит: `e3da20c` — `refactor(ai): default teacher-v2 to DAgger distillation; RLVR opt-in`.
Push: origin main.

## Вне зоны задания (не трогал, требуется владельцу)

- `scripts/train_final_v3.py:187,207` сверяет `stages_completed` с `STAGE_ORDER`:
  старые пятистадийные артефакты больше не совпадают с новым дефолтом — их надо
  явно перезапустить или заархивировать.
- `ai/docs/TRAINING_RESULTS.md` и `.bulba/night/ENGINE01.md` описывают старый
  пятистадийный порядок; обновляются отдельной задачей.
- Оценка самого A/B (DAgger/privileged против GR2PO/Prism-GRPO) не проводилась:
  это решение о порядке стадий, а не измеренное превосходство качества.
