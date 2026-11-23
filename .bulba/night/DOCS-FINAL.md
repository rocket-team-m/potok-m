# DOCS-FINAL — полная документация Potok-M и RocketCraft

Дата: 23.11.2026. Репозиторий: `/home/sehaxe/potok-m`. Сайт: `docs/` (Astro Starlight),
внешняя страница — `site/`. Без субагентов.

## Scope

- Все страницы `docs/src/content/docs/**/*.md` и `*.mdx`, кроме `process/decisions.md`
  (владелец — другой агент): переписаны существующие 16 и добавлены 14 новых.
- `docs/astro.config.mjs` — только массив `sidebar` (13 разделов, 30 страниц).
  Базовый путь `/potok-m/docs`, заданный параллельной задачей, сохранён.
- `ai/docs/MODEL_CARD.md` — по отдельному указанию обновлён раздел A/B.
- `.github/workflows/guard.yml` — минимальная правка link-check: SITE_ROOTS теперь
  принимает и `/potok-m/docs/`, и legacy `/potok-m/`. Без этого новая база сайта
  делала гейт несовместимым с переходом документации в подкаталог.
- Отчёт — этот файл. `README.md`, `SIM_FIDELITY.md` и код не тронуты.

## Что написано

| Раздел | Страницы | Содержание |
| --- | --- | --- |
| Проект | goals, architecture, roadmap, roles, licenses, run | карта 8 публичных репозиториев + приватный RocketCraft, поток данных, ТТХ, лицензии, команды запуска и проверки, честный список ограничений |
| Борт | block-diagram, bom, recovery, search | STM32H743, датчики, контуры управления, черновой BOM, спасение и поиск без выдуманных пинов |
| ИИ | concept, models, training, telemetry, safety, gates | LRS-TCN/RMS-TCN, LoopWM, LoopWM-lite, JEPA/MTP, быстрая модель с MAC 52 896, цель v3, watchguard, калибровка, 70 ночных hard-gate, v3 A/B |
| Симулятор | model, validation | 6-DoF/RK4 2 мс, ветер и Драйден, датчики, 20 сценариев, OR parity, аэрокарточка, манифест, 61 ночной тест + Wave B |
| Протокол и edge | uart, embedded, inference | 56/14 и 42/16 байт, CRC-16 `0x29B1`, seq, кросс-языковой golden, `no_std`, Embassy, SIL, 856 циклов, форматы v2/v3 |
| Форма | cad, aero | STEP/STL и Releases, OR parity, граница аэрокарточки |
| RocketCraft | overview, ui | C01–C24 с текущими статусами, DAG и crates, солверы, демо, GPUI, темы, визуальный язык, 0/12 скриншотов |
| Полёты | protocol, log | регламент, 12 подписанных HIL-гейтов, пусков нет |
| Процессы | work, glossary | правила, 40+ терминов, ссылки на гейты |

## Канонические факты и источники

- V3 A/B: **PASS**, `ab_report.json/.md` (commit `1750599`). `loopwm` 3/3 seed:
  RMS 1528,366 → 1502,180 м (−26,19; 98,29 %), апогей 1734,977 → 1860,093 м
  (+125,12) на seed 1; `lite` в пределах 0,03 м; `lrs` выигрывает seed 1/2,
  seed 3 отсутствует (`runs/lrs/s3/lrs.bin`), не измерен; oracle 1528,366 →
  1035,228 м. В карточке модели удалено утверждение, что v3 A/B не запускался.
- OR parity: 1,86 % апогей, 1,45 % vmax, 0,14 с, 4,95 % высота — `SIM_FIDELITY.md`,
  `SPEC_ENGINE_IDEAL.md`, `or_parity_extended_tests.rs`. Это регрессионные пины.
- Гейты прошивки: 216 host-тестов, 25 SIL-отказов, 34 тайминга, 856 циклов
  худшего сценария, 24 000 циклов бюджет — `VERIFY-FW.md`, `SIL04.md`, `SIL05.md`.
- Edge: p99 на Pi не измерен; v3 writer и `Runtime::load` для `POTOKM3\0`
  отсутствуют — `EDGE02.md`, `POLISH-EDGE.md`.
- RocketCraft: статусы C01–C24 из `README.md` воркбенча; визуальный долг 7 из 11
  замечаний и 0/12 скриншотов — `VISUAL_STATUS.md`, `VISUAL_AUDIT.md`, `FIX-WB.md`.

Числа даны без истории изменений; там, где данные относятся к разным деревьям
(монорепозиторий против split-worktree), это указано явно.

## Проверка

```
cd docs && bun run build                       -> exit 0, 32 page(s) built
link-check из .github/workflows/guard.yml       -> checked 52 local navigation links,
                                                  skipped 2 asset links, 0 broken
license-gate из guard.yml                      -> OK: own code is GPL-3.0-or-later
git diff --cached --check                       -> clean
проверка sidebar <-> content                    -> 30/30, 0 отсутствующих, 0 лишних
сканирование на CJK/эмодзи/провенанс/заглушки  -> 0 совпадений
проверка built HTML                        -> 4 ссылки в decisions.md (вне scope)
```

Предупреждения сборки (`i18n` collection, отсутствующий 404, чанк >500 kB) —
существовали до этой задачи и не относятся к содержанию.

## Вне scope / остаточные риски

- `process/decisions.md` (другой владелец) всё ещё использует префикс
  `/potok-m/...` в четырёх внутренних ссылках; guard их принимает через
  legacy-fallback, но в собранном HTML они ведут мимо `/docs`. Нужна правка
  владельцем файла.
- Реального лётного лога нет; плата не вводилась в строй; симулятор не цифровой
  двойник. Положительный v3 A/B получен только в симуляции.
- Визуальный гейт RocketCraft красный по замыслу; точка съёмки и 12 скриншотов
  отсутствуют.
