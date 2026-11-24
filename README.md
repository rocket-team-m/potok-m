# Поток-М

[![Docs](https://github.com/rocket-team-m/potok-m/actions/workflows/docs.yml/badge.svg)](https://github.com/rocket-team-m/potok-m/actions/workflows/docs.yml)
[![AI](https://github.com/rocket-team-m/potok-m/actions/workflows/ai.yml/badge.svg)](https://github.com/rocket-team-m/potok-m/actions/workflows/ai.yml)
[![Guard](https://github.com/rocket-team-m/potok-m/actions/workflows/guard.yml/badge.svg)](https://github.com/rocket-team-m/potok-m/actions/workflows/guard.yml)

Репозиторий команды: документация, сборка и общие артефакты. Код вынесен в отдельные репозитории.

**Сайт:** https://rocket-team-m.github.io/potok-m/

## Текущий честный статус (24.11.2026)

| Подсистема | Статус |
| --- | --- |
| ИИ | **v5 closed-loop A/B FAIL** (exit 1, 5 PASS / 6 FAIL): deployment-латераль 1473,06 / 1558,44 / 1496,79 м > лимита 1000 м; приросты −13,06 / −9,72 / −12,31 % латерали и +18,28 / +14,11 / +16,28 % апогея не дают −35 % / +15 %; oracle-контроль недостоверен по апогею. Гейты AIPOST1 не ослаблены, победы не заявлено. |
| Симулятор | OR-parity PASS — регрессионный пин, а не цифровой двойник; внешних измерений и реальных полётных логов нет. |
| Прошивка | host/SIL/Renode green (303 теста), HIL не пройден: не армить, не прошивать на аппарат. |
| Приложение / RocketCraft | app: 95 тестов зелёных, clippy `-D warnings` красный при CI без clippy; RocketCraft — приватный воркбенч, 3/24 shipped. |
| Лицензия | GPL-3.0-or-later, коммерческая лицензия отдельно; сторонние лицензии сохраняются. |

Полная сводка с цифрами, источниками и границами применимости — в
[текущем статусе](/potok-m/docs/project/status/). Формулировки «A/B пройден»
без номера детектора относятся к истории (v4 FAIL, F1 PASS в узком детекторе)
и не являются текущим вердиктом.

## Канонические репозитории

| Репозиторий | Что лежит |
| --- | --- |
| [potok-m-ai](https://github.com/rocket-team-m/potok-m-ai) | обучение модели, LoopWM/LRS-RNN, бортовой инференс на Rust |
| [potok-m-sim](https://github.com/rocket-team-m/potok-m-sim) | 6-DoF симулятор, рандомизация, генерация данных |
| [potok-m-protocol](https://github.com/rocket-team-m/potok-m-protocol) | UART-протокол, генератор биндингов |
| [potok-m-firmware](https://github.com/rocket-team-m/potok-m-firmware) | прошивка STM32H743 на Rust/Embassy |
| [potok-m-edge](https://github.com/rocket-team-m/potok-m-edge) | инференс на Raspberry Pi |
| [potok-m-3d](https://github.com/rocket-team-m/potok-m-3d) | CAD/STEP/STL, аэродинамические входы |
| potok-m-workbench | приватный RocketCraft: CAD/CFD/FEA/GUI |

Код вынесен в отдельные репозитории; здесь остаются только docs, hardware, vehicle, flights и общие артефакты;
изменения кодам вносить в канонических репозиториях.

## Структура

| Папка | Что внутри | Инструменты |
| --- | --- | --- |
| `docs/` | сайт документации (Astro Starlight) | bun |
| `ai/` | модель, обучение, симулятор, телеметрия | Python, uv |
| `hardware/` | электроника, авионика, спасение и поиск | KiCad |
| `vehicle/` | схемы, аэродинамические входы, OpenRocket | OpenRocket |
| `flights/` | протоколы, журнал пусков, разборы | — |

## Правила

- В git — только текст и мелкие файлы. Тяжёлые артефакты (CAD-сборки, логи
  полётов, результаты Ansys) живут в **Releases**; CI не пропустит файл больше
  2 МБ и бинарные расширения.
- Одна задача — одна ветка, в `main` только через pull request.
- Задачи — в Issues, доска — GitHub Projects.
- Секреты и личные данные в репозиторий не попадают.

## Локальный запуск сайта

```bash
cd docs
bun install
bun run dev      # http://localhost:4321/potok-m/
```

## Сборка

- `docs`: `bun run build` → `docs/dist`
- `ai`: `uv sync && uv run pytest -q`

Собственный код: GPL-3.0-or-later; коммерческая лицензия доступна отдельно. Third-party компоненты сохраняют свои лицензии.

Репозитории: [potok-m](https://github.com/rocket-team-m/potok-m) · [potok-m-ai](https://github.com/rocket-team-m/potok-m-ai) · [potok-m-edge](https://github.com/rocket-team-m/potok-m-edge) · [potok-m-sim](https://github.com/rocket-team-m/potok-m-sim) · [potok-m-protocol](https://github.com/rocket-team-m/potok-m-protocol) · [potok-m-firmware](https://github.com/rocket-team-m/potok-m-firmware) · [potok-m-3d](https://github.com/rocket-team-m/potok-m-3d).
