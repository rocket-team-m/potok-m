# Поток-М

[![Docs](https://github.com/rocket-team-m/potok-m/actions/workflows/docs.yml/badge.svg)](https://github.com/rocket-team-m/potok-m/actions/workflows/docs.yml)
[![AI](https://github.com/rocket-team-m/potok-m/actions/workflows/ai.yml/badge.svg)](https://github.com/rocket-team-m/potok-m/actions/workflows/ai.yml)
[![Guard](https://github.com/rocket-team-m/potok-m/actions/workflows/guard.yml/badge.svg)](https://github.com/rocket-team-m/potok-m/actions/workflows/guard.yml)

Репозиторий команды: документация, сборка и общие артефакты. Код вынесен в отдельные репозитории.

**Сайт:** https://rocket-team-m.github.io/potok-m/

## Канонические репозитории

| Репозиторий | Что лежит |
| --- | --- |
| [potok-m-ai](https://github.com/rocket-team-m/potok-m-ai) | обучение модели, LoopWM/LRS-RNN, бортовой инференс на Rust |
| [potok-m-sim](https://github.com/rocket-team-m/potok-m-sim) | 6-DoF симулятор, рандомизация, генерация данных |
| [potok-m-protocol](https://github.com/rocket-team-m/potok-m-protocol) | UART-протокол, генератор биндингов, план прошивки STM32 |

Папки `ai/`, `sim/`, `protocol/`, `edge/` здесь остаются как рабочие копии;
изменения кодам вносить в канонических репозиториях.

## Структура

| Папка | Что внутри | Инструменты |
| --- | --- | --- |
| `docs/` | сайт документации (Astro Starlight) | bun |
| `ai/` | модель, обучение, симулятор, телеметрия | Python, uv |
| `hardware/` | электроника, авионика, спасение и поиск, прошивка | KiCad, C |
| `vehicle/` | SolidWorks, чертежи, Ansys и OpenRocket | SolidWorks, Ansys |
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
