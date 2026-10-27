# Поток-М

Один репозиторий на весь проект: документация, код и артефакты.

**Сайт:** https://rocket-team-m.github.io/potok-m/

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
