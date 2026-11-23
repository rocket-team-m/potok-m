# SITE03 — реальные ассеты в документации: UI, архитектура, ИИ, A/B, телеметрия

Дата: 23.11.2026. Репозиторий: `/home/sehaxe/potok-m`. Без субагентов.

## Scope

- `docs/src/assets/**` — 7 новых файлов: 3 PNG-скриншота RocketCraft и 4 SVG.
- `docs/src/content/docs/ai/{concept,gates,telemetry}.md` — по одному SVG, короткие
  фактические подписи.
- `docs/src/content/docs/project/{architecture,run}.md` — 3 скриншота и
  архитектурный SVG; в `architecture.md` Mermaid-схема потока данных заменена
  SVG-схемой (компонент `Mermaid.astro` остаётся в использовании —
  `hardware/block-diagram.mdx`).
- `docs/astro.config.mjs` — не менялся: все страницы уже в сайдбаре, новые
  разделы не добавлены.
- Отчёт — этот файл. Пути вне scope (включая чужие рабочие изменения в `ai/out/`)
  не тронуты.

## Ассеты

| Файл | Источник | Что показано |
| --- | --- | --- |
| `rocketcraft-workspace-dark.png` | `potok-m-workbench/docs/screenshots/baseline/dark-1920x1080-workspace.png` | пустое рабочее место, тёмная тема, 1920×1080, 30 085 B |
| `rocketcraft-demo-light.png` | `.../light-1920x1080-demo.png` | `Sample Rocket`, 4 детали, Mesh/Aero/FEA — NOT RUN, 58 588 B |
| `rocketcraft-palette-dark.png` | `.../dark-1920x1080-palette.png` | палитра по Ctrl+K, 14 команд, 39 991 B |
| `system-architecture.svg` | `project/architecture.md` | 8 публичных репозиториев + приватный воркбенч, поток от аэрокарточки до журнала, пунктир — обратная связь |
| `ai-pipeline.svg` | `ai/concept.md`, `ai/safety.md` | офлайн-обучение и бортовой контур, UART-кадры, интерлок, цель v3 |
| `ab-results-v3.svg` | `ab_report.json` | боковое RMS и апогей по 3 сидам, 5 рук; `lrs` seed 3 — «н/д» (нет экспорта) |
| `telemetry-trajectory.svg` | `ai/out/v3_data/held-out/validation.trajectory` | средняя высота и RMS бокового расстояния, 20 эпизодов × 1500 тиков |

Все PNG — подлинные снимки из визуальной матрицы воркбенча, не отрендеренные
заново; каждый < 60 КБ и < 2 МБ. SVG имеют SPDX `GPL-3.0-or-later`.

## Происхождение чисел

- A/B: `ab_report.json`, `exit_code = 0`, `verdict.pass = true`, config hash
  `1969edcc…79fed6`. В график попали агрегаты `seeds.*.{baseline,loopwm,lite,lrs,oracle_gtheta}.aggregate.{assisted,baseline}`:
  например, seed 1 — боковое RMS 1528,366 → 1502,180 м, апогей 1734,977 →
  1860,093 м (`loopwm`); oracle — 1035,228 м / 3971,197 м. Seed 3 для `lrs`
  показан как «н/д»: экспорта нет, подстановок не делалось.
- Траектория: `validation.trajectory` v3 (stride 26, 20 эпизодов × 1500 тиков),
  SHA-256 `0be79f08…128c9ee` — тот же файл, что записан в `ab_report.json` как
  baseline trajectory. Из него посчитаны средний апогей 1734,9775 м и боковое
  RMS 1528,3663 м по всем 30 000 отсчётам; оба совпадают с числами A/B отчёта.
  Кривые — `mean_z(t)` и `sqrt(mean(x²+y²))` по 20 эпизодам, прорежены до 503 точек.
- Скриншоты: реальные capture-файлы из приватного `potok-m-workbench`; в подписях
  явно сказано, что это снимки интерфейса, а не baseline визуальной матрицы
  (её гейт остаётся красным — 0 из 12).

## Проверка

| Проверка | Результат |
| --- | --- |
| `python3 xml.etree.ElementTree` на 4 SVG | валидны |
| `rsvg-convert` + визуальный осмотр всех 4 SVG | прочитаны, дефекты вёрстки исправлены |
| `bun run build` (docs) | `32 page(s) built`, 7 ассетов оптимизировано, Pagefind готов |
| Результаты сборки в `dist` | все 7 URL начинаются с `/potok-m/docs/_astro/`, файлы существуют |
| Link-check из `.github/workflows/guard.yml` (тот же Python) | `checked 52 local navigation links, skipped 9 asset links` |
| `bash .github/scripts/guard.sh` | `OK: тяжёлых файлов в git нет.` |
| `git diff --check` по индексируемым путям | чисто |
| SPDX в SVG | `GPL-3.0-or-later`, конфликтующих деклараций нет |

Astro сам переводит исходные PNG в WebP при сборке; в git остаются исходные PNG.

## Коммит

Только перечисленные выше пути (13 файлов), локальный коммит
`docs(site): add real UI captures and data-derived SVG diagrams`. Push не выполнялся.
