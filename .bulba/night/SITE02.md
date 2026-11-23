# SITE02 — динамическая посадочная страница

Commit: `8540a53` (локально, без push). Изменён только `site/index.html`: CSS внутри `<style>` и `<body>`. HEAD-мета SITE01 (коммит `8364025`) не тронута.

## Что сделано

- **Оркестрованный reveal hero**: kicker → h1 → lede → кнопки, CSS-анимация с задержкой через `--i`; при `prefers-reduced-motion: reduce` всё сразу видно.
- **Телеметрия с реальными числами** из `ab_report.json` (seed 1, loopwm, 20 эпизодов, сценарии 16–19): анимированная линия `1528.366 → 1502.180 м`, `Δ −26.186 м`, `98.287 % от baseline`, апогей `+125.12 м`, OR-parity `1.86 %` при лимите `2 %`. Все значения округлены из JSON; ULP сидкара `6.1e-05 м` указан в подписи.
- **Два inline SVG-графика** по сценариям: боковое СКО и апогей (baseline vs assist), линии рисуются анимацией `stroke-dashoffset`; точки и подписи — mono, координаты взяты из строк `per_scenario` отчёта.
- **Схема контура** `sim → ai → edge → firmware` со ссылками на docs, нижним контуром телеметрии и границей ИИ (двигатель → ассистент выключен, STM32 → вето). На мобильном стрелки становятся вертикальными.
- **Полоса превью** из трёх schematic SVG-панелей (RocketCraft, sim validation, AI closed loop) со ссылками на `docs/rocketcraft/ui/`, `docs/sim/validation/`, `docs/ai/gates/`. Явная подпись: превью схематичны, не скриншоты — растровых скриншотов в репозитории нет.
- Визуальный язык сохранён: near-black/light, hairlines, mono-индексы, синий акцент; нет карточек, теней, эмодзи и градиентов. Фокус `:focus-visible` расширен, табы получили `aria-controls`/`role=tabpanel`, стрелки ←/→ и Home/End переключают табы. htmx-атрибуты `hx-get`/`hx-target`/`hx-trigger="load"` сохранены.

## Проверка

- `python3 -m http.server 8123 --directory site`: `/` → HTTP 200, 27522 B; `fragments/{overview,stack,status}.html` → 200.
- Playwright (http://127.0.0.1:8123/): htmx загружен (`typeof htmx === 'object'`), таб «Репозитории» меняет панель и `aria-current`, ArrowRight переводит на «Статус» и меняет панель; ошибок в консоли нет.
- Десктоп 1440×900, светлая тема и мобильный 390×844: скриншоты `/home/sehaxe/dormouse/.playwright-mcp/site02-{desktop-settled,light,mobile}.png`. Горизонтального переполнения на 390 px нет (`scrollWidth == clientWidth == 390`), сетки графиков и превью одноколоночные, `nav` скрыт.
- `prefers-reduced-motion: reduce`: hero `opacity: 1`, линия заполнена, `stroke-dashoffset: 0`, точки видимы сразу.

## Известные ограничения (не в скоупе)

- `site/fragments/status.html` всё ещё содержит «A/B OPEN · студенты пока не обыгрывают zero command», что противоречит `ab_report.json` (`verdict.status = PASS`). Файл фрагментов не трогался — нужна правка отдельной задачей.
- Ссылки `docs/` работают в собранном сайте (`base: /potok-m/docs`); локальный сервер из `site/` не содержит `docs/`, поэтому локальный curl их не проверяет.
- Первый скриншот «горячего» кадра делает hero невидимым до конца анимации — ожидаемое поведение оркестровки, проверено на устоявшемся кадре.
