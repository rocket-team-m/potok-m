# SITE05 — исправление дефектов SITE04 на сайте

Дата: 23.11.2026. Репозиторий: `/home/sehaxe/potok-m`, ветка `main`. Субагентов нет.
Исходный список дефектов — `.bulba/night/SITE04.md`. Все перечисленные в задании
пункты исправлены; сайт-файлы собираются и рендерятся.

## Что исправлено

### P1 · Эпохи A/B больше не противоречат друг другу

- `site/fragments/status.html`: вместо одной строки «v3 A/B PASS» — две явно
  размеченные эпохи:
  - `AI · v3 (текущая эпоха)` → `A/B PASS 3/3`, loopwm 3/3: латераль
    1528,366 → 1502,180 м (−26,19 м), апогей +125,12 м; `lrs` seed 3
    отсутствует; не лётная квалификация.
  - `AI · g_theta (архив)` → `A/B FAIL · отозван`, студенты 40/60/100 эпох
    с угловой целью не обыгрывали zero command; прогон v1/v2 не является
    контрольной точкой v3.
- `site/index.html`: в schematic-журнале `VERDICT PASS / 3 / 3` заменено на
  `V3 A/B PASS / loopwm 3/3`. Все числа сохранены без изменений.
- Источник исторической формулировки — `ai/docs/MODEL_CARD.md:121-124`;
  текущего вердикта — `ab_report.json`, `docs/src/content/docs/ai/gates.md:29-32`.

### P2 · Мобильная навигация

В `@media (max-width: 760px)` больше нет `nav { display: none; }`. Шапка
разворачивается, а пять ссылок идут горизонтальной лентой с `overflow-x: auto`
и видимой верхней границей. Новых файлов нет.

### P3 · Skip-link

Первый ребёнок `<body>` — `<a class="skip" href="#main">Перейти к содержимому</a>`;
`<main id="main">`. Ссылка появляется при `:focus`.

### P4 · OG-карточка документации

- `docs/src/components/Head.astro` (новый) — оверрайд компонента `Head`
  Starlight: рендерит стандартный `Head` и добавляет `og:image`, `og:image:*`,
  `twitter:image`, alt.
- `docs/astro.config.mjs` — `components: { Head: './src/components/Head.astro' }`.
- `docs/public/assets/og.png` (новый) — копия канонической карточки.

Отклонение от формулировки SITE04: URL картинки —
`https://rocket-team-m.github.io/potok-m/docs/assets/og.png`, а не
`.../potok-m/assets/og.png`. Причина: копия лежит в `docs/public/`, и такой URL
работает и на задеплоенном сайте, и в локальном `bun run dev` / previews;
лендинг продолжает использовать свой `.../potok-m/assets/og.png`.

### P5 · Mermaid

`docs/src/components/Mermaid.astro`: тема берётся из
`document.documentElement.dataset.theme` (`dark`/`neutral`), `fontSize: 15px`,
`flowchart.useMaxWidth: false`, `overflow-x: auto` и `:global(svg) { max-width: none; }`
(иначе глобальное правило Starlight сжимает SVG, введённый в DOM уже после
компиляции компонента). `MutationObserver` на `data-theme` возвращает исходный
код из `data-source` и перерисовывает диаграмму при смене темы.

### Прочее

- `site/assets/og.svg`: `500 ГЦ` → `500 Гц`; `og.png` перегенерирован
  `rsvg-convert -w 1200 -h 630` (1200×630, 47 396 байт).
- htmx: CDN-скрипт заменён локальным `site/assets/htmx-2.0.4.min.js`
  (0BSD, пометка в шапке файла) с SRI `sha384-5TCF...T86` и `crossorigin="anonymous"`;
  добавлен `<noscript>` с отсылкой к документации.
- Ссылки на организацию `https://github.com/rocket-team-m` →
  `https://github.com/rocket-team-m/potok-m` в `site/index.html` (2) и
  `docs/src/content/docs/index.mdx` (1).
- GPL-3.0-or-later и коммерческая оговорка не менялись; лицензионный гейт
  репозитория затронут только добавлением 0BSD-заголовка к вендорённому htmx.

## Проверка

1. `cd docs && bun run build` — 32 страницы, `Complete!`, без ошибок
   (только штатные warnings Astro про `i18n`, размер чанка и `404.html`).
2. Локальный **link check** из `.github/workflows/guard.yml`:
   `checked 52 local navigation links, skipped 9 asset links`, битых нет.
3. Сборка как в CI (`docs/dist` → `/docs`, поверх `site/.`) и
   `python3 -m http.server 8771`; `curl` по страницам:
   - лендинг: skip-link, `main#main`, `V3 A/B PASS`, `loopwm 3/3`, локальный
     htmx + SRI, ссылка на репозиторий; старых `VERDICT PASS` / `500 ГЦ` /
     ссылок на организацию нет;
   - фрагмент статуса: обе эпохи и числа на месте;
   - `/docs/`: `og:image` и `twitter:image` присутствуют;
   - `/assets/og.png`, `/docs/assets/og.png`, `/assets/htmx-2.0.4.min.js`,
     `/docs/hardware/block-diagram/`, `/docs/ai/gates/` — все HTTP 200.
4. Playwright, реальный рендер (390×844 и 1280×900):
   - мобильный `nav`: `display: flex`, все 5 ссылок видимы, горизонтального
     overflow нет (`scrollW 375 <= innerW 390`); `htmx` загружен, клик по
     «Статус» подставляет фрагмент и ставит `aria-current="true"`;
   - docs Mermaid при `data-theme="dark"`: `data-processed=1`, заливка узла
     `rgb(31,32,32)`, текст `rgb(204,204,204)`, `font-size 15px`, SVG 1293 px
     при ширине колонки 617 px с `overflow-x: auto` (прокрутка, а не сжатие);
     после переключения на `light` — `rgb(238,238,238)` / `rgb(0,0,0)`,
     обратно на `dark` — снова тёмные узлы.
   - консоль на текущей странице: 0 ошибок.

## Не делал (в задании не было)

- P6 (мёртвый `Inter` в font-stack), P8 (реальная фигура вместо схематичного
  превью), P11 (`[01]` у бренда) и удаление осиротевших `og.svg` / `banner.svg`
  — не трогал: это отдельные решения, а не дефекты из списка задания.
- `ai/README.md` и `ai/docs/MODEL_CARD.md` продолжают описывать старую эпоху
  как текущую; это канонические репозитории ИИ, сюда не входят.

Сайт-файлы изменены, отчёт записан в `.bulba/night/SITE05.md`.
