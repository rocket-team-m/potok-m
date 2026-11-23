<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# CHUTE01 — двухступенчатый парашют в симуляторе

**Статус:** модель и тесты готовы (9 гейтов зелёные), **интеграция в
`lib.rs` / `Config` / truth / manifest ещё не выполнена** — это отдельный шаг
владельца симулятора. Функция выключена по умолчанию, поэтому текущие A/B и
золотой хеш эпизода не меняются.

Источники: `WORLD08.md` §12 (десантирование и дрейф), `recovery.md`
(классическая логика, ИИ не допускается до пиросредств), `LATERAL01.md`
(1528 м латерального RMS не содержит ни метра под куполом).

## 1. Что реализовано

`core/src/parachute.rs` — самодостаточный детерминированный модуль:

- **`ParachuteConfig`** — конфигурация с `serde(default)`, `enabled = false`.
- **`DeployTrigger`** — типизированные триггеры: `altitude_m`,
  `time_since_apogee`, `time_since_launch`. Дополнительно
  `require_descent_confirm` запрещает раскрытие на наборе высоты.
- **`Parachute`** — конечный автомат `Stowed → DrogueInflating → Drogue →
  MainInflating → Main → Landed`. Две ступени, однаразовая защёлка: повторное
  срабатывание невозможно, `Landed` наступает ровно один раз, `reset()`
  начинает новый эпизод.
- **Физика**: вертикальная скорость релаксирует к `-target_descent_mps` с
  постоянной `inflation_s`; горизонтальная скорость релаксирует к **истинному**
  ветру с `wind_coupling_s`; положение — интеграл этих скоростей. Посадка —
  при достижении `landing_altitude_m`, с обнулением скорости.
- **Пружинное раскрытие, без пиро.** В конфиге нет пироканала и нет поля
  «механизм» — это константа модели, а не настраиваемое значение.
- **Без активного управления под куполом.** У `step` нет аргумента-команды;
  Подпись зафиксирована тестом (`fn(&mut Parachute, f64, FlightCtx)`),
  `AI_AUTHORITY_UNDER_CANOPY = false`.
- **LOUD-валидация.** `validate()` вызывается и при `enabled = false`: ошибка
  называет поле, значение и способ исправления; молчаливого clamp нет.
  Запрещены: нефинитные числа, неположительные триггеры, неположительные
  скорости/времена, основная ступень быстрее или выше/раньше дrogue,
  отрицательная высота посадки.

Точные числа — не измерение купола Potok-M, а настраиваемый плейсхолдер
(25 м/с дrogue, 6.5 м/с main). Измеренного аэродинамического профиля купола в
репозитории нет; модель даёт форму дрейфа `w_h × t_canopy`, а не сертифицированный
прогноз точки посадки.

## 2. Конфигурация (`config/default.toml`)

```toml
[parachute]
enabled = false
require_descent_confirm = true
wind_coupling_s = 3.0
landing_altitude_m = 0.0

[parachute.drogue]
trigger = { kind = "time_since_apogee", value = 2.0 }
target_descent_mps = 25.0
inflation_s = 1.2

[parachute.main]
trigger = { kind = "altitude_m", value = 1500.0 }
target_descent_mps = 6.5
inflation_s = 2.0
```

`run.duration_s = 30` (значение по умолчанию) **не хватает времени для полного
спуска**: при 6.5 м/с с 1500 м это ~230 с. Для события посадки нужно
увеличить длительность прогона.

## 3. Emission: truth и manifest

Имена объявлены в модуле, чтобы писатель и читатель не могли разойтись:

- `CHUTE_TRUTH_NAMES` (8 f32 на тик, в этом порядке): `chute_phase`,
  `chute_inflation`, `chute_drogue_t_s`, `chute_main_t_s`, `chute_vz_mps`,
  `chute_wind_x`, `chute_wind_y`, `chute_landed`. Значения даёт
  `ParachuteState::truth_floats(true_wind_xy)`; `-1.0` в полях времени означает
  «не срабатывал», `chute_landed` — 0/1.
- `CHUTE_MANIFEST_FIELDS` (9 полей): `chute_enabled`, `chute_drogue_t_s`,
  `chute_main_t_s`, `chute_landing_t_s`, `chute_deploy_altitude_m`,
  `chute_deploy_x_m`, `chute_deploy_y_m`, `chute_landing_dispersion_m`,
  `chute_landing_radius_m`. Значения даёт
  `LandingDispersion::manifest_named_floats(enabled)`.

**Dispersion считается отдельно от латерального RMS.** `lateral_rms_m`
(`trajectory.rs`) остаётся метрикой fin-полёта от взлёта до раскрытия. Поле
`chute_landing_dispersion_m` — это `LandingDispersion::lateral_offset_m()`,
расстояние от точки раскрытия до точки посадки (только дрейф под куполом);
`chute_landing_radius_m` — полный промах от стартовой площадки. Смешивать их
нельзя: RMS усредняет по времени, dispersion — однократное событие.

## 4. Что осталось сделать для интеграции

1. `core/src/lib.rs`: `pub mod parachute;` (и при желании `pub use`).
2. `core/src/config.rs`: поле `#[serde(default = "ParachuteConfig::default")]
   pub parachute: ParachuteConfig` в `Config`, и в `Config::validate()`:
   ```rust
   self.parachute
       .validate()
       .map_err(|e| SimError::Config(e.to_string()))?;
   ```
3. Эпизод: создать `Parachute` на эпизод; на каждом шаге заполнять `FlightCtx`
   из flown-состояния и ветра; пока фаза `Stowed`, интегрировать обычную
   6-DoF динамику; после раскрытия брать `state().altitude_m`, `pos_m`,
   `vel_xy_mps` как новые значения состояния. Передавать `t_apogee_s` из
   трекера фаз.
4. Truth: дописать 8 полей в v4-сайдкар (или в отдельный chute-сайдкар), чтобы
   v1/v2/v3 байты и читатели остались нетронутыми.
5. Manifest: добавить 9 имен в `bounds`/отдельный блок; при включённом куполе
   писать dispersion, при выключенном — не писать ничего (поля отсутствуют, а
   не нули).

Пока шаги 1–5 не сделаны, модуль тестируется через
`#[path = "../src/parachute.rs"] mod parachute;` в
`core/tests/parachute_tests.rs` — это позволяет гейтам компилировать код без
изменения `lib.rs`.

## 5. Гейты

`cargo test -p potok-sim --test parachute_tests`: 9 тестов — инертность при
`enabled = false`; LOUD-отклонение 8 невалидных конфигов; последовательность
ступеней и idempotence; всегда ли скорость снижения в диапазоне; дрейф по
истинному ветру (включая смену знака ветра); отсутствие команды в API;
побитовая повторяемость; типизированные триггеры и подтверждение снижения;
имена truth/manifest и отдельность dispersion от RMS.

## 6. Ограничения модели

- Инфляция — first-order релаксация, без реального надувания купола,
  колебаний и удара о землю. WORLD08 §12 перечисляет эти пункты как отдельную
  работу.
- Ветер берётся как `Wind::at` (истинный), без оценки датчиками — это
  привилегированная правда для метрики, не сигнал управления.
- Нет второго купола, запасного таймера и резервного пиро (пиро нет в принципе):
  защёлка одноразовая, `reset()` — только между эпизодами.

Собственный код — `GPL-3.0-or-later`.
