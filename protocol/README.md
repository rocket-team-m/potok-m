![Поток-М: Протокол](docs/banner.svg)

[![CI](https://github.com/rocket-team-m/potok-m-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/rocket-team-m/potok-m-protocol/actions/workflows/ci.yml)

<div style="background:#0b0b0d;border-left:2px solid #34d399;padding:14px 18px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#d4d4d4;line-height:1.7">

<span style="color:#34d399">UART / CONTRACT</span> · STM32H743 ⇄ Raspberry Pi · little-endian · CRC-16/CCITT-FALSE · no heap · <span style="color:#34d399">no_std</span>

</div>

# Поток-М: UART-протокол

Байтовый контракт между контроллером стабилизации и бортовым компьютером. Запрос с датчиков уходит **STM32 → Pi**, ответ с поправками и состоянием ИИ возвращается **Pi → STM32**. Описание кадров, контрольная сумма, нумерация, фазы и флаги живут в `uart.yaml`; три биндинга в `generated/` порождаются из него и ни в коем случае не редактируются руками.

Прошивки STM32 в этом репозитории нет — только контракт. Прошивка живёт в [potok-m-firmware](https://github.com/rocket-team-m/potok-m-firmware), план — [FIRMWARE_RUST_PLAN.md](https://github.com/rocket-team-m/potok-m/blob/main/FIRMWARE_RUST_PLAN.md).

---

## `01` / Направление — не переворачивать

Кадр **request** формируется на STM32 и передаётся на Raspberry Pi. Кадр **reply** формируется на Pi и передаётся на STM32. Это не взаимозаменяемые структуры: у них разные magic, разный размер и разный смысл полей. Перестановка направлений — не косметическая ошибка, а путь к выполнению неверной команды.

```text
  STM32H743                         Raspberry Pi (edge)
  датчики, фаза, seq, flags  ────►   decode → ИИ (potok-m-ai) → encode
  применяет команду, держит  ◄────   g_pitch, g_yaw, confidence, ai_healthy
  вето и фазу                       запрос/ответ
```

Единственный источник правды — [`uart.yaml`](uart.yaml). Генератор [`generator/`](generator/) читает только его, проходит строгую serde-схему (`deny_unknown_fields`: лишний или опечатанный ключ — ошибка) и валидацию: размер кадра сверяется с суммой полей, CRC-field стоит в конце, версии и magic совпадают с объявленными, ссылки на enum и флаги разрешимы. Любое расхождение — громкая ошибка, а не молчаливая подстановка.

---

## `02` / Два профиля: compat и lowlat

Канонический профиль по умолчанию — **compat**. **lowlat** добавлен вторым профилем и не отменяет первый: у каждого свои версии и кадры, но общие CRC, `Phase` и биты флагов.

| | `compat` | `lowlat` |
|---|---|---|
| Запрос STM32 → Pi | 56 Б, magic `0x504D` | 42 Б, magic `0x4C54` |
| Ответ Pi → STM32 | 14 Б, magic `0x5250` | 16 Б, magic `0x4C52` |
| UART | 921600 бод, 8N1 | 2 Мбит/с, 8N1 |
| Версия request / reply | 1 / 1 | 1 / 1 |
| Время в проводе | 0,49 мс + 0,12 мс — расчёт | 0,168 мс + 0,064 мс — расчёт |

Значения magic — ASCII-пары `PM`/`PR` и `LT`/`LR` в little-endian: `4d50…c67f` и `5052…c549` в compat, `544c…0a2b` и `524c…fd1b` в lowlat.

**compat request, 56 Б:** `magic u16`, `seq u16`, `time_us u32`, `phase u8`, `version u8`, `flags u8`, `reserved u8`, `gyro f32[3]`, `accel f32[3]`, `mag f32[3]`, `height f32`, `pitot_dp i16`, `crc16 u16`.

**compat reply, 14 Б:** `magic u16`, `seq u16`, `g_pitch i16` (×0,001), `g_yaw i16` (×0,001), `confidence u8`, `ai_healthy u8`, `version u16`, `crc16 u16`.

**lowlat request, 42 Б:** те же 12 байт заголовка, затем 14 × `int16` признаков и CRC. Масштабы заданы в YAML и сгенерированы как `LOWLAT_REQUEST_FEATURES_SCALES` / `_NAMES`: гиро ×0,001, аксель ×0,001, магнитометр ×0,0001, высота ×0,01, давление ×1, Pitot ×1, температура ×0,1, батарея ×1. int16-диапазон покрывает диапазоны датчиков, но **проверка насыщения обязательна**; в библиотеке за это отвечают `quantize_i16` / `dequantize_i16` и `clamp_i16` (ограничение `-32768..32767`).

**lowlat reply, 16 Б:** `magic u16`, `seq u16`, `g_pitch i16` (×0,001), `g_yaw i16` (×0,001), `confidence u8`, `ai_healthy u8`, `version u8`, `compute_us u8` (единица — 100 мкс), `reserved u8[2]`, `crc16 u16`.

Две неоднозначности спецификации lowlat разрешены явно и зафиксированы валидацией: перечисление полей ответа давало 14 Б при заявленных 16 — добавлены два `reserved` по смещению 12; magic ответа в спецификации отсутствовал — выбран `0x4C52` и объявлен в YAML.

---

## `03` / CRC, seq, версия, фаза, флаги

**CRC-16/CCITT-FALSE.** Полином `0x1021`, init `0xFFFF`, нет отражения на входе и выходе, `xor_out = 0x0000`. Покрывает все байты кадра, кроме последних двух — поля `crc16`. Контрольное значение для `123456789` — `29B1` (прогон C-харнесса: `crc123456789=29B1`). CRC ловит случайные искажения; это **не** криптозащита от злонамеренных атак.

C-биндинг считает CRC по таблице `POTOK_CRC16_TABLE[256]` (512 Б `.rodata`), которую генератор строит из `crc.polynomial` в YAML. Байтовый lookup вместо побитового цикла: ~330 вместо ~2600 инструкций на 40-байтовый кадр на Cortex-M4 при `-Os` (замер MCUF03). Rust- и Python-биндинги не менялись.

**Seq.** `SeqTracker` считает номер по кругу, окно по умолчанию 64 (должно быть `1..=0x7fff`, чтобы разворот был однозначен). Дубли (`Duplicate`) и запоздавшие кадры (`OutOfWindow`, `delta > window`) отвергаются. Повторной передачи нет: пропуск по дедлайну считается отказом кадра.

**Версия и magic.** Проверяются при каждом decode: несовпадение даёт `VersionMismatch` или `BadMagic` с ожидаемым и полученным значением. У compat reply версия `u16`, у lowlat reply — `u8`.

**Фаза.** Перечисление `Phase` фиксировано и проверяется генератором: `BURN=0`, `COAST=1`, `APOGEE=2`, `DESCENT=3`, `LANDED=4`. Тест `phase_uses_documented_names_and_codes` отвергает старую схему `Idle=0…` и любую другую расстановку. Числовые коды не менялись, поэтому переименование было wire-нейтральным.

**Флаги.** Один байт: бит 0 зарезервирован (не назначен), `IMU_OK 1`, `BARO_OK 2`, `PITOT_OK 3`, `AI_READY 4`, `CALIBRATED 5`, `MOTOR_ARMED 6`, `FAULT 7`. Хелперы `flag_has` / `flag_set` / `flag_clear` работают без выделений.

---

## `04` / Генерация, golden-байты и CI

`cargo run -p potok-generator` порождает `generated/protocol.rs`, `generated/protocol.h`, `generated/protocol.py`. Вывод детерминирован: перегенерация даёт побайтово одинаковые файлы (`sha256sum` → diff). Потребителям генератор не нужен — достаточно `--check`.

Флаг `--check` сверяет содержимое трёх файлов с тем, что генератор произвёл из YAML, и при расхождении падает с сообщением `<path> is stale — rerun cargo run -p potok-generator`. Он же стоит в [CI](.github/workflows/ci.yml):

```text
cargo fmt --all -- --check
cargo clippy --all-targets
cargo test --all-targets
cargo run -p potok-generator -- --check
cargo check -p potok-protocol --no-default-features
```

Кросс-языковой golden-шаг теперь есть и в CI: [`.github/workflows/cross-language.yml`](.github/workflows/cross-language.yml) гоняет шесть проверок `stm32/tests/check_golden.sh` — `self-test`, `license`, `yaml`, `c`, `python`, `rust`. Коммитный C-харнесс — [`stm32/tests/golden_harness.c`](stm32/tests/golden_harness.c), он собирается с `cc -std=c11 -Wall -Wextra -Werror`; Python round-trip проходит, все три языка воспроизводят одни и те же golden-кадры. Ручные замеры и протокол — в [`BUILD_LOG.md`](BUILD_LOG.md). Тесты `golden_wire_bytes_match_c_and_python` и `committed_golden_fixtures_reencode_byte_identical` не дают трём языкам молча разойтись. На момент последнего прогона — 37 тестов: 9 генератор, 3 библиотека, 10 lowlat, 15 compat.

Lowlat-golden, зафиксированные в `lowlat_tests.rs` и воспроизводимые C и Python:

```text
request 544c34127bf15365020112000100feff03000400fbff06000700f8ff09000a000b00f4ff0d000e000a2b
reply   524c34127d0006ffc801012a0000fd1b
```

---

## `05` / Горящий цикл: zero-copy и 128,5 нс

`decode_*` оставляет owns-путь, но базовый путь — через `RequestView<'a>` / `ReplyView<'a>` (и lowlat-аналоги): это провалидированные заимствованные представления поверх входного среза. Аксессоры читают исходные байты напрямую; ни одно поле не копируется до явного `to_owned()`. В кодировании и декодировании **нет выделений памяти**; сгенерированный код пишет и читает байты напрямую, без `try_into().unwrap()` и `copy_from_slice`.

CRC считается таблицей, построенной в compile-time из полинома YAML. `CRC_TABLE` плюс позиционные `CRC_TABLE2` дают slicing-by-4: четыре байта за два зависимых чтения таблиц, хвост из 1–3 байт — побайтово. `crc_table_matches_bitwise_reference` сравнивает табличный результат с побитовым на всех длинах `0..256`.

Замер hotpath: `cargo run --release -p potok-protocol --example hotpath`, 1 000 000 итераций, 5 повторов, минимум, `black_box` на входах и выходах ([BUILD_LOG.md](BUILD_LOG.md), 2026-11-22):

| случай | ns/кадр | MB/s |
|---|---:|---:|
| lowlat request encode+decode+CRC | **128,5** | 653,7 |
| lowlat reply encode+decode+CRC | 28,7 | 1116,9 |
| compat request encode+decode+CRC | 160,0 | 700,2 |
| CRC по 54 Б, slicing-by-4 | 87,2 | 618,9 |
| CRC по 54 Б, побитовый референс | 150,0 | 360,1 |

Slicing-by-4 ускорил полный compat request (encode+decode+CRC) с 328,1 до 160,0 нс (2,05×), а CRC по 54 Б — с 180,9 до 87,2 нс (2,07×); побитовый референс в том же процессе показал 1,72× в пользу таблицы — эта пара нечувствительна к межпроцессному шуму (бокс был под нагрузкой ~5 от других процессов). Явные SIMD-интринзики **не добавлялись**: 16-битному CRC на 54 Б нужны gather и полиномиальная комбинация, и ни один такой кандидат не был измерен быстрее табличного пути. Пропущено сознательно, не предполагалось.

---

## `06` / Безопасность: семантика отказа

Контракт, который обязаны соблюдать потребители кадров:

- **Плохой кадр не вызывает модели.** При ошибке magic, CRC, версии, фазы или усечении ответ содержит признак неисправности, а инференс не запускается.
- **Три подряд отвергнутых кадра отключают ассистента** до конца полёта; снимает защёлку только явный сброс прошивки.
- **Фазу выставляет STM32.** Pi не может переопределить `BURN` и не управляет сервами напрямую; STM32 сохраняет право вето и не исполняет команды Pi бесконтрольно. Команда по умолчанию — ноль.
- **Ретрансляции нет.** Дедлайн — это отказ кадра, а не повод повторить отправку.
- CRC не заменяет защиту от умышленных искажений; для случайных помех на линии его достаточно, но fault injection на линии и проверка `BURN` с подключёнными сервами ещё не закрыты.

Эти правила — часть протокольного контракта; исполняет их сторонняя прошивка и edge-потребитель, в этом репозитории реализованы проверки кадра и хелперы (`SeqTracker`, валидация фаз и флагов).

---

## `07` / Встраивание: no_std, C, Python

`potok-protocol` не имеет зависимостей и по умолчанию `no_std` (`#![cfg_attr(not(feature = "std"), no_std)]`); фича `std` добавляет лишь `impl std::error::Error`. Это проверяется в CI отдельной строкой `cargo check -p potok-protocol --no-default-features`.

Для C достаточно header-only [`generated/protocol.h`](generated/protocol.h): C11, явная укладка байтов, без `memcpy` структуры. Для Python — [`generated/protocol.py`](generated/protocol.py). Все три языка воспроизводят одни и те же golden-байты.

Бюджет МКУ из [`PROTOCOL_BUDGET.md`](PROTOCOL_BUDGET.md) — именно **расчёт**, не измерение: 70 Б RAM под два статических буфера и 1536 Б Flash под таблицы CRC. Замеров на реальной STM32 нет; время в проводе посчитано из размера кадра и скорости.

---

## `08` / Команды

```bash
cargo check -p potok-protocol --no-default-features   # no_std, без аллокатора
cargo test                                             # 37 тестов
cargo run -p potok-generator -- --check                # сгенерированное совпадает с YAML
cargo run -p potok-generator                           # перегенерировать .rs/.h/.py
cargo run --release -p potok-protocol --example hotpath 1000000
```

---

## `09` / Границы и честные ограничения

**Нет прошивки.** В репозитории лежит контракт; STM32-прошивка — в [potok-m-firmware](https://github.com/rocket-team-m/potok-m-firmware).

**Нет замеров на реальной STM32.** Время в проводе — расчёт из размера кадра и скорости. Есть замеры только на edge: полный цикл lowlat p50 40,6 мкс, p99 356,6 мкс; с проводами 588,6 мкс; худший цикл с учителем 1,18 мс из отведённых 10 мс.

**Не закрыто:** проверка на помехи, fault injection на линии, проверка `BURN` с подключёнными сервами, кросс-языковая интероперабельность с прошивкой, формальный safety case. Хотпат-замеры проводились на общей машине при load ~5. `PROTOCOL_BUDGET.md` и `SPEC_PROTOCOL_LOWLAT.md` помечены как черновики.

**Соседние репозитории:** [potok-m-ai](https://github.com/rocket-team-m/potok-m-ai) — потребитель кадров на Pi; [potok-m-sim](https://github.com/rocket-team-m/potok-m-sim) — генератор кадров в симуляции; [potok-m-firmware](https://github.com/rocket-team-m/potok-m-firmware) — прошивка STM32.

---

## `10` / Структура

| Путь | Что внутри |
|---|---|
| [`uart.yaml`](uart.yaml) | Профили, кадры, CRC, `Phase`, флаги, версии, magic — источник правды |
| [`generator/`](generator/) | Строгая serde-схема, валидация, три эмиттера `.rs` / `.h` / `.py` |
| [`generated/`](generated/) | Зафиксированные в git детерминированные артефакты |
| [`potok-protocol/`](potok-protocol/) | CRC slicing-by-4, codec, zero-copy views, `SeqTracker`, хелперы флагов и углов |
| [`BUILD_LOG.md`](BUILD_LOG.md) | Протокол проверок, golden-байты, замеры hotpath |
| [`PROTOCOL_BUDGET.md`](PROTOCOL_BUDGET.md), [`SPEC_PROTOCOL_LOWLAT.md`](SPEC_PROTOCOL_LOWLAT.md) | Бюджет/надёжность и спецификация lowlat (черновики) |

Лицензия: GPL-3.0-or-later (или коммерческая).

Репозитории: [potok-m](https://github.com/rocket-team-m/potok-m) · [potok-m-ai](https://github.com/rocket-team-m/potok-m-ai) · [potok-m-edge](https://github.com/rocket-team-m/potok-m-edge) · [potok-m-sim](https://github.com/rocket-team-m/potok-m-sim) · [potok-m-protocol](https://github.com/rocket-team-m/potok-m-protocol) · [potok-m-firmware](https://github.com/rocket-team-m/potok-m-firmware) · [potok-m-3d](https://github.com/rocket-team-m/potok-m-3d).
