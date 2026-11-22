//! Deterministic emitters for Rust, C and Python from the validated spec.

use crate::model::{type_size, Field, Frame, Profile, Spec};
use std::fmt::Write as _;

fn up(s: &str) -> String {
    s.to_ascii_uppercase()
}

fn pascal(s: &str) -> String {
    s.split('_')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut c = p.chars();
            match c.next() {
                Some(f) => format!("{}{}", f.to_ascii_uppercase(), c.as_str()),
                None => String::new(),
            }
        })
        .collect()
}

fn frame_ty(frame: &Frame) -> String {
    format!("{}Frame", pascal(&frame.name))
}

fn le(spec: &Spec) -> bool {
    spec.protocol.endianness == "little"
}

fn all_frames(spec: &Spec) -> Vec<(&Profile, &Frame)> {
    spec.protocol
        .profiles
        .values()
        .flat_map(|profile| profile.frames.iter().map(move |frame| (profile, frame)))
        .collect()
}

fn is_request(frame: &Frame) -> bool {
    frame.name == "request" || frame.name.ends_with("_request")
}

fn version_expr(profile: &Profile, frame: &Frame) -> (String, String) {
    let v = if is_request(frame) {
        profile.versions.request
    } else {
        profile.versions.reply
    };
    let ty = frame
        .fields
        .iter()
        .find(|f| f.name == "version")
        .unwrap()
        .ty
        .clone();
    (ty.clone(), format!("{v}{ty}"))
}

fn version_value(profile: &Profile, frame: &Frame) -> u16 {
    if is_request(frame) {
        profile.versions.request
    } else {
        profile.versions.reply
    }
}

fn used_types(spec: &Spec) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (_profile, frame) in all_frames(spec) {
        for f in &frame.fields {
            if !out.contains(&f.ty) {
                out.push(f.ty.clone());
            }
        }
    }
    out
}

fn header(lang: &str) -> String {
    format!(
        "// Generated from uart.yaml by `cargo run -p potok-generator`.\n\
         // Language: {lang}. DO NOT EDIT — edit uart.yaml and regenerate.\n\n"
    )
}

pub fn gen_rust(spec: &Spec) -> String {
    let mut s = header("Rust");
    let p = &spec.protocol;

    let _ = writeln!(s, "pub const PROTOCOL_NAME: &str = {:?};", p.name);
    let _ = writeln!(s, "pub const SPEC_VERSION: u32 = {};", p.version);
    let _ = writeln!(s, "pub const ENDIANNESS: &str = {:?};", p.endianness);
    let _ = writeln!(
        s,
        "pub const CRC_POLYNOMIAL: u16 = {:#06x};",
        p.crc.polynomial
    );
    let _ = writeln!(s, "pub const CRC_INIT: u16 = {:#06x};", p.crc.init);
    let _ = writeln!(s, "pub const CRC_XOR_OUT: u16 = {:#06x};", p.crc.xor_out);
    let _ = writeln!(s, "pub const CRC_REFLECT_IN: bool = {};", p.crc.reflect_in);
    let _ = writeln!(
        s,
        "pub const CRC_REFLECT_OUT: bool = {};",
        p.crc.reflect_out
    );
    let _ = writeln!(s, "pub const FLAG_WIDTH: usize = {};", p.flags.width);
    let _ = writeln!(s);

    for e in &p.enums {
        let _ = writeln!(s, "#[repr(u8)]");
        let _ = writeln!(s, "#[derive(Clone, Copy, Debug, PartialEq, Eq)]");
        let _ = writeln!(s, "pub enum {} {{", e.name);
        for v in &e.variants {
            let _ = writeln!(s, "    {} = {},", v.name, v.value);
        }
        let _ = writeln!(s, "}}");
        let _ = writeln!(s);
        let _ = writeln!(s, "impl core::convert::TryFrom<u8> for {} {{", e.name);
        let _ = writeln!(s, "    type Error = u8;");
        let _ = writeln!(s, "    fn try_from(v: u8) -> Result<Self, Self::Error> {{");
        let _ = writeln!(s, "        match v {{");
        for v in &e.variants {
            let _ = writeln!(s, "            {} => Ok({}::{}),", v.value, e.name, v.name);
        }
        let _ = writeln!(s, "            other => Err(other),");
        let _ = writeln!(s, "        }}");
        let _ = writeln!(s, "    }}");
        let _ = writeln!(s, "}}");
        let _ = writeln!(s);
        let _ = writeln!(s, "impl From<{}> for u8 {{", e.name);
        let _ = writeln!(s, "    fn from(v: {}) -> u8 {{ v as u8 }}", e.name);
        let _ = writeln!(s, "}}");
        let _ = writeln!(s);
    }

    let _ = writeln!(s, "pub mod flags {{");
    for b in &p.flags.bits {
        let _ = writeln!(s, "    pub const {}: u8 = 1u8 << {};", up(&b.name), b.bit);
    }
    let _ = writeln!(s, "    #[inline(always)]");
    let _ = writeln!(
        s,
        "    pub fn has(flags: u8, mask: u8) -> bool {{ flags & mask != 0 }}"
    );
    let _ = writeln!(s, "    #[inline(always)]");
    let _ = writeln!(
        s,
        "    pub fn set(flags: &mut u8, mask: u8) {{ *flags |= mask; }}"
    );
    let _ = writeln!(s, "    #[inline(always)]");
    let _ = writeln!(
        s,
        "    pub fn clear(flags: &mut u8, mask: u8) {{ *flags &= !mask; }}"
    );
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);

    for (profile, frame) in all_frames(spec) {
        let f = up(&frame.name);
        let (vty, vexpr) = version_expr(profile, frame);
        let _ = writeln!(s, "pub const {f}_SIZE: usize = {};", frame.size);
        let _ = writeln!(s, "pub const {f}_MAGIC: u16 = {:#06x};", frame.magic);
        let _ = writeln!(s, "pub const {f}_VERSION: {vty} = {vexpr};");
        let mut off = 0usize;
        for field in &frame.fields {
            let sz = type_size(&field.ty).unwrap() * field.count();
            let _ = writeln!(s, "pub const {f}_OFF_{}: usize = {off};", up(&field.name));
            if field.crc {
                let _ = writeln!(s, "pub const {f}_CRC_OFF: usize = {off};");
                let _ = writeln!(s, "pub const {f}_CRC_LEN: usize = {sz};");
            }
            if let Some(scale) = field.scale {
                let _ = writeln!(
                    s,
                    "pub const {f}_{}_SCALE: f64 = {:?};",
                    up(&field.name),
                    scale
                );
            }
            if !field.scales.is_empty() {
                let scales = field
                    .scales
                    .iter()
                    .map(|v| format!("{v:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let names = field
                    .feature_names
                    .iter()
                    .map(|n| format!("{n:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let n = up(&field.name);
                let _ = writeln!(
                    s,
                    "pub const {f}_{n}_SCALES: [f64; {}] = [{scales}];",
                    field.scales.len()
                );
                let _ = writeln!(
                    s,
                    "pub const {f}_{n}_NAMES: [&str; {}] = [{names}];",
                    field.feature_names.len()
                );
            }
            off += sz;
        }
        let _ = writeln!(s);
    }

    for (_profile, frame) in all_frames(spec) {
        let ty = frame_ty(frame);
        let _ = writeln!(s, "#[derive(Clone, Copy, Debug, PartialEq)]");
        let _ = writeln!(s, "pub struct {ty} {{");
        for field in &frame.fields {
            if field.value.is_some() || field.crc {
                continue;
            }
            let ft = match &field.r#enum {
                Some(e) => e.clone(),
                None => field.ty.clone(),
            };
            if field.count() > 1 {
                let _ = writeln!(s, "    pub {}: [{ft}; {}],", field.name, field.count());
            } else {
                let _ = writeln!(s, "    pub {}: {ft},", field.name);
            }
        }
        let _ = writeln!(s, "}}");
        let _ = writeln!(s);
    }

    let _ = writeln!(s, "pub mod codec {{");
    for ty in used_types(spec) {
        let width = type_size(&ty).unwrap();
        let method = if le(spec) { "le" } else { "be" };
        let puts = (0..width)
            .map(|i| format!("buf[off + {i}] = b[{i}];"))
            .collect::<String>();
        let gets = (0..width)
            .map(|i| format!("buf[off + {i}]"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(s, "    #[inline(always)]");
        if width == 1 {
            let _ = writeln!(
                s,
                "    pub fn put_{ty}(buf: &mut [u8], off: usize, v: {ty}) {{ buf[off] = v; }}"
            );
            let _ = writeln!(s, "    #[inline(always)]");
            let _ = writeln!(
                s,
                "    pub fn get_{ty}(buf: &[u8], off: usize) -> {ty} {{ buf[off] }}"
            );
        } else {
            let _ = writeln!(
                s,
                "    pub fn put_{ty}(buf: &mut [u8], off: usize, v: {ty}) {{ let b = v.to_{method}_bytes(); {puts} }}"
            );
            let _ = writeln!(s, "    #[inline(always)]");
            let _ = writeln!(
                s,
                "    pub fn get_{ty}(buf: &[u8], off: usize) -> {ty} {{ {ty}::from_{method}_bytes([{gets}]) }}"
            );
        }
    }
    let _ = writeln!(s, "}}");
    s
}

fn c_type(ty: &str) -> &'static str {
    match ty {
        "u8" => "uint8_t",
        "i8" => "int8_t",
        "u16" => "uint16_t",
        "i16" => "int16_t",
        "u32" => "uint32_t",
        "i32" => "int32_t",
        "u64" => "uint64_t",
        "i64" => "int64_t",
        "f32" => "float",
        "f64" => "double",
        _ => "uint8_t",
    }
}

fn c_int_type(ty: &str) -> &'static str {
    match ty {
        "u8" => "uint8_t",
        "i8" => "int8_t",
        "u16" => "uint16_t",
        "i16" => "int16_t",
        "u32" => "uint32_t",
        "i32" => "int32_t",
        "u64" => "uint64_t",
        _ => "int64_t",
    }
}

fn c_scalar_prefix(ty: &str) -> &'static str {
    match ty {
        "u8" => "u8",
        "i8" => "i8",
        "u16" => "u16",
        "i16" => "i16",
        "u32" => "u32",
        "i32" => "i32",
        "f32" => "u32",
        "u64" => "u64",
        "i64" => "i64",
        "f64" => "u64",
        _ => "u8",
    }
}

pub fn gen_c(spec: &Spec) -> String {
    let mut s = String::from(
        "/* Generated from uart.yaml by `cargo run -p potok-generator`. DO NOT EDIT. */\n",
    );
    let p = &spec.protocol;
    let guard = format!("POTOK_{}_H", up(&p.name.replace('-', "_")));
    let _ = writeln!(s, "#ifndef {guard}");
    let _ = writeln!(s, "#define {guard}");
    let _ = writeln!(s);
    let _ = writeln!(s, "#include <stdint.h>");
    let _ = writeln!(s, "#include <stddef.h>");
    let _ = writeln!(s, "#include <string.h>");
    let _ = writeln!(s);
    let _ = writeln!(s, "#ifdef __cplusplus");
    let _ = writeln!(s, "extern \"C\" {{");
    let _ = writeln!(s, "#endif");
    let _ = writeln!(s);

    let _ = writeln!(s, "enum {{");
    let _ = writeln!(s, "    POTOK_OK = 0,");
    let _ = writeln!(s, "    POTOK_ERR_BUFFER = -1,");
    let _ = writeln!(s, "    POTOK_ERR_MAGIC = -2,");
    let _ = writeln!(s, "    POTOK_ERR_CRC = -3,");
    let _ = writeln!(s, "    POTOK_ERR_PHASE = -4,");
    let _ = writeln!(s, "    POTOK_ERR_VERSION = -5");
    let _ = writeln!(s, "}};");
    let _ = writeln!(s);

    let _ = writeln!(s, "enum PotokPhase {{");
    for (i, e) in p.enums.iter().enumerate() {
        for (j, v) in e.variants.iter().enumerate() {
            let comma = if i + 1 == p.enums.len() && j + 1 == e.variants.len() {
                ""
            } else {
                ","
            };
            let _ = writeln!(
                s,
                "    POTOK_{}_{} = {}{}",
                up(&e.name),
                up(&v.name),
                v.value,
                comma
            );
        }
    }
    let _ = writeln!(s, "}};");
    let _ = writeln!(s);

    let _ = writeln!(s, "enum PotokFlags {{");
    for (i, b) in p.flags.bits.iter().enumerate() {
        let comma = if i + 1 == p.flags.bits.len() { "" } else { "," };
        let _ = writeln!(
            s,
            "    POTOK_FLAG_{} = (1u << {}){comma}",
            up(&b.name),
            b.bit
        );
    }
    let _ = writeln!(s, "}};");
    let _ = writeln!(s);

    for (profile, frame) in all_frames(spec) {
        let f = up(&frame.name);
        let (vty, _) = version_expr(profile, frame);
        let cver = c_type(&vty);
        let _ = writeln!(s, "#define POTOK_{f}_SIZE {}u", frame.size);
        let _ = writeln!(s, "#define POTOK_{f}_MAGIC 0x{:04X}u", frame.magic);
        let _ = writeln!(
            s,
            "#define POTOK_{f}_VERSION (({cver}){})",
            version_value(profile, frame)
        );
        let mut off = 0usize;
        for field in &frame.fields {
            let sz = type_size(&field.ty).unwrap() * field.count();
            let _ = writeln!(s, "#define POTOK_{f}_OFF_{} {}u", up(&field.name), off);
            if field.crc {
                let _ = writeln!(s, "#define POTOK_{f}_CRC_OFF {}u", off);
                let _ = writeln!(s, "#define POTOK_{f}_CRC_LEN {}u", sz);
            }
            if let Some(scale) = field.scale {
                let _ = writeln!(
                    s,
                    "#define POTOK_{f}_{}_SCALE ({:?}f)",
                    up(&field.name),
                    scale
                );
            }
            if !field.scales.is_empty() {
                let n = up(&field.name);
                let vals = field
                    .scales
                    .iter()
                    .map(|v| format!("{v:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let names = field
                    .feature_names
                    .iter()
                    .map(|x| format!("{x:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = writeln!(
                    s,
                    "static const double POTOK_{f}_{n}_SCALES[{}] = {{{vals}}};",
                    field.scales.len()
                );
                let _ = writeln!(
                    s,
                    "static const char *const POTOK_{f}_{n}_NAMES[{}] = {{{names}}};",
                    field.feature_names.len()
                );
            }
            off += sz;
        }
        let _ = writeln!(s);
    }

    for (_profile, frame) in all_frames(spec) {
        let ty = format!("Potok{}", pascal(&frame.name));
        let _ = writeln!(s, "typedef struct {{");
        for field in &frame.fields {
            if field.value.is_some() || field.crc {
                continue;
            }
            let ct = match &field.r#enum {
                Some(_) => "enum PotokPhase",
                None => c_type(&field.ty),
            };
            if field.count() > 1 {
                let _ = writeln!(s, "    {ct} {}[{}];", field.name, field.count());
            } else {
                let _ = writeln!(s, "    {ct} {};", field.name);
            }
        }
        let _ = writeln!(s, "}} {ty};");
        let _ = writeln!(s);
    }

    // Scalar byte accessors.
    for ty in used_types(spec) {
        if ty.starts_with('f') {
            continue;
        }
        let pfx = c_scalar_prefix(&ty);
        let ct = c_int_type(&ty);
        let _ = writeln!(
            s,
            "static inline void potok_put_{pfx}(uint8_t *b, size_t o, {ct} v) {{"
        );
        if le(spec) {
            let _ = writeln!(s, "    for (size_t i = 0; i < sizeof({ct}); ++i) b[o + i] = (uint8_t)(v >> (8u * i));");
        } else {
            let _ = writeln!(s, "    for (size_t i = 0; i < sizeof({ct}); ++i) b[o + i] = (uint8_t)(v >> (8u * (sizeof({ct}) - 1u - i)));");
        }
        let _ = writeln!(s, "}}");
        let _ = writeln!(
            s,
            "static inline {ct} potok_get_{pfx}(const uint8_t *b, size_t o) {{"
        );
        let _ = writeln!(s, "    {ct} v = 0;");
        if le(spec) {
            let _ = writeln!(
                s,
                "    for (size_t i = 0; i < sizeof({ct}); ++i) v |= ({ct})b[o + i] << (8u * i);"
            );
        } else {
            let _ = writeln!(s, "    for (size_t i = 0; i < sizeof({ct}); ++i) v |= ({ct})b[o + i] << (8u * (sizeof({ct}) - 1u - i));");
        }
        let _ = writeln!(s, "    return v;");
        let _ = writeln!(s, "}}");
    }
    let _ = writeln!(
        s,
        "static inline void potok_put_f32(uint8_t *b, size_t o, float v) {{"
    );
    let _ = writeln!(
        s,
        "    uint32_t u; memcpy(&u, &v, sizeof(u)); potok_put_u32(b, o, u);"
    );
    let _ = writeln!(s, "}}");
    let _ = writeln!(
        s,
        "static inline float potok_get_f32(const uint8_t *b, size_t o) {{"
    );
    let _ = writeln!(
        s,
        "    uint32_t u = potok_get_u32(b, o); float v; memcpy(&v, &u, sizeof(v)); return v;"
    );
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);

    let _ = writeln!(
        s,
        "static inline uint16_t potok_crc16(const uint8_t *data, size_t len) {{"
    );
    let _ = writeln!(s, "    uint16_t crc = 0x{:04X}u;", p.crc.init);
    let _ = writeln!(s, "    for (size_t i = 0; i < len; ++i) {{");
    let _ = writeln!(s, "        crc ^= (uint16_t)((uint16_t)data[i] << 8);");
    let _ = writeln!(s, "        for (int j = 0; j < 8; ++j) {{");
    let _ = writeln!(s, "            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x{:04X}u) : (uint16_t)(crc << 1);", p.crc.polynomial);
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    return (uint16_t)(crc ^ 0x{:04X}u);", p.crc.xor_out);
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "static inline int potok_flag_has(uint8_t f, uint8_t mask) {{ return (f & mask) != 0; }}"
    );
    let _ = writeln!(s, "static inline void potok_flag_set(uint8_t *f, uint8_t mask) {{ *f = (uint8_t)(*f | mask); }}");
    let _ = writeln!(s, "static inline void potok_flag_clear(uint8_t *f, uint8_t mask) {{ *f = (uint8_t)(*f & (uint8_t)~mask); }}");
    let _ = writeln!(s);

    for (_profile, frame) in all_frames(spec) {
        let f = up(&frame.name);
        let ty = format!("Potok{}", pascal(&frame.name));
        let _ = writeln!(
            s,
            "static inline int potok_{}_encode(const {ty} *fr, uint8_t *buf, size_t len) {{",
            frame.name
        );
        let _ = writeln!(
            s,
            "    if (!fr || !buf || len < POTOK_{f}_SIZE) return POTOK_ERR_BUFFER;"
        );
        for field in &frame.fields {
            if field.crc {
                continue;
            }
            let off = field_offset(frame, field);
            if field.value.is_some() {
                let _ = writeln!(
                    s,
                    "    potok_put_u16(buf, POTOK_{f}_OFF_{}, (uint16_t)POTOK_{f}_MAGIC);",
                    up(&field.name)
                );
                continue;
            }
            let pfx = c_scalar_prefix(&field.ty);
            if field.count() > 1 {
                let _ = writeln!(s, "    for (size_t i = 0; i < {}u; ++i) {{", field.count());
                if field.ty == "f32" {
                    let _ = writeln!(
                        s,
                        "        potok_put_f32(buf, {off} + {}u * i, fr->{}[i]);",
                        type_size(&field.ty).unwrap(),
                        field.name
                    );
                } else {
                    let _ = writeln!(
                        s,
                        "        potok_put_{pfx}(buf, {off} + {}u * i, ({})fr->{}[i]);",
                        type_size(&field.ty).unwrap(),
                        c_int_type(&field.ty),
                        field.name
                    );
                }
                let _ = writeln!(s, "    }}");
            } else if field.ty == "f32" {
                let _ = writeln!(s, "    potok_put_f32(buf, {off}, fr->{});", field.name);
            } else if field.r#enum.is_some() {
                let _ = writeln!(
                    s,
                    "    potok_put_u8(buf, {off}, (uint8_t)fr->{});",
                    field.name
                );
            } else {
                let _ = writeln!(
                    s,
                    "    potok_put_{pfx}(buf, {off}, ({})fr->{});",
                    c_int_type(&field.ty),
                    field.name
                );
            }
        }
        let _ = writeln!(
            s,
            "    potok_put_u16(buf, POTOK_{f}_CRC_OFF, potok_crc16(buf, POTOK_{f}_CRC_OFF));"
        );
        let _ = writeln!(s, "    return POTOK_OK;");
        let _ = writeln!(s, "}}");

        let _ = writeln!(
            s,
            "static inline int potok_{}_decode(const uint8_t *buf, size_t len, {ty} *fr) {{",
            frame.name
        );
        let _ = writeln!(
            s,
            "    if (!buf || !fr || len < POTOK_{f}_SIZE) return POTOK_ERR_BUFFER;"
        );
        let _ = writeln!(
            s,
            "    if (potok_get_u16(buf, 0) != (uint16_t)POTOK_{f}_MAGIC) return POTOK_ERR_MAGIC;"
        );
        let _ = writeln!(s, "    if (potok_get_u16(buf, POTOK_{f}_CRC_OFF) != potok_crc16(buf, POTOK_{f}_CRC_OFF)) return POTOK_ERR_CRC;");
        for field in &frame.fields {
            if field.crc || field.value.is_some() {
                continue;
            }
            let off = field_offset(frame, field);
            let pfx = c_scalar_prefix(&field.ty);
            if field.count() > 1 {
                let _ = writeln!(s, "    for (size_t i = 0; i < {}u; ++i) {{", field.count());
                if field.ty == "f32" {
                    let _ = writeln!(
                        s,
                        "        fr->{}[i] = potok_get_f32(buf, {off} + 4u * i);",
                        field.name
                    );
                } else {
                    let _ = writeln!(
                        s,
                        "        fr->{}[i] = ({})potok_get_{pfx}(buf, {off} + {}u * i);",
                        field.name,
                        c_type(&field.ty),
                        type_size(&field.ty).unwrap()
                    );
                }
                let _ = writeln!(s, "    }}");
            } else if field.r#enum.is_some() {
                let _ = writeln!(s, "    {{");
                let _ = writeln!(s, "        uint8_t raw = potok_get_u8(buf, {off});");
                let max_phase = p
                    .enums
                    .iter()
                    .flat_map(|e| e.variants.iter())
                    .map(|v| v.value)
                    .max()
                    .unwrap_or(0);
                let _ = writeln!(s, "        if (raw > {max_phase}u) return POTOK_ERR_PHASE;");
                let _ = writeln!(s, "        fr->{} = (enum PotokPhase)raw;", field.name);
                let _ = writeln!(s, "    }}");
            } else if field.ty == "f32" {
                let _ = writeln!(s, "    fr->{} = potok_get_f32(buf, {off});", field.name);
            } else {
                let _ = writeln!(
                    s,
                    "    fr->{} = ({})potok_get_{pfx}(buf, {off});",
                    field.name,
                    c_type(&field.ty)
                );
            }
        }
        let _ = writeln!(
            s,
            "    if (fr->version != POTOK_{f}_VERSION) return POTOK_ERR_VERSION;"
        );
        let _ = writeln!(s, "    return POTOK_OK;");
        let _ = writeln!(s, "}}");
        let _ = writeln!(s);
    }

    for (_profile, frame) in all_frames(spec) {
        for field in &frame.fields {
            if let Some(_scale) = field.scale {
                let f = up(&frame.name);
                let n = &field.name;
                let c = up(&field.name);
                let ct = c_type(&field.ty);
                let (min, max) = int_bounds(&field.ty);
                let _ = writeln!(s, "static inline double potok_{}_{n}_scaled({ct} raw) {{ return (double)raw * POTOK_{f}_{c}_SCALE; }}", frame.name);
                let _ = writeln!(
                    s,
                    "static inline {ct} potok_{}_{n}_from_scaled(double value) {{",
                    frame.name
                );
                let _ = writeln!(s, "    double v = value / POTOK_{f}_{c}_SCALE;");
                let _ = writeln!(s, "    double r = (v >= 0.0) ? v + 0.5 : v - 0.5;");
                let _ = writeln!(s, "    if (r > {max}.0) return ({ct}){max};");
                let _ = writeln!(s, "    if (r < {min}.0) return ({ct}){min};");
                let _ = writeln!(s, "    return ({ct})r;");
                let _ = writeln!(s, "}}");
            }
        }
    }
    let _ = writeln!(s);

    let _ = writeln!(s, "#ifdef __cplusplus");
    let _ = writeln!(s, "}}");
    let _ = writeln!(s, "#endif");
    let _ = writeln!(s, "#endif /* {guard} */");
    s
}

fn int_bounds(ty: &str) -> (i64, i64) {
    match ty {
        "i8" => (i8::MIN as i64, i8::MAX as i64),
        "u8" => (0, u8::MAX as i64),
        "i16" => (i16::MIN as i64, i16::MAX as i64),
        "u16" => (0, u16::MAX as i64),
        "i32" => (i32::MIN as i64, i32::MAX as i64),
        "u32" => (0, u32::MAX as i64),
        _ => (i64::MIN, i64::MAX),
    }
}

fn field_offset(frame: &Frame, target: &Field) -> usize {
    let mut off = 0usize;
    for f in &frame.fields {
        if std::ptr::eq(f, target) {
            return off;
        }
        off += type_size(&f.ty).unwrap() * f.count();
    }
    unreachable!("field not in its own frame")
}

fn py_struct_prefix(ty: &str) -> &'static str {
    match ty {
        "u8" => "B",
        "i8" => "b",
        "u16" => "H",
        "i16" => "h",
        "u32" => "I",
        "i32" => "i",
        "u64" => "Q",
        "i64" => "q",
        "f32" => "f",
        "f64" => "d",
        _ => "B",
    }
}

pub fn gen_python(spec: &Spec) -> String {
    let mut s = String::from(
        "# Generated from uart.yaml by `cargo run -p potok-generator`. DO NOT EDIT.\n\
         \"\"\"Potok-M UART wire format (generated). All frames are fixed-size.\"\"\"\n\n",
    );
    let p = &spec.protocol;
    let endian = if le(spec) { "<" } else { ">" };
    let _ = writeln!(s, "from __future__ import annotations");
    let _ = writeln!(s);
    let _ = writeln!(s, "import struct");
    let _ = writeln!(s, "from dataclasses import dataclass");
    let _ = writeln!(s, "from enum import IntEnum");
    let _ = writeln!(s);
    let _ = writeln!(s, "PROTOCOL_NAME = {:?}", p.name);
    let _ = writeln!(s, "SPEC_VERSION = {}", p.version);
    let _ = writeln!(s, "ENDIANNESS = {:?}", p.endianness);
    let _ = writeln!(s, "CRC_POLYNOMIAL = 0x{:04X}", p.crc.polynomial);
    let _ = writeln!(s, "CRC_INIT = 0x{:04X}", p.crc.init);
    let _ = writeln!(s, "CRC_XOR_OUT = 0x{:04X}", p.crc.xor_out);
    let _ = writeln!(s);
    let _ = writeln!(s, "class ProtocolError(ValueError): pass");
    let _ = writeln!(s, "class Truncated(ProtocolError): pass");
    let _ = writeln!(s, "class BadMagic(ProtocolError): pass");
    let _ = writeln!(s, "class BadCrc(ProtocolError): pass");
    let _ = writeln!(s, "class BadPhase(ProtocolError): pass");
    let _ = writeln!(s, "class VersionMismatch(ProtocolError): pass");
    let _ = writeln!(s);

    for e in &p.enums {
        let _ = writeln!(s, "class {}(IntEnum):", e.name);
        for v in &e.variants {
            let _ = writeln!(s, "    {} = {}", v.name, v.value);
        }
        let _ = writeln!(s);
    }

    for b in &p.flags.bits {
        let _ = writeln!(s, "FLAG_{} = 1 << {}", up(&b.name), b.bit);
    }
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "def flag_has(flags: int, mask: int) -> bool: return (flags & mask) != 0"
    );
    let _ = writeln!(s);

    let _ = writeln!(s, "def crc16(data: bytes) -> int:");
    let _ = writeln!(s, "    crc = 0x{:04X}", p.crc.init);
    let _ = writeln!(s, "    for byte in data:");
    let _ = writeln!(s, "        crc ^= byte << 8");
    let _ = writeln!(s, "        for _ in range(8):");
    let _ = writeln!(s, "            crc = ((crc << 1) ^ 0x{:04X}) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF", p.crc.polynomial);
    let _ = writeln!(s, "    return crc ^ 0x{:04X}", p.crc.xor_out);
    let _ = writeln!(s);

    for (profile, frame) in all_frames(spec) {
        let f = up(&frame.name);
        let cls = pascal(&frame.name);
        let ver = version_value(profile, frame);
        let _ = writeln!(s, "{f}_SIZE = {}", frame.size);
        let _ = writeln!(s, "{f}_MAGIC = 0x{:04X}", frame.magic);
        let _ = writeln!(s, "{f}_VERSION = {ver}");
        let mut off = 0usize;
        for field in &frame.fields {
            let sz = type_size(&field.ty).unwrap() * field.count();
            let _ = writeln!(s, "{f}_OFF_{} = {off}", up(&field.name));
            if field.crc {
                let _ = writeln!(s, "{f}_CRC_OFF = {off}");
                let _ = writeln!(s, "{f}_CRC_LEN = {sz}");
            }
            if let Some(scale) = field.scale {
                let _ = writeln!(s, "{f}_{}_SCALE = {:?}", up(&field.name), scale);
            }
            if !field.scales.is_empty() {
                let n = up(&field.name);
                let vals = field
                    .scales
                    .iter()
                    .map(|v| format!("{v:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let names = field
                    .feature_names
                    .iter()
                    .map(|x| format!("{x:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = writeln!(s, "{f}_{n}_SCALES = ({vals},)");
                let _ = writeln!(s, "{f}_{n}_NAMES = ({names},)");
            }
            off += sz;
        }
        let _ = writeln!(s);

        let _ = writeln!(s, "@dataclass");
        let _ = writeln!(s, "class {cls}:");
        let data_fields: Vec<&Field> = frame
            .fields
            .iter()
            .filter(|x| x.value.is_none() && !x.crc)
            .collect();
        if data_fields.is_empty() {
            let _ = writeln!(s, "    pass");
        }
        for field in &data_fields {
            let (base, comment) = match &field.r#enum {
                Some(e) => (e.clone(), String::new()),
                None => match field.ty.as_str() {
                    "f32" | "f64" => ("float".to_string(), String::new()),
                    t => ("int".to_string(), format!("  # {t}")),
                },
            };
            let ann = if field.count() > 1 {
                format!("list[{base}]")
            } else {
                base
            };
            let _ = writeln!(s, "    {}: {}{}", field.name, ann, comment);
        }
        let _ = writeln!(s);

        let _ = writeln!(s, "def encode_{}(frame: {cls}) -> bytes:", frame.name);
        let _ = writeln!(s, "    buf = bytearray({f}_SIZE)");
        let _ = writeln!(
            s,
            "    struct.pack_into({endian:?} + 'H', buf, {f}_OFF_MAGIC, {f}_MAGIC)"
        );
        for field in &data_fields {
            let off = field_offset(frame, field);
            let fmt = py_struct_prefix(&field.ty);
            if field.count() > 1 {
                let _ = writeln!(s, "    for i, item in enumerate(frame.{}):", field.name);
                let _ = writeln!(
                    s,
                    "        struct.pack_into({endian:?} + {fmt:?}, buf, {off} + {sz} * i, item)",
                    sz = type_size(&field.ty).unwrap()
                );
            } else if field.r#enum.is_some() {
                let _ = writeln!(
                    s,
                    "    struct.pack_into({endian:?} + 'B', buf, {off}, int(frame.{}))",
                    field.name
                );
            } else {
                let _ = writeln!(
                    s,
                    "    struct.pack_into({endian:?} + {fmt:?}, buf, {off}, frame.{})",
                    field.name
                );
            }
        }
        let _ = writeln!(s, "    crc = crc16(bytes(buf[:{f}_CRC_OFF]))");
        let _ = writeln!(
            s,
            "    struct.pack_into({endian:?} + 'H', buf, {f}_CRC_OFF, crc)"
        );
        let _ = writeln!(s, "    return bytes(buf)");
        let _ = writeln!(s);

        let _ = writeln!(s, "def decode_{}(data: bytes) -> {cls}:", frame.name);
        let _ = writeln!(s, "    if len(data) < {f}_SIZE: raise Truncated(f'need {f}_SIZE bytes, got {{len(data)}}')");
        let _ = writeln!(s, "    if struct.unpack_from({endian:?} + 'H', data, {f}_OFF_MAGIC)[0] != {f}_MAGIC: raise BadMagic('magic mismatch')");
        let _ = writeln!(s, "    want = crc16(bytes(data[:{f}_CRC_OFF]))");
        let _ = writeln!(
            s,
            "    got = struct.unpack_from({endian:?} + 'H', data, {f}_CRC_OFF)[0]"
        );
        let _ = writeln!(
            s,
            "    if got != want: raise BadCrc(f'want 0x{{want:04X}}, got 0x{{got:04X}}')"
        );
        for field in &data_fields {
            let off = field_offset(frame, field);
            let fmt = py_struct_prefix(&field.ty);
            if let Some(ref en) = field.r#enum {
                let _ = writeln!(
                    s,
                    "    {}_raw = struct.unpack_from({endian:?} + 'B', data, {off})[0]",
                    field.name
                );
                let _ = writeln!(s, "    try: {}_v = {}({}_raw)", field.name, en, field.name);
                let _ = writeln!(
                    s,
                    "    except ValueError as exc: raise BadPhase(str(exc)) from exc"
                );
            } else if field.count() > 1 {
                let _ = writeln!(
                    s,
                    "    {}_v = list(struct.unpack_from({endian:?} + {fmt:?} * {}, data, {off}))",
                    field.name,
                    field.count()
                );
            } else {
                let _ = writeln!(
                    s,
                    "    {}_v = struct.unpack_from({endian:?} + {fmt:?}, data, {off})[0]",
                    field.name
                );
            }
        }
        let ver_name = data_fields
            .iter()
            .find(|x| x.name == "version")
            .map(|x| format!("{}_v", x.name))
            .unwrap_or_default();
        let _ = writeln!(s, "    if {ver_name} != {f}_VERSION: raise VersionMismatch(f'want {ver}, got {{{ver_name}}}')");
        let _ = writeln!(s, "    return {cls}(");
        for field in &data_fields {
            let _ = writeln!(s, "        {}={}_v,", field.name, field.name);
        }
        let _ = writeln!(s, "    )");
        let _ = writeln!(s);

        for field in &frame.fields {
            if let Some(_scale) = field.scale {
                let n = &field.name;
                let c = up(&field.name);
                let (min, max) = int_bounds(&field.ty);
                let _ = writeln!(
                    s,
                    "def {}_{n}_scaled(raw: int) -> float: return raw * {f}_{c}_SCALE",
                    frame.name
                );
                let _ = writeln!(
                    s,
                    "def {}_{n}_from_scaled(value: float) -> int:",
                    frame.name
                );
                let _ = writeln!(s, "    v = value / {f}_{c}_SCALE");
                let _ = writeln!(s, "    r = int(v + 0.5 if v >= 0 else v - 0.5)");
                let _ = writeln!(s, "    return min({max}, max({min}, r))");
            }
        }
        let _ = writeln!(s);
    }

    s
}
