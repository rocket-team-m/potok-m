//! Strict YAML schema and semantic validation for `uart.yaml`.
//!
//! Every struct uses `deny_unknown_fields`: a misspelled key, an extra key or a
//! key in the wrong section is an error, never a silently ignored setting.

use serde::de::{self, Deserializer};
use serde::Deserialize;

fn parse_int(text: &str) -> Result<u64, String> {
    let s = text.trim();
    let (radix, digits) = if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))
    {
        (16, rest)
    } else if let Some(rest) = s.strip_prefix("0b").or_else(|| s.strip_prefix("0B")) {
        (2, rest)
    } else if let Some(rest) = s.strip_prefix("0o").or_else(|| s.strip_prefix("0O")) {
        (8, rest)
    } else {
        (10, s)
    };
    u64::from_str_radix(digits, radix).map_err(|e| format!("bad integer {s:?}: {e}"))
}

/// Accepts either a YAML integer or a quoted string like `0x1021`.
fn de_num<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: TryFrom<u64>,
    T::Error: std::fmt::Debug,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Num {
        Int(u64),
        Str(String),
    }
    let raw = match Num::deserialize(d)? {
        Num::Int(v) => v,
        Num::Str(s) => parse_int(&s).map_err(de::Error::custom)?,
    };
    T::try_from(raw)
        .map_err(|_| de::Error::custom(format!("value {raw} out of range for target type")))
}

fn de_opt_u64<'de, D>(d: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Some(de_num::<D, u64>(d)?))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub protocol: Protocol,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Protocol {
    pub name: String,
    pub version: u32,
    pub endianness: String,
    pub crc: Crc,
    pub enums: Vec<EnumDef>,
    pub flags: Flags,
    pub profiles: std::collections::BTreeMap<String, Profile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub versions: Versions,
    pub frames: Vec<Frame>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Versions {
    #[serde(deserialize_with = "de_num")]
    pub request: u16,
    #[serde(deserialize_with = "de_num")]
    pub reply: u16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Crc {
    pub algorithm: String,
    #[serde(deserialize_with = "de_num")]
    pub polynomial: u16,
    #[serde(deserialize_with = "de_num")]
    pub init: u16,
    pub reflect_in: bool,
    pub reflect_out: bool,
    #[serde(deserialize_with = "de_num")]
    pub xor_out: u16,
    pub coverage: String,
    pub field: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnumDef {
    pub name: String,
    pub width: usize,
    pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnumVariant {
    pub name: String,
    #[serde(deserialize_with = "de_num")]
    pub value: u16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Flags {
    pub width: usize,
    pub bits: Vec<FlagBit>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlagBit {
    pub name: String,
    pub bit: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frame {
    pub name: String,
    pub size: usize,
    #[serde(deserialize_with = "de_num")]
    pub magic: u16,
    pub fields: Vec<Field>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub count: usize,
    #[serde(default, deserialize_with = "de_opt_u64")]
    pub value: Option<u64>,
    #[serde(default)]
    pub r#enum: Option<String>,
    #[serde(default)]
    pub flags: bool,
    #[serde(default)]
    pub scale: Option<f64>,
    #[serde(default)]
    pub scales: Vec<f64>,
    #[serde(default)]
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub crc: bool,
}

impl Field {
    pub fn count(&self) -> usize {
        if self.count == 0 {
            1
        } else {
            self.count
        }
    }
}

pub fn type_size(ty: &str) -> Option<usize> {
    match ty {
        "u8" | "i8" => Some(1),
        "u16" | "i16" => Some(2),
        "u32" | "i32" | "f32" => Some(4),
        "u64" | "i64" | "f64" => Some(8),
        _ => None,
    }
}

fn is_integer(ty: &str) -> bool {
    matches!(
        ty,
        "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64"
    )
}

fn ensure(cond: bool, msg: impl AsRef<str>) -> Result<(), String> {
    if cond {
        Ok(())
    } else {
        Err(msg.as_ref().to_string())
    }
}

pub fn parse(text: &str) -> Result<Spec, String> {
    serde_yaml::from_str(text).map_err(|e| format!("YAML parse error: {e}"))
}

pub fn validate(spec: &Spec) -> Result<(), String> {
    let p = &spec.protocol;
    ensure(!p.name.is_empty(), "protocol.name must not be empty")?;
    ensure(p.version > 0, "protocol.version must be > 0")?;
    ensure(
        p.endianness == "little" || p.endianness == "big",
        "protocol.endianness must be 'little' or 'big'",
    )?;

    let crc = &p.crc;
    ensure(crc.algorithm == "crc16", "crc.algorithm must be 'crc16'")?;
    ensure(crc.polynomial != 0, "crc.polynomial must be non-zero")?;
    ensure(
        !crc.reflect_in && !crc.reflect_out,
        "reflected CRC is not supported; set reflect_in/out to false",
    )?;
    ensure(
        crc.coverage == "all_except_crc_field",
        "crc.coverage must be 'all_except_crc_field'",
    )?;
    ensure(!crc.field.is_empty(), "crc.field must not be empty")?;

    ensure(!p.enums.is_empty(), "enums must not be empty")?;
    let mut enum_names = std::collections::HashSet::new();
    for e in &p.enums {
        ensure(!e.name.is_empty(), "enum name must not be empty")?;
        ensure(
            enum_names.insert(e.name.as_str()),
            format!("duplicate enum name {}", e.name),
        )?;
        ensure(
            e.width == 1,
            format!("enum {} width must be 1 byte", e.name),
        )?;
        ensure(
            !e.variants.is_empty(),
            format!("enum {} has no variants", e.name),
        )?;
        let mut names = std::collections::HashSet::new();
        let mut values = std::collections::HashSet::new();
        let max = (1u64 << (e.width * 8)) - 1;
        for v in &e.variants {
            ensure(
                !v.name.is_empty(),
                format!("enum {} has an empty variant name", e.name),
            )?;
            ensure(
                names.insert(v.name.as_str()),
                format!("duplicate variant {} in enum {}", v.name, e.name),
            )?;
            ensure(
                u64::from(v.value) <= max,
                format!("variant {} value out of range", v.name),
            )?;
            ensure(
                values.insert(v.value),
                format!("duplicate variant value {} in enum {}", v.value, e.name),
            )?;
        }
    }

    // Domain invariant: the telemetry phase vocabulary is the documented flight
    // sequence, and its numeric codes are part of the wire contract.
    if let Some(phase) = p.enums.iter().find(|e| e.name == "Phase") {
        const DOCUMENTED: [(&str, u16); 5] = [
            ("BURN", 0),
            ("COAST", 1),
            ("APOGEE", 2),
            ("DESCENT", 3),
            ("LANDED", 4),
        ];
        ensure(
            phase.variants.len() == DOCUMENTED.len(),
            "Phase must have exactly the five documented flight phases",
        )?;
        for (name, value) in DOCUMENTED {
            ensure(
                phase
                    .variants
                    .iter()
                    .any(|v| v.name == name && v.value == value),
                format!("Phase variant {name}={value} is required by the flight-phase vocabulary"),
            )?;
        }
    }

    ensure(p.flags.width == 1, "flags.width must be 1 byte")?;
    ensure(!p.flags.bits.is_empty(), "flags.bits must not be empty")?;
    let mut flag_names = std::collections::HashSet::new();
    let mut flag_bits = std::collections::HashSet::new();
    for b in &p.flags.bits {
        ensure(!b.name.is_empty(), "flag bit name must not be empty")?;
        ensure(
            flag_names.insert(b.name.as_str()),
            format!("duplicate flag name {}", b.name),
        )?;
        ensure(
            b.bit < p.flags.width * 8,
            format!("flag bit {} out of range", b.name),
        )?;
        ensure(
            flag_bits.insert(b.bit),
            format!("duplicate flag bit index {}", b.bit),
        )?;
    }

    ensure(!p.profiles.is_empty(), "profiles must not be empty")?;
    for required in ["compat", "lowlat"] {
        ensure(
            p.profiles.contains_key(required),
            format!("missing profile {required}"),
        )?;
    }

    let mut generated_prefixes = std::collections::HashSet::new();
    for (profile_name, profile) in &p.profiles {
        let required_frames: &[&str] = if profile_name == "compat" {
            &["request", "reply"]
        } else {
            &["lowlat_request", "lowlat_reply"]
        };
        ensure(
            !profile.frames.is_empty(),
            format!("profile {profile_name}: frames must not be empty"),
        )?;

        for frame in &profile.frames {
            let f = frame.name.as_str();
            ensure(!f.is_empty(), "frame name must not be empty")?;
            ensure(
                generated_prefixes.insert(f.to_string()),
                format!("duplicate generated frame prefix {f} across profiles"),
            )?;
            ensure(!frame.fields.is_empty(), format!("frame {f} has no fields"))?;
            ensure(frame.size > 0, format!("frame {f} size must be > 0"))?;

            let mut names = std::collections::HashSet::new();
            let mut crc_count = 0usize;
            let mut offset = 0usize;
            let nfields = frame.fields.len();
            for (i, field) in frame.fields.iter().enumerate() {
                ensure(
                    !field.name.is_empty(),
                    format!("frame {f} has an empty field name"),
                )?;
                ensure(
                    names.insert(field.name.as_str()),
                    format!("duplicate field {}.{}", f, field.name),
                )?;
                let sz = type_size(&field.ty).ok_or_else(|| {
                    format!(
                        "frame {f} field {}: unsupported type {:?}",
                        field.name, field.ty
                    )
                })?;
                offset += sz * field.count();
                if field.crc {
                    crc_count += 1;
                    ensure(
                        i + 1 == nfields,
                        format!("frame {f}: crc field must be last"),
                    )?;
                    ensure(
                        sz == 2 && field.count() == 1,
                        format!("frame {f}: crc field must be u16"),
                    )?;
                    ensure(
                        field.name == crc.field,
                        format!("frame {f}: crc field name must match crc.field"),
                    )?;
                }
                if let Some(v) = field.value {
                    ensure(
                        i == 0 && field.name == "magic",
                        format!("frame {f}: only the first field may be a constant"),
                    )?;
                    ensure(
                        sz == 2 && field.count() == 1,
                        format!("frame {f}: magic must be u16"),
                    )?;
                    ensure(
                        v == u64::from(frame.magic),
                        format!("frame {f}: magic constant disagrees with frame magic"),
                    )?;
                } else if field.name == "magic" {
                    return Err(format!(
                        "frame {f}: magic field is missing its constant value"
                    ));
                }
                if field.crc && field.value.is_some() {
                    return Err(format!("frame {f}: crc field cannot also be a constant"));
                }
                if let Some(ref en) = field.r#enum {
                    ensure(
                        p.enums.iter().any(|e| &e.name == en),
                        format!("frame {f} field {}: unknown enum {en}", field.name),
                    )?;
                    ensure(
                        field.ty == "u8" && field.count() == 1,
                        format!("frame {f} field {}: enum must be u8", field.name),
                    )?;
                }
                if field.flags {
                    ensure(
                        field.ty == "u8" && field.count() == 1,
                        format!("frame {f} field {}: flags must be u8", field.name),
                    )?;
                }
                if field.scale.is_some() {
                    ensure(
                        is_integer(&field.ty),
                        format!(
                            "frame {f} field {}: scale requires an integer type",
                            field.name
                        ),
                    )?;
                }
            }
            ensure(
                crc_count == 1,
                format!("frame {f}: expected exactly one crc field, found {crc_count}"),
            )?;
            ensure(
                offset == frame.size,
                format!(
                    "frame {f}: declared size {} but fields occupy {}",
                    frame.size, offset
                ),
            )?;

            let ver = frame
                .fields
                .iter()
                .find(|x| x.name == "version")
                .ok_or_else(|| format!("frame {f}: missing version field"))?;
            ensure(
                ver.ty == "u8" || ver.ty == "u16",
                format!("frame {f}: version field must be u8 or u16"),
            )?;
            let is_request = f == "request" || f.ends_with("_request");
            let expected_ver = if is_request {
                profile.versions.request
            } else {
                profile.versions.reply
            };
            let max_for_ty = if ver.ty == "u8" {
                u64::from(u8::MAX)
            } else {
                u64::from(u16::MAX)
            };
            ensure(
                u64::from(expected_ver) <= max_for_ty,
                format!("frame {f}: expected version out of range"),
            )?;

            for field in &frame.fields {
                if field.scales.is_empty() && field.feature_names.is_empty() {
                    continue;
                }
                ensure(
                    field.ty == "i16" && field.count() > 1,
                    format!(
                        "frame {f} field {}: feature scales require an i16 array",
                        field.name
                    ),
                )?;
                ensure(
                    field.scales.len() == field.count(),
                    format!(
                        "frame {f} field {}: {} scales for {} features",
                        field.name,
                        field.scales.len(),
                        field.count()
                    ),
                )?;
                ensure(
                    field.feature_names.len() == field.count(),
                    format!(
                        "frame {f} field {}: {} names for {} features",
                        field.name,
                        field.feature_names.len(),
                        field.count()
                    ),
                )?;
                ensure(
                    field.scale.is_none(),
                    format!("frame {f} field {}: use scales[], not scale", field.name),
                )?;
                ensure(
                    field.scales.iter().all(|s| s.is_finite() && *s > 0.0),
                    format!(
                        "frame {f} field {}: scales must be finite and positive",
                        field.name
                    ),
                )?;
                let mut seen = std::collections::HashSet::new();
                for name in &field.feature_names {
                    ensure(
                        name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                        format!("frame {f} field {}: bad feature name {name:?}", field.name),
                    )?;
                    ensure(
                        seen.insert(name.as_str()),
                        format!(
                            "frame {f} field {}: duplicate feature name {name}",
                            field.name
                        ),
                    )?;
                }
            }
        }

        for required in required_frames {
            ensure(
                profile.frames.iter().any(|f| f.name == *required),
                format!("profile {profile_name}: missing frame {required}"),
            )?;
        }
    }

    let lowlat = &p.profiles["lowlat"];
    let request = lowlat
        .frames
        .iter()
        .find(|f| f.name == "lowlat_request")
        .unwrap();
    let features = request
        .fields
        .iter()
        .find(|f| f.name == "features")
        .unwrap();
    ensure(
        features.count() == 14 && features.scales.len() == 14,
        "lowlat_request must carry exactly 14 scaled int16 features",
    )?;
    let reply = lowlat
        .frames
        .iter()
        .find(|f| f.name == "lowlat_reply")
        .unwrap();
    ensure(reply.size == 16, "lowlat_reply must be 16 bytes")?;
    ensure(
        reply
            .fields
            .iter()
            .any(|f| f.name == "compute_us" && f.ty == "u8"),
        "lowlat_reply must carry compute_us as u8",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = include_str!("../../uart.yaml");

    #[test]
    fn shipped_spec_parses_and_validates() {
        let spec = parse(GOOD).unwrap();
        validate(&spec).unwrap();
    }

    #[test]
    fn unknown_top_level_key_is_rejected() {
        let text = GOOD.replacen("protocol:", "protcol:", 1);
        assert!(parse(&text).is_err());
    }

    #[test]
    fn unknown_frame_key_is_rejected() {
        let text = GOOD.replacen("      size: 56", "      sizee: 56", 1);
        assert!(parse(&text).is_err());
    }

    #[test]
    fn unknown_field_key_is_rejected() {
        let text = GOOD.replacen(
            "{ name: seq, type: u16 }",
            "{ name: seq, type: u16, extra: 1 }",
            1,
        );
        assert!(parse(&text).is_err());
    }

    #[test]
    fn bad_frame_size_is_rejected() {
        let text = GOOD.replacen("      size: 56", "      size: 55", 1);
        let spec = parse(&text).unwrap();
        assert!(validate(&spec).unwrap_err().contains("fields occupy"));
    }

    #[test]
    fn phase_uses_documented_names_and_codes() {
        let spec = parse(GOOD).unwrap();
        validate(&spec).unwrap();
        let phase = spec
            .protocol
            .enums
            .iter()
            .find(|e| e.name == "Phase")
            .unwrap();
        let got: Vec<_> = phase
            .variants
            .iter()
            .map(|v| (v.name.as_str(), v.value))
            .collect();
        assert_eq!(
            got,
            vec![
                ("BURN", 0),
                ("COAST", 1),
                ("APOGEE", 2),
                ("DESCENT", 3),
                ("LANDED", 4),
            ]
        );

        let text = GOOD.replace("{ name: BURN, value: 0 }", "{ name: Idle, value: 0 }");
        let bad = parse(&text).unwrap();
        assert!(validate(&bad).unwrap_err().contains("Phase variant BURN=0"));
    }

    #[test]
    fn hex_values_are_accepted() {
        let spec = parse(GOOD).unwrap();
        assert_eq!(spec.protocol.crc.polynomial, 0x1021);
        assert_eq!(spec.protocol.profiles["compat"].frames[0].magic, 0x504D);
        assert_eq!(spec.protocol.profiles["lowlat"].frames[0].magic, 0x4C54);
    }

    #[test]
    fn lowlat_profile_is_validated() {
        let spec = parse(GOOD).unwrap();
        validate(&spec).unwrap();
        let lowlat = &spec.protocol.profiles["lowlat"];
        assert_eq!(lowlat.frames[0].fields[7].scales.len(), 14);
        assert_eq!(lowlat.frames[1].size, 16);

        let text = GOOD
            .replacen("      size: 16", "      size: 14", 1)
            .replacen(
                "            - { name: reserved, type: u8, count: 2 }\n",
                "",
                1,
            );
        let bad = parse(&text).unwrap();
        assert!(validate(&bad)
            .unwrap_err()
            .contains("lowlat_reply must be 16 bytes"));

        let text = GOOD.replacen("0.0001, 0.0001, 0.0001", "0.0, 0.0, 0.0", 1);
        let bad = parse(&text).unwrap();
        assert!(validate(&bad)
            .unwrap_err()
            .contains("scales must be finite and positive"));
    }
}
