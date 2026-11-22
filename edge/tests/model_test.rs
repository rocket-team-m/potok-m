mod common;

use common::fixture;
use potok_edge::model::{Model, ModelError, FORMAT_VERSION, HEADER_BYTES};

#[test]
fn loads_canonical_fixture() {
    let bytes = fixture();
    let model = Model::from_bytes(&bytes).expect("canonical fixture must load");
    assert_eq!(model.n_layers(), 5);
    assert_eq!(model.window, 50);
    assert_eq!(model.channels, 8);
    assert!(model.memory_bytes() > 1000);
}

#[test]
fn refuses_checksum_mismatch() {
    let mut bytes = fixture();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    assert_eq!(
        Model::from_bytes(&bytes).unwrap_err(),
        ModelError::ChecksumMismatch
    );
}

#[test]
fn refuses_bad_magic_and_unsupported_version() {
    let mut bytes = fixture();
    bytes[0] = b'X';
    let mut digest = vec![0u8; 32];
    let n = bytes.len();
    digester(&bytes[..n - 32], &mut digest);
    bytes[n - 32..].copy_from_slice(&digest);
    assert_eq!(Model::from_bytes(&bytes).unwrap_err(), ModelError::BadMagic);

    let mut bytes = fixture();
    bytes[4..8].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
    digester(&bytes[..n - 32], &mut digest);
    bytes[n - 32..].copy_from_slice(&digest);
    assert_eq!(
        Model::from_bytes(&bytes).unwrap_err(),
        ModelError::UnsupportedVersion(FORMAT_VERSION + 1)
    );
}

#[test]
fn refuses_truncated_file() {
    assert_eq!(
        Model::from_bytes(&fixture()[..HEADER_BYTES]).unwrap_err(),
        ModelError::Truncated
    );
}

fn digester(body: &[u8], out: &mut [u8]) {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(body);
    out.copy_from_slice(&d);
}
