#[test]
fn sidecar_binary_is_available_to_integration_tests() {
    assert!(std::path::Path::new(env!("CARGO_BIN_EXE_map-sync-node")).is_file());
}
