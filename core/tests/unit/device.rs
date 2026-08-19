//! The place where a node runs: how it is written and what gets rejected.
//!
//! What is tested here is the enum's honest boundary: the **shape** is
//! validated by the core, the **existence** only torch knows. `cuda:7` compiles
//! on a machine with a single GPU; that it does not exist shows at run time.

use soma_next_core::{Device, DeviceError};

fn read(s: &str) -> Result<Device, DeviceError> {
    s.parse()
}

#[test]
fn the_three_there_are_today() {
    assert_eq!(read("cpu"), Ok(Device::Cpu));
    assert_eq!(read("cuda:0"), Ok(Device::Cuda(0)));
    assert_eq!(read("cuda:3"), Ok(Device::Cuda(3)));
    assert_eq!(read("meta"), Ok(Device::Meta));
}

#[test]
fn it_is_written_the_way_torch_writes_it() {
    // It genuinely matters: what reaches the node is handed to `.to()` as is,
    // without translating along the way.
    assert_eq!(Device::Cpu.to_string(), "cpu");
    assert_eq!(Device::Cuda(1).to_string(), "cuda:1");
    assert_eq!(Device::Meta.to_string(), "meta");
}

#[test]
fn the_round_trip_gives_the_same_thing() {
    for device in [Device::Cpu, Device::Cuda(0), Device::Cuda(7), Device::Meta] {
        assert_eq!(read(&device.to_string()), Ok(device));
    }
}

#[test]
fn a_typo_is_caught_at_declaration_and_not_halfway_through_a_run() {
    // It is the reason this is an enum: a `Device(String)` validated by shape
    // alone would accept `cude:0` and the failure would surface inside torch.
    assert_eq!(read("cude:0"), Err(DeviceError::Unknown("cude".into())));
    assert_eq!(read("gpu:0"), Err(DeviceError::Unknown("gpu".into())));
}

#[test]
fn bare_cuda_is_not_a_placement() {
    // In torch it means "the current GPU", which is thread state. To whoever is
    // placing, that says nothing.
    assert_eq!(read("cuda"), Err(DeviceError::NeedsIndex("cuda".into())));
}

#[test]
fn what_is_not_shaped_like_a_device() {
    for bad in [
        "", "cuda:", "cuda:-1", "cuda:x", "cuda:1:2", "cpu:0", "meta:0",
    ] {
        assert_eq!(
            read(bad),
            Err(DeviceError::Malformed(bad.into())),
            "`{bad}` had to come out malformed"
        );
    }
}

#[test]
fn the_errors_say_what_to_do() {
    assert!(read("cude:0").unwrap_err().to_string().contains("cuda:N"));
    assert!(read("cuda").unwrap_err().to_string().contains("cuda:0"));
}
