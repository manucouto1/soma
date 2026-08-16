//! El sitio donde corre un nodo: cómo se escribe y qué se rechaza.
//!
//! Lo que se prueba aquí es la frontera honesta del enum: la **forma** la
//! valida el núcleo, la **existencia** solo la sabe torch. `cuda:7` compila en
//! una máquina con una sola GPU; que no exista se ve al ejecutar.

use soma_next_core::{Device, DeviceError};

fn lee(s: &str) -> Result<Device, DeviceError> {
    s.parse()
}

#[test]
fn los_tres_que_hay_hoy() {
    assert_eq!(lee("cpu"), Ok(Device::Cpu));
    assert_eq!(lee("cuda:0"), Ok(Device::Cuda(0)));
    assert_eq!(lee("cuda:3"), Ok(Device::Cuda(3)));
    assert_eq!(lee("meta"), Ok(Device::Meta));
}

#[test]
fn se_escribe_como_lo_escribe_torch() {
    // Importa de verdad: lo que llega al nodo se le pasa a `.to()` tal cual,
    // sin traducir por el camino.
    assert_eq!(Device::Cpu.to_string(), "cpu");
    assert_eq!(Device::Cuda(1).to_string(), "cuda:1");
    assert_eq!(Device::Meta.to_string(), "meta");
}

#[test]
fn la_ida_y_la_vuelta_dan_lo_mismo() {
    for device in [Device::Cpu, Device::Cuda(0), Device::Cuda(7), Device::Meta] {
        assert_eq!(lee(&device.to_string()), Ok(device));
    }
}

#[test]
fn un_typo_se_cuenta_al_declarar_y_no_a_mitad_de_un_run() {
    // Es la razón de que sea un enum: un `Device(String)` validado solo por
    // forma daría `cude:0` por bueno y el fallo saldría dentro de torch.
    assert_eq!(lee("cude:0"), Err(DeviceError::Unknown("cude".into())));
    assert_eq!(lee("gpu:0"), Err(DeviceError::Unknown("gpu".into())));
}

#[test]
fn cuda_a_secas_no_es_una_colocacion() {
    // En torch significa «la GPU actual», que es estado del hilo. Para quien
    // coloca, eso no dice nada.
    assert_eq!(lee("cuda"), Err(DeviceError::NeedsIndex("cuda".into())));
}

#[test]
fn lo_que_no_tiene_forma_de_dispositivo() {
    for malo in [
        "", "cuda:", "cuda:-1", "cuda:x", "cuda:1:2", "cpu:0", "meta:0",
    ] {
        assert_eq!(
            lee(malo),
            Err(DeviceError::Malformed(malo.into())),
            "`{malo}` tenía que salir malformado"
        );
    }
}

#[test]
fn los_errores_dicen_que_hacer() {
    assert!(lee("cude:0").unwrap_err().to_string().contains("cuda:N"));
    assert!(lee("cuda").unwrap_err().to_string().contains("cuda:0"));
}
