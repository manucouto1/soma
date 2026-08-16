//! Dónde corre un nodo.
//!
//! Es un enum y no un `String` validado porque así un typo es un error **al
//! declarar**: `.on("cude:0")` falla donde se escribió, con el nombre delante,
//! en vez de convertirse en un `RuntimeError` de torch a mitad de un run. Un
//! `Device(String)` que solo comprobara la forma daría `cude:0` por bueno.
//!
//! El precio es que el vocabulario pasa a ser nuestro y no de torch: un
//! backend que no esté aquí no se puede declarar hasta que lo añadamos. Sale
//! barato, y es la razón de que se pueda pagar: **el núcleo no hace `match`
//! sobre un `Device` en ninguna otra parte**. No decide nada según cuál sea,
//! solo lo transporta hasta el nodo. Añadir una variante son tres líneas —el
//! enum, un brazo de [`FromStr`] y otro de [`Display`](fmt::Display)— y ningún
//! sitio más deja de compilar.
//!
//! Solo están las que tienen consumidor hoy. `mps`, `xpu` y compañía entran el
//! día que alguien las ejecute.

use std::fmt;
use std::str::FromStr;

/// El sitio donde se ejecuta un nodo.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    /// El procesador.
    Cpu,
    /// Una GPU de CUDA, por índice.
    ///
    /// El índice es obligatorio a propósito. En torch, `"cuda"` a secas
    /// significa «la GPU actual», que es estado del hilo; para quien coloca,
    /// «la actual» no es una colocación. Una declaración ambigua no se puede
    /// escribir.
    Cuda(usize),
    /// El dispositivo `meta` de torch: forma y dtype, sin memoria ni cómputo.
    ///
    /// Está porque es el único que permite probar que una colocación llega y
    /// se obedece **en cualquier máquina**, sin GPU.
    Meta,
}

impl FromStr for Device {
    type Err = DeviceError;

    /// `cpu`, `cuda:0`, `meta`. Tal cual los escribe torch, para que lo que
    /// llega al nodo se le pueda pasar a `.to()` sin traducir nada.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(DeviceError::Malformed(s.to_string()));
        }
        let (kind, index) = match s.split_once(':') {
            Some((kind, index)) => (kind, Some(index)),
            None => (s, None),
        };
        match (kind, index) {
            ("cpu", None) => Ok(Self::Cpu),
            ("meta", None) => Ok(Self::Meta),
            ("cuda", Some(index)) => index
                .parse()
                .map(Self::Cuda)
                .map_err(|_| DeviceError::Malformed(s.to_string())),
            ("cuda", None) => Err(DeviceError::NeedsIndex(kind.to_string())),
            ("cpu" | "meta", Some(_)) => Err(DeviceError::Malformed(s.to_string())),
            _ => Err(DeviceError::Unknown(kind.to_string())),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda(index) => write!(f, "cuda:{index}"),
            Self::Meta => f.write_str("meta"),
        }
    }
}

// ── Lo que puede salir mal al nombrar un dispositivo ──

/// Por qué eso no nombra un sitio donde ejecutar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    /// No conocemos ese tipo de dispositivo.
    Unknown(String),
    /// El tipo es de los nuestros, pero lo que lo acompaña no.
    Malformed(String),
    /// Falta decir cuál de todos.
    NeedsIndex(String),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(kind) => write!(
                f,
                "no conozco el dispositivo `{kind}`; hoy hay `cpu`, `cuda:N` y `meta`"
            ),
            Self::Malformed(s) => write!(
                f,
                "`{s}` no tiene forma de dispositivo; se escribe `cpu`, `cuda:N` o `meta`"
            ),
            Self::NeedsIndex(kind) => write!(
                f,
                "`{kind}` no dice cuál: escribe `{kind}:0`. «El actual» es estado \
                 del hilo, no una colocación"
            ),
        }
    }
}

impl std::error::Error for DeviceError {}
