//! Nodos y drivers de mentira, compartidos por los demás módulos.
//!
//! Fíjate en que ninguno declara de qué "tipo" es: lo que los distingue es la
//! variante de `Transition` que devuelven.

use soma_next_core::{Ctx, Device, Driver, DriverError, Node, NodeError, Transition, Value};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

/// Añade una constante a un número. Termina siempre.
pub struct Sumar(pub f64);

impl Node for Sumar {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        match input {
            Value::Number(x) => Ok(Transition::Done(Value::number(x + self.0))),
            other => Err(NodeError::new(format!(
                "Sumar necesita un número, le dieron {}",
                other.type_name()
            ))),
        }
    }
}

/// Falla siempre.
pub struct Romper;

impl Node for Romper {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Err(NodeError::new("me rompí"))
    }
}

/// La media de lo que le llegue por sus aristas. Un agregador es esto: un nodo
/// que lee un mapa. No hay ningún tipo nuevo detrás.
pub struct Media;

impl Node for Media {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        let Some(values) = input.values() else {
            return Err(NodeError::new(format!(
                "Media necesita varias entradas, le llegó {}",
                input.type_name()
            )));
        };
        let numeros: Vec<f64> = values
            .iter()
            .map(|v| match v {
                Value::Number(x) => Ok(*x),
                other => Err(NodeError::new(format!(
                    "Media solo promedia números, uno era {}",
                    other.type_name()
                ))),
            })
            .collect::<Result<_, _>>()?;
        Ok(Transition::Done(Value::number(
            numeros.iter().sum::<f64>() / numeros.len() as f64,
        )))
    }
}

/// Devuelve su entrada sin pedir nada.
pub struct Inmediato;

impl Node for Inmediato {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Ok(Transition::Done(input.clone()))
    }
}

/// Pide cosas en el turno 0 y devuelve lo que le contesten.
pub struct Preguntar(pub Vec<Value>);

impl Node for Preguntar {
    fn forward(&self, _input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        if ctx.turn == 0 {
            return Ok(Transition::Await(self.0.clone()));
        }
        Ok(Transition::Done(
            ctx.results.first().cloned().unwrap_or(Value::Null),
        ))
    }
}

/// No sabe parar.
pub struct Insaciable;

impl Node for Insaciable {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Ok(Transition::Await(vec![Value::Null]))
    }
}

/// Contesta a cada petición con su texto en mayúsculas.
pub struct Gritar;

impl Driver for Gritar {
    fn perform(&self, requests: &[Value]) -> Result<Vec<Value>, DriverError> {
        requests
            .iter()
            .map(|r| match r {
                Value::Text(t) => Ok(Value::text(t.to_uppercase())),
                other => Err(DriverError::new(format!(
                    "solo sé gritar texto, me dieron {}",
                    other.type_name()
                ))),
            })
            .collect()
    }
}

/// Contesta cualquier cosa, para probar el tope de turnos.
pub struct SiempreNull;

impl Driver for SiempreNull {
    fn perform(&self, requests: &[Value]) -> Result<Vec<Value>, DriverError> {
        Ok(vec![Value::Null; requests.len()])
    }
}

// ── Los que hacen falta para mirar una wave por dentro ──

/// Por dónde fue pasando la ejecución: quién, en qué orden y en qué hilo.
#[derive(Default)]
pub struct Diario(Mutex<Vec<(String, ThreadId)>>);

impl Diario {
    pub fn nuevo() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn apunta(&self, quien: &str) {
        self.0
            .lock()
            .expect("nadie envenena este mutex")
            .push((quien.to_string(), std::thread::current().id()));
    }

    /// Los nodos en el orden en que se ejecutaron.
    pub fn orden(&self) -> Vec<String> {
        self.0
            .lock()
            .expect("nadie envenena este mutex")
            .iter()
            .map(|(quien, _)| quien.clone())
            .collect()
    }

    /// En qué hilo corrió un nodo.
    pub fn hilo_de(&self, quien: &str) -> ThreadId {
        self.0
            .lock()
            .expect("nadie envenena este mutex")
            .iter()
            .find(|(nombre, _)| nombre == quien)
            .map(|(_, hilo)| *hilo)
            .unwrap_or_else(|| panic!("`{quien}` no llegó a ejecutarse"))
    }
}

/// Se apunta en el diario y devuelve su entrada sin tocarla.
pub struct Testigo(pub &'static str, pub Arc<Diario>);

impl Node for Testigo {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        self.1.apunta(self.0);
        Ok(Transition::Done(input.clone()))
    }
}

/// Un sitio donde varias ramas quedan en verse.
#[derive(Default)]
pub struct Punto {
    llegados: Mutex<usize>,
    aviso: Condvar,
}

impl Punto {
    pub fn nuevo() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

/// No termina hasta que han llegado `cuantos`.
///
/// Es la forma de comprobar que dos ramas corren **a la vez** sin dormir ni un
/// milisegundo: si se ejecutaran una detrás de otra, la primera esperaría para
/// siempre. El plazo convierte ese cuelgue en un error con nombre en vez de en
/// un test que no vuelve.
pub struct Cita {
    pub punto: Arc<Punto>,
    pub cuantos: usize,
    /// Con mensaje, falla con él **después** de llegar a la cita. Sirve para
    /// que dos ramas fallen de verdad a la vez.
    pub falla: Option<&'static str>,
}

impl Node for Cita {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        let plazo = Duration::from_secs(5);
        let mut llegados = self.punto.llegados.lock().expect("nadie lo envenena");
        *llegados += 1;
        if *llegados >= self.cuantos {
            self.punto.aviso.notify_all();
        } else {
            let (guard, espera) = self
                .punto
                .aviso
                .wait_timeout_while(llegados, plazo, |n| *n < self.cuantos)
                .expect("nadie lo envenena");
            llegados = guard;
            if espera.timed_out() {
                return Err(NodeError::new(format!(
                    "solo llegaron {} de {} a la cita: las ramas no corrieron a la vez",
                    *llegados, self.cuantos
                )));
            }
        }
        drop(llegados);

        match self.falla {
            Some(mensaje) => Err(NodeError::new(mensaje)),
            None => Ok(Transition::Done(input.clone())),
        }
    }
}

/// Revienta. Para comprobar que un panic en una rama no se traga.
pub struct Reventar;

impl Node for Reventar {
    fn forward(&self, _input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        panic!("reventé")
    }
}

/// Pide una cosa y devuelve lo que le contesten, pero solo después de que
/// hayan llegado `cuantos` a la cita: comprueba que dos ramas pueden tener al
/// driver ocupado a la vez.
pub struct PreguntarEnCita {
    pub punto: Arc<Punto>,
    pub cuantos: usize,
}

impl Driver for PreguntarEnCita {
    fn perform(&self, requests: &[Value]) -> Result<Vec<Value>, DriverError> {
        let plazo = Duration::from_secs(5);
        let mut llegados = self.punto.llegados.lock().expect("nadie lo envenena");
        *llegados += 1;
        if *llegados >= self.cuantos {
            self.punto.aviso.notify_all();
        } else {
            let (guard, espera) = self
                .punto
                .aviso
                .wait_timeout_while(llegados, plazo, |n| *n < self.cuantos)
                .expect("nadie lo envenena");
            llegados = guard;
            if espera.timed_out() {
                return Err(DriverError::new(
                    "el driver no llegó a estar atendiendo a dos ramas a la vez",
                ));
            }
        }
        drop(llegados);
        Ok(vec![Value::text("atendido"); requests.len()])
    }
}

// ── Los que hacen falta para mirar una colocación por dentro ──

/// Dónde le dijeron a cada nodo que corriera.
#[derive(Default)]
pub struct Registro(Mutex<Vec<(String, Option<Device>)>>);

impl Registro {
    pub fn nuevo() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// El dispositivo que le llegó en el `Ctx`, o `None` si no le llegó
    /// ninguno. Revienta si el nodo no llegó a ejecutarse, que es otra cosa.
    pub fn de(&self, quien: &str) -> Option<Device> {
        self.0
            .lock()
            .expect("nadie envenena este mutex")
            .iter()
            .find(|(nombre, _)| nombre == quien)
            .map(|(_, donde)| donde.clone())
            .unwrap_or_else(|| panic!("`{quien}` no llegó a ejecutarse"))
    }
}

/// Apunta dónde le dijeron que corriera y devuelve su entrada sin tocarla.
///
/// Es todo lo que un nodo del núcleo puede hacer con un dispositivo: verlo.
/// Mover algo a una GPU no es asunto de aquí.
pub struct Ubicuo(pub &'static str, pub Arc<Registro>);

impl Node for Ubicuo {
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        self.1
            .0
            .lock()
            .expect("nadie envenena este mutex")
            .push((self.0.to_string(), ctx.device.cloned()));
        Ok(Transition::Done(input.clone()))
    }
}
