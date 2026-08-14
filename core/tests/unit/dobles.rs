//! Nodos y drivers de mentira, compartidos por los demás módulos.
//!
//! Fíjate en que ninguno declara de qué "tipo" es: lo que los distingue es la
//! variante de `Transition` que devuelven.

use soma_next_core::{Ctx, Driver, DriverError, Node, NodeError, Transition, Value};

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
