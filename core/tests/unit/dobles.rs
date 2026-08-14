//! Filtros, steps y drivers de mentira, compartidos por los demás módulos.

use soma_next_core::{
    Driver, DriverError, Filter, FilterError, Step, StepCtx, StepError, Transition, Value,
};

/// Añade una constante a un escalar.
pub struct Sumar(pub f64);

impl Filter for Sumar {
    fn forward(&self, input: &Value) -> Result<Value, FilterError> {
        match input {
            Value::Number(x) => Ok(Value::number(x + self.0)),
            other => Err(FilterError::new(format!(
                "Sumar necesita un número, le dieron {}",
                other.type_name()
            ))),
        }
    }
}

/// Falla siempre.
pub struct Romper;

impl Filter for Romper {
    fn forward(&self, _input: &Value) -> Result<Value, FilterError> {
        Err(FilterError::new("me rompí"))
    }
}

/// Pide `peticiones` cosas en el turno 0 y devuelve lo que le contesten.
pub struct Preguntar(pub Vec<Value>);

impl Step for Preguntar {
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition, StepError> {
        if ctx.turn == 0 {
            return Ok(Transition::Await(self.0.clone()));
        }
        Ok(Transition::Done(
            ctx.results.first().cloned().unwrap_or(Value::Null),
        ))
    }
}

/// Termina en el primer turno sin pedir nada: un filtro disfrazado de step.
pub struct Inmediato;

impl Step for Inmediato {
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition, StepError> {
        Ok(Transition::Done(ctx.input.clone()))
    }
}

/// No sabe parar.
pub struct Insaciable;

impl Step for Insaciable {
    fn poll(&self, _ctx: &StepCtx<'_>) -> Result<Transition, StepError> {
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

/// La media de lo que le llegue por sus aristas. Un agregador es esto: un
/// filtro que lee un mapa. No hay ningún tipo nuevo detrás.
pub struct Media;

impl Filter for Media {
    fn forward(&self, input: &Value) -> Result<Value, FilterError> {
        let Some(values) = input.values() else {
            return Err(FilterError::new(format!(
                "Media necesita varias entradas, le llegó {}",
                input.type_name()
            )));
        };
        let numeros: Vec<f64> = values
            .iter()
            .map(|v| match v {
                Value::Number(x) => Ok(*x),
                other => Err(FilterError::new(format!(
                    "Media solo promedia números, uno era {}",
                    other.type_name()
                ))),
            })
            .collect::<Result<_, _>>()?;
        Ok(Value::number(
            numeros.iter().sum::<f64>() / numeros.len() as f64,
        ))
    }
}
