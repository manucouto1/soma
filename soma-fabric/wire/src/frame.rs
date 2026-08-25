//! Chunks with a length in front, so a pipe is not a stream.
//!
//! A `pipe` has no messages: it has bytes, and one read can come back with half
//! of what was written. Four length bytes in front turn that into "read all of
//! this, or tell me it is over".

use std::io::{self, Read, Write};

/// The largest chunk accepted. Not a limit of the format — the length is a
/// `u32` — but a safety net: four ASCII characters written to the worker's
/// `stdout` are read as a length of 500 MB to 2 GB, and without the cap a stray
/// `print()` becomes a hung process with no message.
const MAX: usize = 256 * 1024 * 1024;

/// Sends a chunk and pushes it to the other side. The `flush` is not optional:
/// without it both sides wait, and no error.
pub fn send(to: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    if payload.len() > MAX {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "a {} MB chunk does not fit down a pipe (the cap is {} MB); \
                 what weighs that much goes through a store, not the wire",
                payload.len() / (1024 * 1024),
                MAX / (1024 * 1024)
            ),
        ));
    }
    to.write_all(&(payload.len() as u32).to_le_bytes())?;
    to.write_all(payload)?;
    to.flush()
}

/// Reads a whole chunk. `None` if the other side closed **between** chunks,
/// which is finishing; closing halfway through one is the process having died.
pub fn recv(from: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut header = [0u8; 4];
    match from.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let announced = u32::from_le_bytes(header) as usize;
    if announced > MAX {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "the other side announced a {} MB chunk, which cannot be: usually \
                 something has written to the worker's `stdout` and we are reading \
                 that as a length. Check its stderr",
                announced / (1024 * 1024)
            ),
        ));
    }
    let mut payload = vec![0u8; announced];
    from.read_exact(&mut payload)?;
    Ok(Some(payload))
}
