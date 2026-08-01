//! Downcasting, without every implementor writing the same three lines.

use std::any::Any;

/// Erase to `dyn Any`, so a `dyn Filter` or `dyn Step` can be downcast to
/// the concrete type behind it.
///
/// A supertrait with a blanket impl rather than a required method.
/// `Filter` and `Step` both demanded `fn as_any(&self) -> &dyn Any { self }`
/// from every implementor — sixty-odd identical bodies across the
/// workspace, and one more required of anyone writing a filter of their
/// own — to serve two downcasts in the whole codebase.
pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Any> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
