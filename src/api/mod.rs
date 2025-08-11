//! API modules for different interfaces

pub mod cli;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports
pub use cli::*;

#[cfg(feature = "wasm")]
pub use wasm::*;