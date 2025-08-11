//! Error handling for the QVM scheduler

use thiserror::Error;

/// Result type for QVM operations
pub type Result<T> = core::result::Result<T, QvmError>;

/// Comprehensive error types for the QVM scheduler
#[derive(Error, Debug, Clone, PartialEq)]
pub enum QvmError {
    /// Circuit parsing errors
    #[error("Parse error: {message} at position {position}")]
    ParseError { message: String, position: usize },

    /// Invalid circuit structure
    #[error("Invalid circuit: {0}")]
    InvalidCircuit(String),

    /// Topology-related errors
    #[error("Topology error: {0}")]
    TopologyError(String),

    /// Scheduling conflicts
    #[error("Scheduling error: {0}")]
    SchedulingError(String),

    /// Resource allocation failures
    #[error("Resource allocation failed: {0}")]
    AllocationError(String),

    /// Composition failures
    #[error("Circuit composition failed: {0}")]
    CompositionError(String),

    /// I/O errors (std feature only)
    #[cfg(feature = "std")]
    #[error("I/O error: {0}")]
    IoError(String),

    /// Invalid configuration
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Internal errors that shouldn't happen
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl QvmError {
    /// Create a parse error
    pub fn parse_error(message: impl Into<String>, position: usize) -> Self {
        Self::ParseError {
            message: message.into(),
            position,
        }
    }

    /// Create an invalid circuit error
    pub fn invalid_circuit(message: impl Into<String>) -> Self {
        Self::InvalidCircuit(message.into())
    }

    /// Create a topology error
    pub fn topology_error(message: impl Into<String>) -> Self {
        Self::TopologyError(message.into())
    }

    /// Create a scheduling error
    pub fn scheduling_error(message: impl Into<String>) -> Self {
        Self::SchedulingError(message.into())
    }

    /// Create an allocation error
    pub fn allocation_error(message: impl Into<String>) -> Self {
        Self::AllocationError(message.into())
    }

    /// Create a composition error
    pub fn composition_error(message: impl Into<String>) -> Self {
        Self::CompositionError(message.into())
    }

    /// Create a configuration error
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::ConfigError(message.into())
    }

    /// Create an internal error
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::InternalError(message.into())
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for QvmError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for QvmError {
    fn from(err: serde_json::Error) -> Self {
        Self::ParseError {
            message: err.to_string(),
            position: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = QvmError::parse_error("Expected token", 10);
        assert_eq!(
            err.to_string(),
            "Parse error: Expected token at position 10"
        );
    }

    #[test]
    fn test_error_chain() {
        let err = QvmError::scheduling_error("No resources available");
        assert!(err.to_string().contains("scheduling"));
    }
}