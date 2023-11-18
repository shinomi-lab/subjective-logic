/// An error indicating that one or more invalid values are used.
#[derive(thiserror::Error, Debug)]
#[error("At least one parameter is invalid because {0}.")]
pub struct InvalidValueError(pub String);
