#[macro_use]
extern crate nom;

use std::{error, fmt, io};
use std::io::prelude::*;
use std::fs::File;

mod syntax;
mod parse;

pub mod expand;
pub use self::syntax::*;

#[derive(Debug)]
pub enum Error {
    Parse(String),
    Io(io::Error),
}

pub fn parse_file(filename: &str) -> Result<Ruleset, Error> {
    let mut f = match File::open(filename) {
        Ok(file) => file,
        Err(err) => return Err(Error::Io(err)),
    };

    let mut content = Vec::new();

    match f.read_to_end(&mut content) {
        Ok(_) => parse_bytes(&content),
        Err(err) => Err(Error::Io(err)),
    }
}

pub fn parse_string(content: &str) -> Result<Ruleset, Error> {
    parse_bytes(content.as_bytes())
}

pub fn parse_bytes(content: &[u8]) -> Result<Ruleset, Error>  {
    match parse::ruleset(content) {
        nom::IResult::Done(_, item) => Ok(item),
        nom::IResult::Error(_) => Err(Error::Parse("Internal parse module failed".to_string())),
        nom::IResult::Incomplete(_) => Err(Error::Parse("Internal parse module failed".to_string())),
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Io(ref err) => write!(f, "IO error: {}", err),
            Error::Parse(ref err) => write!(f, "Parse error: {}", err),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Io(ref err) => err.description(),
            Error::Parse(ref err) => err,
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::Io(ref err) => Some(err),
            Error::Parse(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bytes() {
        assert!(parse_bytes(b"rule = definition").is_ok());
    }

    #[test]
    fn test_parse_string() {
        assert!(parse_string("rule = definition").is_ok());
    }
}
