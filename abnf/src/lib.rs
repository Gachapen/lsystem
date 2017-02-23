#[macro_use]
extern crate nom;

mod syntax;
pub mod parse;

pub use self::syntax::*;

//type ParseResult = Result<Vec<Rule>, String>;

//pub fn parse_str(string: &str) -> ParseResult {
//    Err("Failed".to_string())
//}
