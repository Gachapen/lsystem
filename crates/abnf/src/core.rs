//! ABNF core rules.
//!
//! Note that because the rules, e.g. `ALPHA`, and the grammar `GRAMMAR` are lazily initialized
//! static variables, there are some things to consider:
//!
//! * You might want to initialize them manually with `initialize`.
//! * Each time a variable is used, it has to do an atomic check, so it is recommended to keep a
//!   reference to the variable if it is used multiple times in the same scope:
//!
//! ```
//! use abnf::Grammar;
//!
//! // It is important to specify the &Grammar type, otherwise it will be a reference to the
//! // `LazyStatic` struct, which will still require the atomic check when it is dereferenced.
//! let grammar: &Grammar = &abnf::core::GRAMMAR;
//! ```

use lazy_static;
use syntax::{Content, Grammar, Item, List, Symbol};

lazy_static! {
    pub static ref ALPHA: List = List::Alternatives(vec![
        Item::new(Content::Range('A', 'Z')),
        Item::new(Content::Range('a', 'z')),
    ]);
    pub static ref GRAMMAR: Grammar = Grammar::from_rules(vec![(Symbol::from("ALPHA"), ALPHA.clone())]);
}

/// Initialize the core grammar and all of its rules.
///
/// Not necessary as they are lazily initialized, but useful if a concrete point of
/// iniitialization is desired.
pub fn initialize() {
    lazy_static::initialize(&GRAMMAR);
}
