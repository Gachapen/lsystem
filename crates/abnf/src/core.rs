use syntax::{Content, Grammar, Item, List, Symbol};

// TODO: Make this constant somehow so that the grammar doesn't have to be constructed each time.
// This is one solution, but requires an atomic check each time it is used:
// http://rust-lang-nursery.github.io/lazy-static.rs/lazy_static/index.html
pub fn rules() -> Grammar {
    let rules = vec![(Symbol::from("ALPHA"), alpha())];

    Grammar::from_rules(rules)
}

pub fn alpha() -> List {
    List::Alternatives(vec![
        Item::new(Content::Range('A', 'Z')),
        Item::new(Content::Range('a', 'z')),
    ])
}
