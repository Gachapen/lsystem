use syntax::{Ruleset, List, Item, Content};

pub fn rules() -> Ruleset {
    let mut ruleset = Ruleset::new();

    ruleset.insert("ALPHA".to_string(), alpha());

    ruleset
}

pub fn alpha() -> List {
    List::Alternatives(vec![
        Item::new(Content::Range('A', 'Z')),
        Item::new(Content::Range('a', 'z')),
    ])
}
