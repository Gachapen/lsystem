#[derive(Debug, PartialEq)]
pub struct Repeat {
    pub min: Option<u32>,
    pub max: Option<u32>,
}

impl Repeat {
    pub fn new() -> Repeat {
        Repeat {
            min: None,
            max: None,
        }
    }

    pub fn with_limits(min: u32, max: u32) -> Repeat {
        Repeat {
            min: Some(min),
            max: Some(max),
        }
    }

    pub fn with_min(min: u32) -> Repeat {
        Repeat {
            min: Some(min),
            ..Repeat::new()
        }
    }

    pub fn with_max(max: u32) -> Repeat {
        Repeat {
            max: Some(max),
            ..Repeat::new()
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum CoreRule {
    Alpha,
}

// string (string string) ALPHA (string / string)
#[derive(Debug, PartialEq)]
pub enum Content {
    Group(Vec<Item>),
    Alternatives(Vec<Item>),
    Core(CoreRule),
    Symbol(String),
    Value(String),
}

#[derive(Debug, PartialEq)]
pub struct Item {
    pub repeat: Option<Repeat>,
    pub content: Content,
}

impl Item {
    pub fn new(content: Content) -> Item {
        Item {
            repeat: None,
            content: content,
        }
    }

    pub fn repeated(content: Content, repeat: Repeat) -> Item {
        Item {
            repeat: Some(repeat),
            ..Item::new(content)
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Rule {
    pub name: String,
    pub definition: Item,
}
