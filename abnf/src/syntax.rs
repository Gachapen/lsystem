use std::collections::HashMap;

#[derive(Copy, Clone, Debug, PartialEq, Default)]
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

#[derive(Debug, PartialEq, Clone)]
pub enum Content {
    Value(String),
    Symbol(String),
    Range(char, char),
    Group(List),
}

#[derive(Debug, PartialEq, Clone)]
pub enum List {
    Sequence(Vec<Item>),
    Alternatives(Vec<Item>),
}

#[derive(Debug, PartialEq, Clone)]
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

pub type Sequence = Vec<Item>;
pub type Ruleset = HashMap<String, List>;
