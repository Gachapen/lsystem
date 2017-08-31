use fnv::FnvHashMap;

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
pub struct Symbol {
    pub name: String,
    pub index: Option<usize>,
}

impl<'a> From<&'a str> for Symbol {
    fn from(name: &'a str) -> Symbol {
        Symbol {
            name: name.to_string(),
            index: None,
        }
    }
}

impl From<String> for Symbol {
    fn from(name: String) -> Symbol {
        Symbol {
            name: name,
            index: None,
        }
    }
}

impl Symbol {
    pub fn new(name: String) -> Symbol {
        Symbol::from(name)
    }

    pub fn with_index(name: String, index: usize) -> Symbol {
        Symbol {
            name: name,
            index: Some(index),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Content {
    Value(String),
    Symbol(Symbol),
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
pub type Rule = (Symbol, List);
pub type Ruleset = FnvHashMap<String, List>;

#[derive(Debug)]
pub struct Grammar {
    rules: Vec<List>,
    rule_names: FnvHashMap<usize, String>,
    rule_indices: FnvHashMap<String, usize>,
}

impl Default for Grammar {
    fn default() -> Grammar {
        Grammar {
            rules: Vec::new(),
            rule_names: FnvHashMap::default(),
            rule_indices: FnvHashMap::default(),
        }
    }
}

impl Grammar {
    pub fn map_rule(&self, symbol: &Symbol) -> Option<&List> {
        if let Some(index) = symbol.index {
            self.map_rule_from_index(index)
        } else {
            self.map_rule_from_name(&symbol.name)
        }
    }

    pub fn map_rule_from_name(&self, name: &str) -> Option<&List> {
        let index = self.rule_indices.get(name);
        if let Some(index) = index {
            self.rules.get(*index)
        } else {
            None
        }
    }

    pub fn symbol_index(&self, name: &str) -> Option<usize> {
        let index = self.rule_indices.get(name);
        if let Some(index) = index {
            Some(*index)
        } else {
            None
        }
    }

    pub fn symbol(&self, name: &str) -> Symbol {
        let index = self.rule_indices.get(name);
        if let Some(index) = index {
            Symbol {
                name: name.to_string(),
                index: Some(*index),
            }
        } else {
            Symbol::from(name)
        }
    }

    fn map_rule_from_index(&self, index: usize) -> Option<&List> {
        self.rules.get(index)
    }

    pub fn from_rules(rules: Vec<Rule>) -> Grammar {
        let mut grammar = Grammar::default();

        for (i, ref rule) in rules.iter().enumerate() {
            let &(ref symbol, _) = *rule;
            grammar.rule_names.insert(i, symbol.name.clone());
            grammar.rule_indices.insert(symbol.name.clone(), i);
        }

        fn map_symbols(list: &mut List, grammar: &mut Grammar) {
            fn map_symbols_in_items(items: &mut Vec<Item>, grammar: &mut Grammar) {
                for item in items {
                    match item.content {
                        Content::Symbol(ref mut symbol) => {
                            let index = grammar.rule_indices[&symbol.name];
                            symbol.index = Some(index);
                        }
                        Content::Group(ref mut list) => {
                            map_symbols(list, grammar);
                        }
                        _ => {}
                    }
                }
            }

            match *list {
                List::Sequence(ref mut items) => {
                    map_symbols_in_items(items, grammar);
                }
                List::Alternatives(ref mut items) => {
                    map_symbols_in_items(items, grammar);
                }
            }
        };

        for rule in rules {
            let (_, mut successor) = rule;
            map_symbols(&mut successor, &mut grammar);
            grammar.rules.push(successor);
        }

        grammar
    }
}
