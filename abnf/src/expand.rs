use syntax::{Content, Item, List, Ruleset};
use core;

pub trait SelectionStrategy {
    fn select_alternative(&mut self, num: usize, rulechain: &Vec<&str>) -> usize;
    fn select_repetition(&mut self, min: u32, max: u32, rulechain: &Vec<&str>) -> u32;
}

enum Node<'a> {
    List(Vec<&'a str>, &'a List),
    Item(Vec<&'a str>, &'a Item),
}

pub fn expand_grammar<S>(grammar: &Ruleset, root: &str, strategy: &mut S) -> String
    where S: SelectionStrategy
{
    let core_rules = core::rules();

    let mut string = String::new();
    let mut visit_stack = vec![Node::List(vec![root], &grammar[root])];

    while !visit_stack.is_empty() {
        let node = visit_stack.pop().unwrap();

        match node {
            Node::List(rulechain, list) => {
                match *list {
                    List::Sequence(ref sequence) => {
                        for item in sequence.iter().rev() {
                            visit_stack.push(Node::Item(rulechain.clone(), item));
                        }
                    },
                    List::Alternatives(ref alternatives) => {
                        let index = strategy.select_alternative(alternatives.len(), &rulechain);
                        visit_stack.push(Node::Item(rulechain, &alternatives[index]));
                    },
                }
            },
            Node::Item(rulechain, item) => {
                let times = match item.repeat {
                    Some(ref repeat) => {
                        let min = repeat.min.unwrap_or(0);
                        let max = repeat.max.unwrap_or(u32::max_value());

                        strategy.select_repetition(min, max, &rulechain)
                    },
                    None => 1,
                };

                for _ in 0..times {
                    match item.content {
                        Content::Value(ref value) => {
                            string.push_str(value);
                        },
                        Content::Symbol(ref symbol) => {
                            let list = if let Some(list) = grammar.get(symbol) {
                                list
                            } else if let Some(list) = core_rules.get(symbol) {
                                list
                            } else {
                                // TODO: Return Result instead of panicing.
                                panic!(format!("Symbol '{}' does not exist in ABNF grammar and is not a core rule", symbol));
                            };

                            let mut updated_chain = rulechain.clone();
                            updated_chain.push(symbol);
                            visit_stack.push(Node::List(updated_chain, list));
                        },
                        Content::Group(ref group) => {
                            visit_stack.push(Node::List(rulechain.clone(), group));
                        },
                        Content::Range(min, max) => {
                            let index = strategy.select_alternative(max as usize - min as usize, &rulechain);
                            let character = (index + min as usize) as u8 as char;
                            string.push(character);
                        }
                    }
                }
            },
        }
    }

    string
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyStrategy {}

    impl SelectionStrategy for DummyStrategy {
        #[allow(unused_variables)]
        fn select_alternative(&mut self, num: usize, rulechain: &Vec<&str>) -> usize {
            0
        }

        #[allow(unused_variables)]
        fn select_repetition(&mut self, min: u32, max: u32, rulechain: &Vec<&str>) -> u32 {
            1
        }
    }

    #[test]
    fn test_expand_value() {
        let mut strategy = DummyStrategy {};
        let mut grammar = Ruleset::new();
        grammar.insert(
            "test".to_string(),
            List::Sequence(vec![
                Item::new(Content::Value("value".to_string())),
            ])
        );

        assert_eq!(expand_grammar(&grammar, "test", &mut strategy), "value".to_string());
    }

    #[test]
    fn test_expand_symbol() {
        let mut strategy = DummyStrategy {};

        let mut rules = Ruleset::new();
        rules.insert(
            "symbol".to_string(),
            List::Sequence(vec![
                Item::new(Content::Value("value".to_string())),
            ])
        );
        rules.insert(
            "test".to_string(),
            List::Sequence(vec![
                Item::new(Content::Symbol("symbol".to_string())),
            ])
        );

        assert_eq!(expand_grammar(&rules, "test", &mut strategy), "value".to_string());
    }

    #[test]
    fn test_expand_sequence() {
        let mut strategy = DummyStrategy {};
        let mut rules = Ruleset::new();
        rules.insert(
            "test".to_string(),
            List::Sequence(vec![
                Item::new(Content::Value("value".to_string())),
                Item::new(Content::Value("value".to_string())),
            ])
        );

        assert_eq!(expand_grammar(&rules, "test", &mut strategy), "valuevalue".to_string());
    }

    #[test]
    fn test_expand_alternatives() {
        let mut strategy = DummyStrategy {};
        let mut rules = Ruleset::new();
        rules.insert(
            "test".to_string(),
            List::Alternatives(vec![
                Item::new(Content::Value("one".to_string())),
                Item::new(Content::Value("two".to_string())),
            ])
        );

        assert_eq!(expand_grammar(&rules, "test", &mut strategy), "one".to_string());
    }

    #[test]
    fn test_expand_group() {
        let mut strategy = DummyStrategy {};
        let mut rules = Ruleset::new();
        rules.insert(
            "test".to_string(),
            List::Sequence(vec![
                Item::new(Content::Group(
                    List::Sequence(vec![
                        Item::new(Content::Value("value".to_string())),
                        Item::new(Content::Value("value".to_string())),
                    ])
                ))
            ])
        );

        assert_eq!(expand_grammar(&rules, "test", &mut strategy), "valuevalue".to_string());
    }
}
