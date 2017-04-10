use std::cell::Cell;
use std::rc::Rc;

use syntax::{Content, Item, List, Ruleset};
use core;

pub trait SelectionStrategy {
    fn select_alternative(&mut self, num: usize, rulechain: &[&str], choice: u32) -> usize;
    fn select_repetition(&mut self, min: u32, max: u32, rulechain: &[&str], choice: u32) -> u32;
}

enum Node<'a> {
    List(Vec<&'a str>, Rc<Cell<u32>>, &'a List),
    Item(Vec<&'a str>, Rc<Cell<u32>>, &'a Item),
}

pub fn expand_grammar<S>(grammar: &Ruleset, root: &str, strategy: &mut S) -> String
    where S: SelectionStrategy
{
    let core_rules = core::rules();

    let mut string = String::new();
    let mut visit_stack = vec![Node::List(vec![root], Rc::new(Cell::new(0)), &grammar[root])];

    while !visit_stack.is_empty() {
        let node = visit_stack.pop().unwrap();

        match node {
            Node::List(rulechain, choice, list) => {
                match *list {
                    List::Sequence(ref sequence) => {
                        for item in sequence.iter().rev() {
                            visit_stack.push(Node::Item(rulechain.clone(), choice.clone(), item));
                        }
                    }
                    List::Alternatives(ref alternatives) => {
                        let selection = strategy.select_alternative(alternatives.len(),
                                                                    &rulechain,
                                                                    choice.get());
                        let alternative = &alternatives[selection];
                        visit_stack.push(Node::Item(rulechain.clone(),
                                                    choice.clone(),
                                                    alternative));

                        choice.set(choice.get() + 1);
                    }
                }
            }
            Node::Item(rulechain, choice, item) => {
                let times = match item.repeat {
                    Some(ref repeat) => {
                        let min = repeat.min.unwrap_or(0);
                        let max = repeat.max.unwrap_or(u32::max_value());

                        let times = strategy.select_repetition(min, max, &rulechain, choice.get());
                        choice.set(choice.get() + 1);

                        times
                    }
                    None => 1,
                };

                for i in 0..times {
                    match item.content {
                        Content::Value(ref value) => {
                            string.push_str(value);
                        }
                        Content::Symbol(ref symbol) => {
                            let list = if let Some(list) = grammar.get(symbol) {
                                list
                            } else if let Some(list) = core_rules.get(symbol) {
                                list
                            } else {
                                // TODO: Return Result instead of panicing.
                                panic!(format!("Symbol '{}' does not exist in ABNF grammar and is \
                                                not a core rule",
                                               symbol));
                            };

                            let mut updated_chain = rulechain.clone();
                            updated_chain.push(symbol);

                            visit_stack.push(Node::List(updated_chain,
                                                        Rc::new(Cell::new(0)),
                                                        list));
                        }
                        Content::Group(ref group) => {
                            // Hacky mack hack to prevent each repeat to count towards the total
                            // choice number. Now only the first will.
                            let choice_clone = {
                                if i == 0 {
                                    choice.clone()
                                } else {
                                    Rc::new(Cell::new(choice.get()))
                                }
                            };
                            visit_stack.push(Node::List(rulechain.clone(), choice_clone, group));
                        }
                        Content::Range(min, max) => {
                            let index = strategy.select_alternative(max as usize - min as usize,
                                                                    &rulechain,
                                                                    choice.get());
                            let character = (index + min as usize) as u8 as char;
                            string.push(character);

                            choice.set(choice.get() + 1);
                        }
                    }
                }
            }
        }
    }

    string
}

#[cfg(test)]
mod tests {
    use super::*;
    use syntax::Repeat;
    use List::{Sequence, Alternatives};
    use Content::{Value, Symbol, Group};

    struct DummyStrategy {}

    impl SelectionStrategy for DummyStrategy {
        fn select_alternative(&mut self, _: usize, _: &[&str], _: u32) -> usize {
            0
        }

        fn select_repetition(&mut self, min: u32, _: u32, _: &[&str], _: u32) -> u32 {
            min
        }
    }

    #[test]
    fn test_expand_value() {
        let mut strategy = DummyStrategy {};
        let mut grammar = Ruleset::new();
        grammar.insert("test".to_string(),
                       Sequence(vec![Item::new(Value("value".to_string()))]));

        assert_eq!(expand_grammar(&grammar, "test", &mut strategy),
                   "value".to_string());
    }

    #[test]
    fn test_expand_symbol() {
        let mut strategy = DummyStrategy {};

        let mut rules = Ruleset::new();
        rules.insert("symbol".to_string(),
                     Sequence(vec![Item::new(Value("value".to_string()))]));
        rules.insert("test".to_string(),
                     Sequence(vec![Item::new(Symbol("symbol".to_string()))]));

        assert_eq!(expand_grammar(&rules, "test", &mut strategy),
                   "value".to_string());
    }

    #[test]
    fn test_expand_sequence() {
        let mut strategy = DummyStrategy {};
        let mut rules = Ruleset::new();
        rules.insert("test".to_string(),
                     Sequence(vec![Item::new(Value("value".to_string())),
                                   Item::new(Value("value".to_string()))]));

        assert_eq!(expand_grammar(&rules, "test", &mut strategy),
                   "valuevalue".to_string());
    }

    #[test]
    fn test_expand_alternatives() {
        let mut strategy = DummyStrategy {};
        let mut rules = Ruleset::new();
        rules.insert("test".to_string(),
                     Alternatives(vec![Item::new(Value("one".to_string())),
                                       Item::new(Value("two".to_string()))]));

        assert_eq!(expand_grammar(&rules, "test", &mut strategy),
                   "one".to_string());
    }

    #[test]
    fn test_expand_group() {
        let mut strategy = DummyStrategy {};
        let mut rules = Ruleset::new();
        rules.insert("test".to_string(),
                     Sequence(vec![Item::new(Group(Sequence(vec![
                        Item::new(Value("value".to_string())),
                        Item::new(Value("value".to_string())),
                     ])))]));

        assert_eq!(expand_grammar(&rules, "test", &mut strategy),
                   "valuevalue".to_string());
    }

    #[test]
    fn test_expand_repeat() {
        let mut strategy = DummyStrategy {};
        let mut rules = Ruleset::new();
        rules.insert("test".to_string(),
                     Sequence(vec![Item::repeated(Value("value".to_string()),
                                                  Repeat::with_limits(2, 2))]));

        assert_eq!(expand_grammar(&rules, "test", &mut strategy),
                   "valuevalue".to_string());
    }

    struct VisitsStrategy {
        visits: Vec<(String, u32)>,
    }

    impl VisitsStrategy {
        fn new() -> VisitsStrategy {
            VisitsStrategy { visits: vec![] }
        }
    }

    impl SelectionStrategy for VisitsStrategy {
        fn select_alternative(&mut self, _: usize, rulechain: &[&str], choice: u32) -> usize {
            self.visits.push((rulechain.last().unwrap().to_string(), choice));

            0
        }

        fn select_repetition(&mut self,
                             min: u32,
                             _: u32,
                             rulechain: &[&str],
                             choice: u32)
                             -> u32 {
            self.visits.push((rulechain.last().unwrap().to_string(), choice));

            min
        }
    }

    #[test]
    fn test_visit_alternative() {
        let mut strategy = VisitsStrategy::new();
        let mut rules = Ruleset::new();

        rules.insert("test".to_string(),
                     Alternatives(vec![Item::new(Value("value".to_string())),
                                       Item::new(Value("value".to_string()))]));

        expand_grammar(&rules, "test", &mut strategy);
        assert_eq!(strategy.visits.len(), 1);
        assert_eq!(strategy.visits[0], ("test".to_string(), 0));
    }

    #[test]
    fn test_visit_repeat() {
        let mut strategy = VisitsStrategy::new();
        let mut rules = Ruleset::new();

        rules.insert("test".to_string(),
                     Sequence(vec![Item::repeated(Value("value".to_string()),
                                                  Repeat::with_limits(0, 1))]));

        expand_grammar(&rules, "test", &mut strategy);
        assert_eq!(strategy.visits, vec![("test".to_string(), 0)]);
    }

    #[test]
    fn test_visit_repeat_alternative() {
        let mut strategy = VisitsStrategy::new();
        let mut rules = Ruleset::new();

        // test = 2*2("value" / "value")
        rules.insert("test".to_string(),
                     Sequence(vec![Item::repeated(Group(Alternatives(vec![
                                                    Item::new(Value("value".to_string())),
                                                    Item::new(Value("value".to_string())),
                                                  ])),
                                                  Repeat::with_limits(2, 2))]));

        expand_grammar(&rules, "test", &mut strategy);
        assert_eq!(strategy.visits,
                   vec![("test".to_string(), 0),
                        ("test".to_string(), 1),
                        ("test".to_string(), 1)]);
    }

    #[test]
    fn test_visit_symbol() {
        let mut strategy = VisitsStrategy::new();
        let mut rules = Ruleset::new();

        // a = ("value" / "value") b
        // b = "value" / value
        rules.insert("a".to_string(),
                     Sequence(vec![Item::new(Group(Alternatives(vec![
                                    Item::new(Value("value".to_string())),
                                    Item::new(Value("value".to_string())),
                                   ]))),
                                   Item::new(Symbol("b".to_string()))]));
        rules.insert("b".to_string(),
                     Alternatives(vec![Item::new(Value("value".to_string())),
                                       Item::new(Value("value".to_string()))]));

        expand_grammar(&rules, "a", &mut strategy);
        assert_eq!(strategy.visits,
                   vec![("a".to_string(), 0), ("b".to_string(), 0)]);
    }
}
