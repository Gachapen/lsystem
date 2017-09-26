use std::cell::Cell;
use std::rc::Rc;

use syntax::{Content, Grammar, Item, List, Symbol};
use core;

pub type Rulechain<'a> = [&'a Symbol];

pub trait SelectionStrategy {
    fn select_alternative(&mut self, num: usize, rulechain: &Rulechain, choice: u32) -> usize;
    fn select_repetition(&mut self, min: u32, max: u32, rulechain: &Rulechain, choice: u32) -> u32;
}

enum Node<'a> {
    List(Vec<&'a Symbol>, Rc<Cell<u32>>, &'a List),
    Item(Vec<&'a Symbol>, Rc<Cell<u32>>, &'a Item),
}

pub fn expand_grammar<S>(grammar: &Grammar, root: &str, strategy: &mut S) -> String
where
    S: SelectionStrategy,
{
    let core_grammar: &Grammar = &core::GRAMMAR;
    let root_symbol = grammar.symbol(root);

    let mut string = String::new();
    let mut visit_stack = vec![
        Node::List(
            vec![&root_symbol],
            Rc::new(Cell::new(0)),
            grammar.map_rule(&root_symbol).unwrap(),
        ),
    ];

    while !visit_stack.is_empty() {
        let node = visit_stack.pop().unwrap();

        match node {
            Node::List(rulechain, choice, list) => match *list {
                List::Sequence(ref sequence) => for item in sequence.iter().rev() {
                    visit_stack.push(Node::Item(rulechain.clone(), choice.clone(), item));
                },
                List::Alternatives(ref alternatives) => {
                    let selection =
                        strategy.select_alternative(alternatives.len(), &rulechain, choice.get());
                    let alternative = &alternatives[selection];
                    visit_stack.push(Node::Item(rulechain.clone(), choice.clone(), alternative));

                    choice.set(choice.get() + 1);
                }
            },
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
                            let list = if let Some(list) = grammar.map_rule(symbol) {
                                list
                            } else if let Some(list) = core_grammar.map_rule(symbol) {
                                list
                            } else {
                                // TODO: Return Result instead of panicing.
                                panic!(format!(
                                    "Symbol '{}' does not exist in ABNF grammar and is \
                                     not a core rule",
                                    symbol.name
                                ));
                            };

                            let mut updated_chain = rulechain.clone();
                            updated_chain.push(symbol);

                            visit_stack
                                .push(Node::List(updated_chain, Rc::new(Cell::new(0)), list));
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
                            let index = strategy.select_alternative(
                                max as usize - min as usize,
                                &rulechain,
                                choice.get(),
                            );
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
    use List::{Alternatives, Sequence};
    use Content;
    use Symbol;

    struct DummyStrategy {}

    impl SelectionStrategy for DummyStrategy {
        fn select_alternative(&mut self, _: usize, _: &Rulechain, _: u32) -> usize {
            0
        }

        fn select_repetition(&mut self, min: u32, _: u32, _: &Rulechain, _: u32) -> u32 {
            min
        }
    }

    #[test]
    fn test_expand_value() {
        let mut strategy = DummyStrategy {};
        let grammar = Grammar::from_rules(vec![
            (
                Symbol::from("test"),
                Sequence(vec![Item::new(Content::Value("value".to_string()))]),
            ),
        ]);

        assert_eq!(
            expand_grammar(&grammar, "test", &mut strategy),
            "value".to_string()
        );
    }

    #[test]
    fn test_expand_symbol() {
        let mut strategy = DummyStrategy {};

        let rules = Grammar::from_rules(vec![
            (
                Symbol::from("symbol"),
                Sequence(vec![Item::new(Content::Value("value".to_string()))]),
            ),
            (
                Symbol::from("test"),
                Sequence(vec![Item::new(Content::Symbol(Symbol::from("symbol")))]),
            ),
        ]);

        assert_eq!(
            expand_grammar(&rules, "test", &mut strategy),
            "value".to_string()
        );
    }

    #[test]
    fn test_expand_sequence() {
        let mut strategy = DummyStrategy {};
        let rules = Grammar::from_rules(vec![
            (
                Symbol::from("test"),
                Sequence(vec![
                    Item::new(Content::Value("value".to_string())),
                    Item::new(Content::Value("value".to_string())),
                ]),
            ),
        ]);

        assert_eq!(
            expand_grammar(&rules, "test", &mut strategy),
            "valuevalue".to_string()
        );
    }

    #[test]
    fn test_expand_alternatives() {
        let mut strategy = DummyStrategy {};
        let rules = Grammar::from_rules(vec![
            (
                Symbol::from("test"),
                Alternatives(vec![
                    Item::new(Content::Value("one".to_string())),
                    Item::new(Content::Value("two".to_string())),
                ]),
            ),
        ]);

        assert_eq!(
            expand_grammar(&rules, "test", &mut strategy),
            "one".to_string()
        );
    }

    #[test]
    fn test_expand_group() {
        let mut strategy = DummyStrategy {};
        let rules = Grammar::from_rules(vec![
            (
                Symbol::from("test"),
                Sequence(vec![
                    Item::new(Content::Group(Sequence(vec![
                        Item::new(Content::Value("value".to_string())),
                        Item::new(Content::Value("value".to_string())),
                    ]))),
                ]),
            ),
        ]);

        assert_eq!(
            expand_grammar(&rules, "test", &mut strategy),
            "valuevalue".to_string()
        );
    }

    #[test]
    fn test_expand_repeat() {
        let mut strategy = DummyStrategy {};
        let rules = Grammar::from_rules(vec![
            (
                Symbol::from("test"),
                Sequence(vec![
                    Item::repeated(
                        Content::Value("value".to_string()),
                        Repeat::with_limits(2, 2),
                    ),
                ]),
            ),
        ]);

        assert_eq!(
            expand_grammar(&rules, "test", &mut strategy),
            "valuevalue".to_string()
        );
    }

    struct VisitsStrategy {
        visits: Vec<(Symbol, u32)>,
    }

    impl VisitsStrategy {
        fn new() -> VisitsStrategy {
            VisitsStrategy { visits: vec![] }
        }
    }

    impl SelectionStrategy for VisitsStrategy {
        fn select_alternative(&mut self, _: usize, rulechain: &Rulechain, choice: u32) -> usize {
            self.visits
                .push(((*rulechain.last().unwrap()).clone(), choice));

            0
        }

        fn select_repetition(
            &mut self,
            min: u32,
            _: u32,
            rulechain: &Rulechain,
            choice: u32,
        ) -> u32 {
            self.visits
                .push(((*rulechain.last().unwrap()).clone(), choice));

            min
        }
    }

    #[test]
    fn test_visit_alternative() {
        let mut strategy = VisitsStrategy::new();
        let rules = Grammar::from_rules(vec![
            (
                Symbol::from("test"),
                Alternatives(vec![
                    Item::new(Content::Value("value".to_string())),
                    Item::new(Content::Value("value".to_string())),
                ]),
            ),
        ]);

        expand_grammar(&rules, "test", &mut strategy);
        assert_eq!(strategy.visits.len(), 1);
        assert_eq!(
            strategy.visits[0],
            (Symbol::with_index("test".to_string(), 0), 0,)
        );
    }

    #[test]
    fn test_visit_repeat() {
        let mut strategy = VisitsStrategy::new();
        let rules = Grammar::from_rules(vec![
            (
                Symbol::from("test"),
                Sequence(vec![
                    Item::repeated(
                        Content::Value("value".to_string()),
                        Repeat::with_limits(0, 1),
                    ),
                ]),
            ),
        ]);

        expand_grammar(&rules, "test", &mut strategy);
        assert_eq!(
            strategy.visits,
            vec![(Symbol::with_index("test".to_string(), 0), 0)]
        );
    }

    #[test]
    fn test_visit_repeat_alternative() {
        let mut strategy = VisitsStrategy::new();
        // test = 2*2("value" / "value")
        let rules = Grammar::from_rules(vec![
            (
                Symbol::from("test"),
                Sequence(vec![
                    Item::repeated(
                        Content::Group(Alternatives(vec![
                            Item::new(Content::Value("value".to_string())),
                            Item::new(Content::Value("value".to_string())),
                        ])),
                        Repeat::with_limits(2, 2),
                    ),
                ]),
            ),
        ]);

        expand_grammar(&rules, "test", &mut strategy);
        assert_eq!(
            strategy.visits,
            vec![
                (Symbol::with_index("test".to_string(), 0), 0),
                (Symbol::with_index("test".to_string(), 0), 1),
                (Symbol::with_index("test".to_string(), 0), 1),
            ]
        );
    }

    #[test]
    fn test_visit_symbol() {
        let mut strategy = VisitsStrategy::new();
        // a = ("value" / "value") b
        // b = "value" / value
        let rules = Grammar::from_rules(vec![
            (
                Symbol::from("a"),
                Sequence(vec![
                    Item::new(Content::Group(Alternatives(vec![
                        Item::new(Content::Value("value".to_string())),
                        Item::new(Content::Value("value".to_string())),
                    ]))),
                    Item::new(Content::Symbol(Symbol::from("b"))),
                ]),
            ),
            (
                Symbol::from("b"),
                Alternatives(vec![
                    Item::new(Content::Value("value".to_string())),
                    Item::new(Content::Value("value".to_string())),
                ]),
            ),
        ]);

        expand_grammar(&rules, "a", &mut strategy);
        assert_eq!(
            strategy.visits,
            vec![
                (Symbol::with_index("a".to_string(), 0), 0),
                (Symbol::with_index("b".to_string(), 1), 0),
            ]
        );
    }
}
