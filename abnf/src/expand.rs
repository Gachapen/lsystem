use syntax::{CoreRule, Content, Item, List, Ruleset};

// TODO: Make configurable?
const MAX_DEPTH: u32 = 50;

pub trait SelectionStrategy {
    fn select_alternative(&mut self, num: usize) -> usize;
    fn select_repetition(&mut self, min: u32, max: u32) -> u32;
}

pub fn expand_grammar<T>(grammar: &Ruleset, root: &str, strategy: &mut T) -> String
    where T: SelectionStrategy
{
    expand_list(&grammar[root], 0, grammar, strategy)
}

pub fn expand_list<T>(list: &List, depth: u32, grammar: &Ruleset, strategy: &mut T) -> String
    where T: SelectionStrategy
{
    match *list {
        List::Sequence(ref sequence) => {
            let mut string = String::new();

            for item in sequence {
                string.push_str(&expand_item(item, depth + 1, grammar, strategy));
            }

            string
        },
        List::Alternatives(ref alternatives) => {
            let index = strategy.select_alternative(alternatives.len());
            expand_item(&alternatives[index], depth + 1, grammar, strategy)
        },
    }
}

pub fn expand_item<T>(item: &Item, depth: u32, grammar: &Ruleset, strategy: &mut T) -> String
    where T: SelectionStrategy
{
    if depth > MAX_DEPTH {
        return String::new();
    }

    let times = match item.repeat {
        Some(ref repeat) => {
            let min = repeat.min.unwrap_or(0);
            let max = repeat.max.unwrap_or(u32::max_value());

            strategy.select_repetition(min, max)
        },
        None => 1,
    };

    let mut string = String::new();

    for _ in 0..times {
        let expanded = match item.content {
            Content::Value(ref value) => {
                value.clone()
            },
            Content::Symbol(ref symbol) => {
                expand_list(&grammar[symbol], depth, grammar, strategy)
            },
            Content::Group(ref group) => {
                expand_list(group, depth, grammar, strategy)
            },
            Content::Core(rule) => {
                let content = expand_core_rule(rule);
                expand_item(&Item::new(content), depth + 1, grammar, strategy)
            },
            Content::Range(min, max) => {
                let index = strategy.select_alternative(max as usize - min as usize);
                let character = (index + min as usize) as u8 as char;

                // Probably not the most efficient...
                let mut string = String::new();
                string.push(character);
                string
            }
        };

        string.push_str(&expanded);
    };

    string
}

pub fn expand_core_rule(rule: CoreRule) -> Content {
    use CoreRule::*;
    use Content::*;
    use List::*;

    match rule {
        Alpha => {
            Group(
                Alternatives(vec![
                    Item::new(Range('A', 'Z')),
                    Item::new(Range('a', 'z')),
                ])
            )
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyStrategy {}

    impl SelectionStrategy for DummyStrategy {
        #[allow(unused_variables)]
        fn select_alternative(&mut self, num: usize) -> usize {
            0
        }

        #[allow(unused_variables)]
        fn select_repetition(&mut self, min: u32, max: u32) -> u32 {
            1
        }
    }

    #[test]
    fn test_expand_value() {
        let rules = Ruleset::new();
        let mut strategy = DummyStrategy {};
        let item = Item::new(Content::Value("value".to_string()));

        assert_eq!(expand_item(&item, 0, &rules, &mut strategy), "value".to_string());
    }

    #[test]
    fn test_expand_symbol() {
        let mut rules = Ruleset::new();
        rules.insert(
            "symbol".to_string(),
            List::Sequence(vec![
                Item::new(Content::Value("value".to_string())),
            ])
        );

        let mut strategy = DummyStrategy {};
        let item = Item::new(Content::Symbol("symbol".to_string()));

        assert_eq!(expand_item(&item, 0, &rules, &mut strategy), "value".to_string());
    }

    #[test]
    fn test_expand_core() {
        let rules = Ruleset::new();
        let mut strategy = DummyStrategy {};
        let item = Item::new(Content::Core(CoreRule::Alpha));

        assert_eq!(expand_item(&item, 0, &rules, &mut strategy), "A".to_string());
    }

    #[test]
    fn test_expand_range() {
        let rules = Ruleset::new();
        let mut strategy = DummyStrategy {};
        let item = Item::new(Content::Range('X', 'Z'));

        assert_eq!(expand_item(&item, 0, &rules, &mut strategy), "X".to_string());
    }

    #[test]
    fn test_expand_sequence() {
        let rules = Ruleset::new();
        let mut strategy = DummyStrategy {};
        let list = List::Sequence(vec![
            Item::new(Content::Value("value".to_string())),
            Item::new(Content::Value("value".to_string())),
        ]);

        assert_eq!(expand_list(&list, 0, &rules, &mut strategy), "valuevalue".to_string());
    }

    #[test]
    fn test_expand_alternatives() {
        let rules = Ruleset::new();
        let mut strategy = DummyStrategy {};
        let list = List::Alternatives(vec![
            Item::new(Content::Value("one".to_string())),
            Item::new(Content::Value("two".to_string())),
        ]);

        assert_eq!(expand_list(&list, 0, &rules, &mut strategy), "one".to_string());
    }

    #[test]
    fn test_expand_group() {
        let rules = Ruleset::new();
        let mut strategy = DummyStrategy {};
        let item = Item::new(Content::Group(
            List::Sequence(vec![
                Item::new(Content::Value("value".to_string())),
                Item::new(Content::Value("value".to_string())),
            ])
        ));

        assert_eq!(expand_item(&item, 0, &rules, &mut strategy), "valuevalue".to_string());
    }
}
