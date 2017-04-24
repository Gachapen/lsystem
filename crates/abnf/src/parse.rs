use nom::{alphanumeric, digit, eol, space, hex_u32};
use std::str;

use syntax::{Repeat, Item, Content, Sequence, Ruleset, List};

named!(pub string_literal<String>,
    delimited!(
        char!('"'),
        map_res!(
            fold_many0!(
                is_not_s!("\""),
                Vec::new(),
                |mut acc: Vec<_>, item| {
                    acc.extend_from_slice(item);
                    acc
                }
            ),
            String::from_utf8
        ),
        char!('"')
    )
);

named!(pub hex_value<char>,
    map!(
        do_parse!(
            tag!(&b"%x"[..]) >>
            value: call!(hex_u32) >>
            (value)
        ),
        |v| {
            v as u8 as char
        }
    )
);

// TODO: Add support more ranges with other formats than hex.
named!(pub range<(char, char)>,
    map!(
        do_parse!(
            first: hex_value >>
            char!('-') >>
            last: call!(hex_u32) >>
            (first, last)
        ),
        |(a, b)| {
            (a, b as u8 as char)
        }
    )
);

named!(pub number<u32>,
    fold_many1!(
        call!(digit),
        0u32,
        |mut acc: u32, item| {
            acc += u32::from_str_radix(str::from_utf8(item).unwrap(), 10).unwrap();
            acc
        }
    )
);

named!(pub repeat<Repeat>,
    map!(
        do_parse!(
            min: opt!(complete!(number)) >>
            char!('*') >>
            max: opt!(complete!(number)) >>
            (min, max)
        ),
        |(min, max)| {
            Repeat {
                min: min,
                max: max,
            }
        }
    )
);

named!(pub symbol<String>,
    map!(
        map_res!(
            call!(alphanumeric),
            str::from_utf8
        ),
        String::from
    )
);

named!(pub content<Content>,
    alt!(
        map!(call!(range), |(a, b)| Content::Range(a, b)) |
        map!(call!(hex_value), |c: char| Content::Value(c.to_string())) |
        map!(call!(string_literal), |s| Content::Value(s)) |
        map!(call!(symbol), |s| Content::Symbol(s)) |
        map!(call!(group), |g| Content::Group(g))
    )
);

named!(pub item<Item>,
    map!(
        do_parse!(
            r: opt!(repeat) >>
            c: call!(content) >>
            (r, c)
        ),
        |(repeat, content)| {
            Item {
                repeat: repeat,
                content: content,
            }
        }
    )
);

named!(pub sequence<Sequence>,
    separated_nonempty_list!(
        call!(space),
        call!(item)
    )
);

named!(pub alternatives_sep<char>,
    delimited!(
        call!(space),
        char!('/'),
        call!(space)
    )
);

named!(pub alternatives<Sequence>,
    map!(
        complete!(do_parse!(
            first: call!(item) >>
            remaining: many1!(
                preceded!(
                    call!(alternatives_sep),
                    call!(item)
                )
            ) >>
            (first, remaining)
        )),
        |(first, mut items): (Item, Vec<Item>)| {
            items.insert(0, first);
            items
        }
    )
);

named!(pub list<List>,
    alt!(
        map!(call!(alternatives), |s| List::Alternatives(s)) |
        map!(call!(sequence), |s| List::Sequence(s))
    )
);

named!(pub group<List>,
    delimited!(
        char!('('),
        call!(list),
        char!(')')
    )
);

named!(pub rule<(String, List)>,
    do_parse!(
        name: call!(symbol) >>
        delimited!(
            call!(space),
            char!('='),
            call!(space)
        ) >>
        definition: call!(list) >>
        (name, definition)
    )
);

named!(pub ruleset<Ruleset>,
    map!(
        separated_nonempty_list!(
            call!(eol),
            call!(rule)
        ),
        |rules: Vec<_>| {
            rules.into_iter().collect()
        }
    )
);

#[cfg(test)]
mod tests {
    use nom::IResult::Done;

    use parse;
    use super::*;
    use syntax::List::{Sequence, Alternatives};
    use syntax::Content::{Value, Symbol};

    #[test]
    fn test_string_literal_parser() {
        assert_eq!(parse::string_literal(&b"\"hello world\""[..]),
                   Done(&b""[..], ("hello world".to_string())));
    }

    #[test]
    fn test_number_parser() {
        assert_eq!(parse::number(&b"1"[..]), Done(&b""[..], (1u32)));
        assert_eq!(parse::number(&b"1234"[..]), Done(&b""[..], (1234u32)));
        assert_eq!(parse::number(&b"1*"[..]), Done(&b"*"[..], (1u32)));
    }

    #[test]
    fn test_repeat_parser_any() {
        assert_eq!(parse::repeat(&b"*"[..]),
                   Done(&b""[..],
                        (Repeat {
                             min: None,
                             max: None,
                         })));
    }

    #[test]
    fn test_repeat_parser_min() {
        assert_eq!(parse::repeat(&b"4*"[..]),
                   Done(&b""[..],
                        (Repeat {
                             min: Some(4),
                             max: None,
                         })));
    }

    #[test]
    fn test_repeat_parser_max() {
        assert_eq!(parse::repeat(&b"*8"[..]),
                   Done(&b""[..],
                        (Repeat {
                             min: None,
                             max: Some(8),
                         })));
    }

    #[test]
    fn test_repeat_parser_minmax() {
        assert_eq!(parse::repeat(&b"4*8"[..]),
                   Done(&b""[..],
                        (Repeat {
                             min: Some(4),
                             max: Some(8),
                         })));
    }

    #[test]
    fn test_symbol_parser() {
        assert_eq!(parse::symbol(&b"somesymbol123"[..]),
                   Done(&b""[..], ("somesymbol123".to_string())));
    }

    #[test]
    fn test_item_parser() {
        let result = Item::repeated(Symbol("abc".to_string()), Repeat::with_limits(1, 2));

        assert_eq!(parse::item(&b"1*2abc"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_sequence_parser_single() {
        let result = vec![Item::new(Symbol("sym".to_string()))];
        assert_eq!(parse::sequence(&b"sym"[..]), Done(&b""[..], (result)));

    }

    #[test]
    fn test_sequence_parser_multi() {
        let result = vec![Item::new(Symbol("sym".to_string())),
                          Item::new(Symbol("sym2".to_string()))];
        assert_eq!(parse::sequence(&b"sym sym2"[..]), Done(&b""[..], result));
    }

    #[test]
    fn test_sequence_parser_alternatives() {
        let result = vec![Item::new(Symbol("sym".to_string()))];
        assert_eq!(parse::sequence(&b"sym / sym2"[..]),
                   Done(&b" / sym2"[..], result));
    }

    #[test]
    fn test_alternatives_parser() {
        let result = vec![Item::new(Symbol("sym".to_string())),
                          Item::new(Symbol("sym2".to_string()))];
        assert_eq!(parse::alternatives(&b"sym / sym2"[..]),
                   Done(&b""[..], (result)));
    }

    #[test]
    fn test_alternatives_parser_sequence() {
        assert!(!parse::alternatives(&b"sym sym2"[..]).is_done());
    }

    #[test]
    fn test_list_parser_sequence() {
        let result = Sequence(vec![Item::new(Symbol("sym".to_string())),
                                   Item::new(Symbol("sym2".to_string()))]);
        assert_eq!(parse::list(&b"sym sym2"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_list_parser_alternatives() {
        let result = Alternatives(vec![Item::new(Symbol("sym".to_string())),
                                       Item::new(Symbol("sym2".to_string()))]);
        assert_eq!(parse::list(&b"sym / sym2"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_group_parser_single() {
        let result = Sequence(vec![Item::new(Symbol("sym".to_string()))]);
        assert_eq!(parse::group(&b"(sym)"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_group_parser_multi() {
        let result = Sequence(vec![Item::new(Symbol("sym".to_string())),
                                   Item::new(Symbol("sym2".to_string()))]);
        assert_eq!(parse::group(&b"(sym sym2)"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_rule_parser() {
        let result = ("rule".to_string(), Sequence(vec![Item::new(Symbol("def".to_string()))]));
        assert_eq!(parse::rule(&b"rule = def"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_rule_parser_sequence() {
        let result = ("rule".to_string(),
                      Sequence(vec![Item::new(Symbol("def".to_string())),
                                    Item::new(Symbol("def".to_string()))]));
        assert_eq!(parse::rule(&b"rule = def def"[..]),
                   Done(&b""[..], (result)));
    }

    #[test]
    fn test_rule_parser_alternatives() {
        let result = ("rule".to_string(),
                      Alternatives(vec![Item::new(Symbol("def".to_string())),
                                        Item::new(Symbol("def".to_string()))]));
        assert_eq!(parse::rule(&b"rule = def / def"[..]),
                   Done(&b""[..], (result)));
    }

    #[test]
    fn test_ruleset_parser_eol() {
        let make_result = || {
            vec![("def".to_string(), Sequence(vec![Item::new(Value("value".to_string()))])),
                 ("def2".to_string(), Sequence(vec![Item::new(Symbol("def".to_string()))]))]
                    .into_iter()
                    .collect::<Ruleset>()
        };

        assert_eq!(parse::ruleset(&b"def = \"value\"\ndef2 = def"[..]),
                   Done(&b""[..], (make_result())));

        assert_eq!(parse::ruleset(&b"def = \"value\"\r\ndef2 = def"[..]),
                   Done(&b""[..], (make_result())));
    }

    #[test]
    fn test_ruleset_parser() {
        let result: Ruleset =
            vec![("def".to_string(), Sequence(vec![Item::new(Value("value".to_string()))])),
                 ("rule".to_string(), Sequence(vec![Item::new(Symbol("def".to_string()))])),
                 ("rule2".to_string(),
                  Sequence(vec![Item::new(Symbol("def".to_string())),
                                Item::new(Symbol("def".to_string()))])),
                 ("rule3".to_string(),
                  Alternatives(vec![Item::new(Symbol("def".to_string())),
                                    Item::new(Symbol("def".to_string()))]))]
                    .into_iter()
                    .collect();

        let input = b"def = \"value\"\nrule = def\nrule2 = def def\nrule3 = def / def";

        assert_eq!(parse::ruleset(&input[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_hex_value_parser() {
        assert_eq!(parse::hex_value(&b"%x41"[..]), Done(&b""[..], ('A')));
        assert_eq!(parse::hex_value(&b"%x2A"[..]), Done(&b""[..], ('*')));
        assert_eq!(parse::hex_value(&b"%x2a"[..]), Done(&b""[..], ('*')));
        assert_eq!(parse::hex_value(&b"%x0A"[..]), Done(&b""[..], ('\n')));
    }

    #[test]
    fn test_range_parser() {
        assert_eq!(parse::range(&b"%x41-5A"[..]), Done(&b""[..], ('A', 'Z')));
    }
}