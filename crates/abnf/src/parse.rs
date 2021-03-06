use nom::{alphanumeric, digit, eol, space, hex_u32};
use std::str;

use syntax::{Content, Item, List, Repeat, Rule, Sequence, Symbol};

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

named!(pub symbol<Symbol>,
    map!(
        map_res!(
            call!(alphanumeric),
            str::from_utf8
        ),
        |name| Symbol::from(name)
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
    map!(
        complete!(do_parse!(
            first: call!(item) >>
            remaining: many0!(
                preceded!(
                    call!(space),
                    complete!(call!(item))
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
                complete!(preceded!(
                    call!(alternatives_sep),
                    call!(item)
                ))
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

named!(pub rule<Rule>,
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

named!(pub rules<Vec<Rule>>,
    map!(
        do_parse!(
            first: call!(rule) >>
            remaining: many0!(
                preceded!(
                    call!(eol),
                    complete!(call!(rule))
                )
            ) >>
            (first, remaining)
        ),
        |(first, mut remaining): (Rule, Vec<Rule>)| {
            remaining.insert(0, first);
            remaining
        }
    )
);

#[cfg(test)]
mod tests {
    use nom::IResult::Done;

    use parse;
    use super::*;
    use syntax::List::{Alternatives, Sequence};
    use syntax::{Content, Symbol};

    #[test]
    fn test_string_literal_parser() {
        assert_eq!(
            parse::string_literal(&b"\"hello world\""[..]),
            Done(&b""[..], ("hello world".to_string()))
        );
    }

    #[test]
    fn test_number_parser() {
        assert_eq!(parse::number(&b"1"[..]), Done(&b""[..], (1u32)));
        assert_eq!(parse::number(&b"1234"[..]), Done(&b""[..], (1234u32)));
        assert_eq!(parse::number(&b"1*"[..]), Done(&b"*"[..], (1u32)));
    }

    #[test]
    fn test_repeat_parser_any() {
        assert_eq!(
            parse::repeat(&b"*"[..]),
            Done(
                &b""[..],
                (Repeat {
                    min: None,
                    max: None,
                }),
            )
        );
    }

    #[test]
    fn test_repeat_parser_min() {
        assert_eq!(
            parse::repeat(&b"4*"[..]),
            Done(
                &b""[..],
                (Repeat {
                    min: Some(4),
                    max: None,
                }),
            )
        );
    }

    #[test]
    fn test_repeat_parser_max() {
        assert_eq!(
            parse::repeat(&b"*8"[..]),
            Done(
                &b""[..],
                (Repeat {
                    min: None,
                    max: Some(8),
                }),
            )
        );
    }

    #[test]
    fn test_repeat_parser_minmax() {
        assert_eq!(
            parse::repeat(&b"4*8"[..]),
            Done(
                &b""[..],
                (Repeat {
                    min: Some(4),
                    max: Some(8),
                }),
            )
        );
    }

    #[test]
    fn test_symbol_parser() {
        assert_eq!(
            parse::symbol(&b"somesymbol123"[..]),
            Done(&b""[..], (Symbol::from("somesymbol123")))
        );
    }

    #[test]
    fn test_item_parser() {
        let result = Item::repeated(
            Content::Symbol(Symbol::from("abc")),
            Repeat::with_limits(1, 2),
        );

        assert_eq!(parse::item(&b"1*2abc"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_sequence_parser_single() {
        let result = vec![Item::new(Content::Symbol(Symbol::from("sym")))];
        assert_eq!(parse::sequence(&b"sym"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_sequence_parser_multi() {
        let result = vec![
            Item::new(Content::Symbol(Symbol::from("sym"))),
            Item::new(Content::Symbol(Symbol::from("sym2"))),
        ];
        assert_eq!(parse::sequence(&b"sym sym2"[..]), Done(&b""[..], result));
    }

    #[test]
    fn test_sequence_parser_trailing() {
        let result = vec![Item::new(Content::Symbol(Symbol::from("sym")))];
        assert_eq!(parse::sequence(&b"sym "[..]), Done(&b" "[..], (result)));
    }

    #[test]
    fn test_sequence_parser_alternatives() {
        let result = vec![Item::new(Content::Symbol(Symbol::from("sym")))];
        assert_eq!(
            parse::sequence(&b"sym / sym2"[..]),
            Done(&b" / sym2"[..], result)
        );
    }

    #[test]
    fn test_alternatives_parser() {
        let result = vec![
            Item::new(Content::Symbol(Symbol::from("sym"))),
            Item::new(Content::Symbol(Symbol::from("sym2"))),
        ];
        assert_eq!(
            parse::alternatives(&b"sym / sym2"[..]),
            Done(&b""[..], (result))
        );
    }

    #[test]
    fn test_alternatives_parser_trailing() {
        let result = vec![
            Item::new(Content::Symbol(Symbol::from("sym"))),
            Item::new(Content::Symbol(Symbol::from("sym2"))),
        ];
        assert_eq!(
            parse::alternatives(&b"sym / sym2 "[..]),
            Done(&b" "[..], (result))
        );
    }

    #[test]
    fn test_alternatives_parser_sequence() {
        assert!(!parse::alternatives(&b"sym sym2"[..]).is_done());
    }

    #[test]
    fn test_list_parser_sequence() {
        let result = Sequence(vec![
            Item::new(Content::Symbol(Symbol::from("sym"))),
            Item::new(Content::Symbol(Symbol::from("sym2"))),
        ]);
        assert_eq!(parse::list(&b"sym sym2"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_list_parser_alternatives() {
        let result = Alternatives(vec![
            Item::new(Content::Symbol(Symbol::from("sym"))),
            Item::new(Content::Symbol(Symbol::from("sym2"))),
        ]);
        assert_eq!(parse::list(&b"sym / sym2"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_group_parser_single() {
        let result = Sequence(vec![Item::new(Content::Symbol(Symbol::from("sym")))]);
        assert_eq!(parse::group(&b"(sym)"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_group_parser_multi() {
        let result = Sequence(vec![
            Item::new(Content::Symbol(Symbol::from("sym"))),
            Item::new(Content::Symbol(Symbol::from("sym2"))),
        ]);
        assert_eq!(parse::group(&b"(sym sym2)"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_rule_parser() {
        let result = (
            Symbol::from("rule"),
            Sequence(vec![Item::new(Content::Symbol(Symbol::from("def")))]),
        );
        assert_eq!(parse::rule(&b"rule = def"[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_rule_parser_sequence() {
        let result = (
            Symbol::from("rule"),
            Sequence(vec![
                Item::new(Content::Symbol(Symbol::from("def"))),
                Item::new(Content::Symbol(Symbol::from("def"))),
            ]),
        );
        assert_eq!(
            parse::rule(&b"rule = def def"[..]),
            Done(&b""[..], (result))
        );
    }

    #[test]
    fn test_rule_parser_alternatives() {
        let result = (
            Symbol::from("rule"),
            Alternatives(vec![
                Item::new(Content::Symbol(Symbol::from("def"))),
                Item::new(Content::Symbol(Symbol::from("def"))),
            ]),
        );
        assert_eq!(
            parse::rule(&b"rule = def / def"[..]),
            Done(&b""[..], (result))
        );
    }

    #[test]
    fn test_rules_parser_eol() {
        let make_result = || {
            vec![
                (
                    Symbol::from("def"),
                    Sequence(vec![Item::new(Content::Value("value".to_string()))]),
                ),
                (
                    Symbol::from("def2"),
                    Sequence(vec![Item::new(Content::Symbol(Symbol::from("def")))]),
                ),
            ]
        };

        assert_eq!(
            parse::rules(&b"def = \"value\"\ndef2 = def"[..]),
            Done(&b""[..], (make_result()))
        );

        assert_eq!(
            parse::rules(&b"def = \"value\"\r\ndef2 = def"[..]),
            Done(&b""[..], (make_result()))
        );
    }

    #[test]
    fn test_rules_parser() {
        let result = vec![
            (
                Symbol::from("def"),
                Sequence(vec![Item::new(Content::Value("value".to_string()))]),
            ),
            (
                Symbol::from("rule"),
                Sequence(vec![Item::new(Content::Symbol(Symbol::from("def")))]),
            ),
            (
                Symbol::from("rule2"),
                Sequence(vec![
                    Item::new(Content::Symbol(Symbol::from("def"))),
                    Item::new(Content::Symbol(Symbol::from("def"))),
                ]),
            ),
            (
                Symbol::from("rule3"),
                Alternatives(vec![
                    Item::new(Content::Symbol(Symbol::from("def"))),
                    Item::new(Content::Symbol(Symbol::from("def"))),
                ]),
            ),
        ];

        let input = b"def = \"value\"\nrule = def\nrule2 = def def\nrule3 = def / def";

        assert_eq!(parse::rules(&input[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_rules_parser_single() {
        let result = vec![
            (
                Symbol::from("rule"),
                Sequence(vec![Item::new(Content::Symbol(Symbol::from("def")))]),
            ),
        ];
        let input = b"rule = def";
        assert_eq!(parse::rules(&input[..]), Done(&b""[..], (result)));
    }

    #[test]
    fn test_rules_parser_trailing() {
        let result = vec![
            (
                Symbol::from("rule"),
                Sequence(vec![Item::new(Content::Symbol(Symbol::from("def")))]),
            ),
        ];

        let input = b"rule = def\n";
        assert_eq!(parse::rules(&input[..]), Done(&b"\n"[..], (result.clone())));

        let input = b"rule = def ";
        assert_eq!(parse::rules(&input[..]), Done(&b" "[..], (result)));
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
