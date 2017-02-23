use nom::{alphanumeric, digit, eol};
use std::str;
use std::collections::HashMap;

use syntax::{Repeat, CoreRule, Item, Content};

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

named!(pub core<CoreRule>,
    alt!(
        map!(tag!("ALPHA"), |_| CoreRule::Alpha)
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
        map!(call!(group), |g| Content::Group(g)) |
        map!(call!(core), |c| Content::Core(c)) |
        map!(call!(symbol), |s| Content::Symbol(s)) |
        map!(call!(string_literal), |s| Content::Value(s)) |
        map!(call!(alternatives), |a| Content::Alternatives(a))
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

named!(pub alternatives_content<Content>,
    alt!(
        map!(call!(group), |g| Content::Group(g)) |
        map!(call!(core), |c| Content::Core(c)) |
        map!(call!(symbol), |s| Content::Symbol(s)) |
        map!(call!(string_literal), |s| Content::Value(s))
    )
);

named!(pub alternatives_item<Item>,
    map!(
        do_parse!(
            r: opt!(repeat) >>
            c: call!(alternatives_content) >>
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

named!(pub sequence<Vec<Item>>,
    fold_many0!(
        ws!(item),
        Vec::new(),
        |mut acc: Vec<_>, item| {
            acc.push(item);
            acc
        }
    )
);

named!(pub group<Vec<Item>>,
    delimited!(
        char!('('),
        call!(sequence),
        char!(')')
    )
);

named!(pub alternatives<Vec<Item>>,
    separated_nonempty_list!(
        ws!(char!('/')),
        call!(alternatives_item)
    )
);

named!(pub rule<(String, Item)>,
    do_parse!(
        s: call!(symbol) >>
        ws!(char!('=')) >>
        d: call!(item) >>
        (s, d)
    )
);

named!(pub abnf<HashMap<String, Item>>,
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

    #[test]
    fn test_string_literal_parser() {
        assert_eq!(
            parse::string_literal(&b"\"hello world\""[..]),
            Done(&b""[..], ("hello world".to_string()))
        );
    }

    #[test]
    fn test_number_parser() {
        assert_eq!(
            parse::number(&b"1"[..]),
            Done(&b""[..], (1u32))
        );

        assert_eq!(
            parse::number(&b"1234"[..]),
            Done(&b""[..], (1234u32))
        );

        assert_eq!(
            parse::number(&b"1*"[..]),
            Done(&b"*"[..], (1u32))
        );
    }

    #[test]
    fn test_repeat_parser_any() {
        assert_eq!(
            parse::repeat(&b"*"[..]),
            Done(&b""[..], (Repeat{ min: None, max: None }))
        );
    }

    #[test]
    fn test_repeat_parser_min() {
        assert_eq!(
            parse::repeat(&b"4*"[..]),
            Done(&b""[..], (Repeat{ min: Some(4), max: None }))
        );
    }

    #[test]
    fn test_repeat_parser_max() {
        assert_eq!(
            parse::repeat(&b"*8"[..]),
            Done(&b""[..], (Repeat{ min: None, max: Some(8) }))
        );
    }

    #[test]
    fn test_repeat_parser_minmax() {
        assert_eq!(
            parse::repeat(&b"4*8"[..]),
            Done(&b""[..], (Repeat{ min: Some(4), max: Some(8) }))
        );
    }

    #[test]
    fn test_core_parser() {
        assert_eq!(
            parse::core(&b"ALPHA"[..]),
            Done(&b""[..], (CoreRule::Alpha))
        );
    }

    #[test]
    fn test_symbol_parser() {
        assert_eq!(
            parse::symbol(&b"somesymbol123"[..]),
            Done(&b""[..], ("somesymbol123".to_string()))
        );
    }

    #[test]
    fn test_item_parser() {
        let result = Item::repeated(
            Content::Symbol("abc".to_string()),
            Repeat::with_limits(1, 2)
        );

        assert_eq!(
            parse::item(&b"1*2abc"[..]),
            Done(&b""[..], (result))
        );
    }

    #[test]
    fn test_sequence_parser_single() {
        let result = vec![
            Item::new(Content::Symbol("sym".to_string())),
        ];
        assert_eq!(
            parse::sequence(&b"sym"[..]),
            Done(&b""[..], (result))
        );

    }

    #[test]
    fn test_sequence_parser_multi() {
        let result = vec![
            Item::new(Content::Symbol("sym".to_string())),
            Item::new(Content::Symbol("sym2".to_string())),
        ];
        assert_eq!(
            parse::sequence(&b"sym sym2"[..]),
            Done(&b""[..], result)
        );
    }

    #[test]
    fn test_group_parser_single() {
        let result = vec![
            Item::new(Content::Symbol("sym".to_string())),
        ];
        assert_eq!(
            parse::group(&b"(sym)"[..]),
            Done(&b""[..], (result))
        );
    }

    #[test]
    fn test_group_parser_multi() {
        let result = vec![
            Item::new(Content::Symbol("sym".to_string())),
            Item::new(Content::Symbol("sym2".to_string())),
        ];
        assert_eq!(
            parse::group(&b"(sym sym2)"[..]),
            Done(&b""[..], (result))
        );
    }

    #[test]
    fn test_alternatives_parser() {
        let result = vec![
            Item::new(Content::Symbol("sym".to_string())),
            Item::new(Content::Symbol("sym2".to_string())),
        ];
        assert_eq!(
            parse::alternatives(&b"sym / sym2"[..]),
            Done(&b""[..], (result))
        );
    }

    #[test]
    fn test_rule_parser() {
        let result = (
            "rule".to_string(),
            Item::new(Content::Symbol("def".to_string())),
        );
        assert_eq!(
            parse::rule(&b"rule = def"[..]),
            Done(&b""[..], (result))
        );
    }

    #[test]
    fn test_abnf_parser() {
        let make_result = || {
            vec![
                (
                    "def".to_string(),
                    Item::new(Content::Value("value".to_string())),
                ),
                (
                    "rule".to_string(),
                    Item::new(Content::Symbol("def".to_string())),
                ),
                (
                    "rule2".to_string(),
                    Item::new(Content::Symbol("def".to_string())),
                )
            ].into_iter().collect::<HashMap<_, _>>()
        };

        assert_eq!(
            parse::abnf(&b"def = \"value\"\nrule = def\nrule2 = def"[..]),
            Done(&b""[..], (make_result()))
        );
        assert_eq!(
            parse::abnf(&b"def = \"value\"\r\nrule = def\r\nrule2 = def"[..]),
            Done(&b""[..], (make_result()))
        );
    }
}
