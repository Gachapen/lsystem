use std::fmt;

use common::Command;
use common::CommandMap;
use common::Instruction;
use common::Rewriter;

#[derive(Copy, Clone, Debug)]
pub enum Param {
    F(f32),
    I(i32),
}

#[macro_export]
macro_rules! params_f {
    ( $( $x:expr ),* ) => {
        {
            vec![$(Param::F($x)),*]
        }
    };
}

#[macro_export]
macro_rules! params_i {
    ( $( $x:expr ),* ) => {
        {
            vec![$(Param::I($x)),*]
        }
    };
}

impl Param {
    pub fn f(self) -> f32 {
        match self {
            Param::F(x) => x,
            Param::I(_) => panic!("Tried converting Param with int into float"),
        }
    }

    pub fn i(self) -> i32 {
        match self {
            Param::F(_) => panic!("Tried converting Param with float into int"),
            Param::I(x) => x,
        }
    }
}

impl fmt::Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Param::F(v) => write!(f, "{}", v),
            Param::I(v) => write!(f, "{}", v),
        }
    }
}

#[derive(Debug)]
pub struct Letter {
    pub character: u8,
    pub params: Vec<Param>,
}

impl Letter {
    pub fn new(character: char) -> Letter {
        Letter {
            character: character as u8,
            params: vec![],
        }
    }

    pub fn with_params(character: char, params: Vec<Param>) -> Letter {
        Letter {
            character: character as u8,
            params: params,
        }
    }
}

impl Clone for Letter {
    fn clone(&self) -> Letter {
        Letter {
            character: self.character,
            params: self.params.clone(),
        }
    }
}

impl fmt::Display for Letter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}(", self.character as char)?;
        for param in &self.params {
            write!(f, "{}", param)?;
        }
        write!(f, ")")
    }
}

pub type Word = Vec<Letter>;
pub type WordSlice = [Letter];

pub trait WordFromString {
    fn from_str(string: &str) -> Word;
}

impl WordFromString for Word {
    fn from_str(string: &str) -> Word {
        let mut word = Word::with_capacity(string.len());
        let bytes = string.as_bytes();

        let mut iter = bytes.iter().cloned().peekable();
        while let Some(byte) = iter.next() {
            let mut params = vec![];

            if let Some(next) = iter.clone().peek() {
                if *next == '(' as u8 {
                    iter.next();

                    let param_list_end =
                        iter.clone()
                            .position(|b| b == ')' as u8)
                            .expect("Syntax error: Parameter has no end paranthesis ')'");
                    // Iterator for parameter, including ending ')', e.g. "A(50,3)" => "50,3)"
                    let mut param_list_iter = iter.clone().take(param_list_end + 1);

                    while let Some(param_end) =
                        param_list_iter
                            .clone()
                            .position(|b| b == ',' as u8 || b == ')' as u8) {
                        let param_iter = param_list_iter.clone().take(param_end);

                        let mut param_str = String::new();
                        for param_byte in param_iter {
                            param_str.push(param_byte as char);
                        }

                        if let Ok(param) = param_str.parse::<i32>() {
                            params.push(Param::I(param));
                        } else if let Ok(param) = param_str.parse::<f32>() {
                            params.push(Param::F(param));
                        } else {
                            panic!("Failed parsing parameter '{}' as int or float", param_str);
                        }

                        param_list_iter.nth(param_end);
                    }

                    iter.nth(param_list_end);
                }
            }

            let letter = byte as char;
            if !params.is_empty() {
                word.push(Letter::with_params(letter, params));
            } else {
                word.push(Letter::new(letter));
            }
        }

        word
    }
}

type Condition = Fn(&Vec<Param>) -> bool;

type ParamTransform = Fn(&[Param], f32) -> Vec<Param>;

pub struct ProductionLetter {
    character: u8,
    transformer: Box<ParamTransform>,
}

pub mod transform {
    use super::Param;

    pub fn identity(params: &[Param], _: f32) -> Vec<Param> {
        params.to_vec()
    }

    pub fn empty(_: &[Param], _: f32) -> Vec<Param> {
        vec![]
    }
}

impl ProductionLetter {
    pub fn new(character: char) -> ProductionLetter {
        ProductionLetter::with_transform(character, transform::empty)
    }

    pub fn with_transform<F>(character: char, transformer: F) -> ProductionLetter
        where F: Fn(&[Param], f32) -> Vec<Param> + 'static
    {

        ProductionLetter {
            character: character as u8,
            transformer: Box::new(transformer),
        }
    }

    pub fn with_params(character: char, params: Vec<Param>) -> ProductionLetter {
        ProductionLetter::with_transform(character, move |_, _| params.clone())
    }
}

/// Convert from `Letter` to `ProductionLetter`, changing params to transforms that
/// return the params.
impl From<Letter> for ProductionLetter {
    fn from(letter: Letter) -> ProductionLetter {
        ProductionLetter::with_params(letter.character as char, letter.params.clone())
    }
}

/// Convert from `&Letter` to `ProductionLetter`, changing params to transforms that
/// return the params.
impl<'a> From<&'a Letter> for ProductionLetter {
    fn from(letter: &Letter) -> ProductionLetter {
        ProductionLetter::with_params(letter.character as char, letter.params.clone())
    }
}

impl fmt::Display for ProductionLetter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.character as char)?;
        if !(self.character == '[' as u8 || self.character == ']' as u8) {
            write!(f, "(.)")
        } else {
            Ok(())
        }
    }
}

pub type ProductionWord = Vec<ProductionLetter>;

pub trait ProductionWordFromString {
    fn from_str(string: &str) -> ProductionWord;
}

impl ProductionWordFromString for ProductionWord {
    fn from_str(string: &str) -> ProductionWord {
        let mut word = ProductionWord::with_capacity(string.len());
        for character in string.as_bytes() {
            word.push(ProductionLetter::new(*character as char));
        }
        word
    }
}

pub struct Production {
    predecessor: u8,
    condition: Box<Condition>,
    successor: ProductionWord,
}

impl Production {
    pub fn with_condition<F>(predecessor: char,
                             condition: F,
                             successor: ProductionWord)
                             -> Production
        where F: Fn(&Vec<Param>) -> bool + 'static
    {

        Production {
            predecessor: predecessor as u8,
            condition: Box::new(condition),
            successor: successor,
        }
    }

    pub fn new(predecessor: char, successor: ProductionWord) -> Production {
        Production::with_condition(predecessor, |_| true, successor)
    }
}

impl fmt::Display for Production {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let prod_str = self.successor
            .iter()
            .fold(String::new(), |mut word, l| {
                word.push_str(&format!("{}", l));
                word
            });
        write!(f, "{} -> {}", self.predecessor as char, prod_str)
    }
}

fn expand_lsystem(axiom: &WordSlice, productions: &[Production], iterations: u32) -> Word {
    let mut word = axiom.to_vec();

    for _ in 0..iterations {
        word = step(&word, productions, 1.0);
    }

    word
}

fn params_to_args(params: &[Param]) -> Vec<f32> {
    params
        .iter()
        .map(|p| match *p {
                 Param::I(x) => x as f32,
                 Param::F(x) => x,
             })
        .collect()
}

pub fn map_word_to_instructions(word: &WordSlice, command_map: &CommandMap) -> Vec<Instruction> {
    let mut instructions = Vec::with_capacity(word.len());
    for letter in word {
        let command = command_map[letter.character];
        if command != Command::Noop {
            instructions.push(Instruction {
                                  command: command,
                                  args: Some(params_to_args(&letter.params)),
                              });
        }
    }
    instructions
}

pub fn step(prev: &WordSlice, productions: &[Production], dt: f32) -> Word {
    let mut expansion = Word::with_capacity(prev.len());

    for letter in prev {
        let prod = productions
            .iter()
            .find(|&prod| {
                if prod.predecessor != letter.character {
                    return false;
                }

                if !(prod.condition)(&letter.params) {
                    return false;
                }

                true
            });

        if let Some(prod) = prod {
            for prod_letter in &prod.successor {
                let params = (prod_letter.transformer)(&letter.params, dt);
                expansion.push(Letter::with_params(prod_letter.character as char, params));
            }
        } else {
            expansion.push(letter.clone());
        }
    }

    expansion
}

pub struct LSystem {
    pub productions: Vec<Production>,
    pub axiom: Word,
}

impl LSystem {
    pub fn new() -> LSystem {
        Default::default()
    }
}

impl Default for LSystem {
    fn default() -> LSystem {
        LSystem {
            productions: Vec::new(),
            axiom: Word::new(),
        }
    }
}

impl Rewriter for LSystem {
    fn instructions(&self, iterations: u32, command_map: &CommandMap) -> Vec<Instruction> {
        let word = expand_lsystem(&self.axiom, &self.productions, iterations);
        map_word_to_instructions(&word, command_map)
    }
}

impl fmt::Display for LSystem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "w: ")?;
        for letter in &self.axiom {
            write!(f, "{}", letter)?;
        }

        for prod in &self.productions {
            write!(f, "\n{}", prod)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words_eq(a: &Word, b: &Word) -> bool {
        if a.len() != b.len() {
            return false;
        }

        for i in 0..a.len() {
            let al = &a[i];
            let bl = &b[i];

            if al.character != bl.character {
                return false;
            }

            if al.params.len() != bl.params.len() {
                return false;
            }

            for j in 0..al.params.len() {
                let ap = al.params[j];
                let bp = bl.params[j];

                match ap {
                    Param::I(x) => {
                        match bp {
                            Param::F(_) => return false,
                            Param::I(y) => {
                                if x != y {
                                    return false;
                                }
                            }
                        }
                    }
                    Param::F(x) => {
                        match bp {
                            Param::I(_) => return false,
                            Param::F(y) => {
                                if x != y {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }

        true
    }

    #[test]
    fn expand_lsystem_test() {
        let axiom = vec![Letter::new('A')];
        let productions = vec![Production::new('A', ProductionWord::from_str("ABC"))];

        assert!(words_eq(&Word::from_str("ABC"),
                         &super::expand_lsystem(&axiom, &productions, 1)));
    }

    #[test]
    fn expand_lsystem_param_test() {
        let axiom = vec![Letter::with_params('A', vec![Param::I(0), Param::F(1.0)])];
        let productions = vec![Production::new('A',
                                               vec![
                    ProductionLetter::with_params('A', vec![Param::I(1), Param::F(0.0)]),
                    ProductionLetter::new('B'),
                ])];

        let expected = vec![Letter::with_params('A', vec![Param::I(1), Param::F(0.0)]),
                            Letter::new('B')];

        println!("{:?}", super::expand_lsystem(&axiom, &productions, 1));
        assert!(words_eq(&expected, &super::expand_lsystem(&axiom, &productions, 1)));
    }

    #[test]
    fn expand_lsystem_condition_test() {
        let axiom = vec![Letter::with_params('A', vec![Param::I(0)])];
        let productions = vec![Production::with_condition('A',
                                                          |params| params[0].i() == 0,
                                                          vec![
                    ProductionLetter::with_params('A', vec![Param::I(0)]),
                    ProductionLetter::with_params('B', vec![Param::I(0)]),
                ]),
                               Production::with_condition('B',
                                                          |params| params[0].i() == 1,
                                                          vec![ProductionLetter::new('C')])];

        let expected = vec![Letter::with_params('A', vec![Param::I(0)]),
                            Letter::with_params('B', vec![Param::I(0)]),
                            Letter::with_params('B', vec![Param::I(0)])];

        assert!(words_eq(&expected, &super::expand_lsystem(&axiom, &productions, 2)));
    }

    #[test]
    fn expand_lsystem_transform_test() {
        let axiom = vec![Letter::with_params('A', vec![Param::I(0)])];
        let productions = vec![Production::new('A',
                                               vec![ProductionLetter::with_transform('A',
                                                     |p, _| {
                                                         vec![Param::I(p[0].i() + 1)]
                                                     })])];

        let expected = vec![Letter::with_params('A', vec![Param::I(4)])];

        assert!(words_eq(&expected, &super::expand_lsystem(&axiom, &productions, 4)));
    }

    #[test]
    fn word_from_str_test_no_param() {
        let expected = vec![Letter::new('A'), Letter::new('B'), Letter::new('C')];
        assert!(words_eq(&expected, &Word::from_str("ABC")));
    }

    #[test]
    fn word_from_str_test_single_param() {
        let expected = vec![Letter::with_params('A', params_i![1]),
                            Letter::new('B'),
                            Letter::with_params('C', params_f![2.0])];
        let actual = Word::from_str("A(1)BC(2.0)");

        println!("Expected: {:?}", expected);
        println!("Actual: {:?}", actual);

        assert!(words_eq(&expected, &actual));
    }

    #[test]
    fn word_from_str_test_multi_param() {
        let expected = vec![Letter::with_params('A', params_i![1, 2]),
                            Letter::new('B'),
                            Letter::with_params('C',
                                                vec![Param::F(3.0), Param::I(4), Param::I(5)])];
        let actual = Word::from_str("A(1,2)BC(3.0,4,5)");

        println!("Expected: {:?}", expected);
        println!("Actual: {:?}", actual);

        assert!(words_eq(&expected, &actual));
    }
}
