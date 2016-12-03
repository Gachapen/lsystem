use std::rc::Rc;
use std::fmt;

use common::Command;
use common::CommandMap;
use common::Instruction;
use common::create_command_map;

#[derive(Copy, Clone, Debug)]
pub enum Param {
    Float(f32),
    Int(i32),
}

impl Param {
    pub fn is_float(self) -> bool {
        match self {
            Param::Float(_) => true,
            Param::Int(_) => false,
        }
    }

    pub fn is_int(self) -> bool {
        match self {
            Param::Float(_) => false,
            Param::Int(_) => true,
        }
    }

    pub fn float(self) -> Option<f32> {
        match self {
            Param::Float(x) => Some(x),
            Param::Int(_) => None,
        }
    }

    pub fn int(self) -> Option<i32> {
        match self {
            Param::Float(_) => None,
            Param::Int(x) => Some(x),
        }
    }
}

pub struct Letter {
    character: u8,
    params: Vec<Param>,
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

impl fmt::Debug for Letter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Letter {{ character: {}, params: {:?} }}", self.character as char, self.params)
    }
}

pub trait WordFromString {
    fn from_str(string: &str) -> Word;
}

pub type Word = Vec<Letter>;

impl WordFromString for Word {
    fn from_str(string: &str) -> Word {
        let mut word = Word::with_capacity(string.len());
        for character in string.as_bytes() {
            word.push(Letter::new(*character as char));
        }
        word
    }
}

type Condition = Fn(&Vec<Param>) -> bool;

type ParamTransform = Fn(&Vec<Param>) -> Vec<Param>;

pub struct ProductionLetter {
    character: u8,
    transformer: Rc<ParamTransform>,
}

pub mod transform {
    use super::Param;

    pub fn identity(params: &Vec<Param>) -> Vec<Param> {
        params.clone()
    }

    pub fn empty(_: &Vec<Param>) -> Vec<Param> {
        vec![]
    }
}

impl ProductionLetter {
    pub fn new(character: char) -> ProductionLetter {
        ProductionLetter {
            character: character as u8,
            transformer: Rc::new(transform::empty),
        }
    }

    pub fn with_transform(character: char, transformer: Rc<ParamTransform>) -> ProductionLetter {
        ProductionLetter {
            character: character as u8,
            transformer: transformer,
        }
    }

    pub fn with_params(character: char, params: Vec<Param>) -> ProductionLetter {
        ProductionLetter {
            character: character as u8,
            transformer: Rc::new(move |_| params.clone()),
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
    condition: Rc<Condition>,
    successor: ProductionWord,
}

impl Production {
    pub fn with_condition(predecessor: char, condition: Rc<Condition>, successor: ProductionWord) -> Production {
        Production {
            predecessor: predecessor as u8,
            condition: condition,
            successor: successor,
        }
    }

    pub fn new(predecessor: char, successor: ProductionWord) -> Production {
        Production {
            predecessor: predecessor as u8,
            condition: Rc::new(|_| true),
            successor: successor,
        }
    }
}

fn expand_lsystem(axiom: &Word, productions: &Vec<Production>, iterations: u32) -> Word {
    let mut word = axiom.clone();

    for _ in 0..iterations {
        let mut expansion = Word::with_capacity(word.len());

        for letter in word {
            let prod = productions.iter().find(|&prod| {
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
                    let params = (prod_letter.transformer)(&letter.params);
                    expansion.push(Letter::with_params(prod_letter.character as char, params));
                }
            } else {
                expansion.push(letter);
            }
        }

        word = expansion;
    }

    word
}

fn params_to_args(params: &Vec<Param>) -> Vec<f32> {
    params.iter().map(|p| {
        match *p {
            Param::Int(x) => x as f32,
            Param::Float(x) => x,
        }
    }).collect()
}

pub fn map_word_to_instructions(word: &Word, command_map: &CommandMap) -> Vec<Instruction> {
    let mut instructions = Vec::with_capacity(word.len());
    for letter in word {
        let command = command_map[letter.character as usize];
        if command != Command::Noop {
            instructions.push(Instruction{ command: command, args: params_to_args(&letter.params) });
        }
    }
    instructions
}

pub struct LSystem {
    pub command_map: CommandMap,
    pub productions: Vec<Production>,
    pub axiom: Word,
}

impl LSystem {
    pub fn new() -> LSystem {
        LSystem {
            command_map: create_command_map(),
            productions: Vec::new(),
            axiom: Word::new(),
        }
    }

    pub fn instructions(&self, iterations: u32) -> Vec<Instruction> {
        let word = expand_lsystem(&self.axiom, &self.productions, iterations);
        map_word_to_instructions(&word, &self.command_map)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
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
                    Param::Int(x) => {
                        match bp {
                            Param::Float(y) => return false,
                            Param::Int(y) => {
                                if x != y {
                                    return false
                                }
                            },
                        }
                    },
                    Param::Float(x) => {
                        match bp {
                            Param::Int(y) => return false,
                            Param::Float(y) => {
                                if x != y {
                                    return false
                                }
                            },
                        }
                    },
                }
            }
        }

        true
    }

    #[test]
    fn expand_lsystem_test() {
        let axiom = vec![
            Letter::new('A'),
        ];
        let productions = vec![
            Production::new('A', ProductionWord::from_str("ABC")),
        ];

        assert!(words_eq(&Word::from_str("ABC"), &super::expand_lsystem(&axiom, &productions, 1)));
    }

    #[test]
    fn expand_lsystem_param_test() {
        let axiom = vec![
            Letter::with_params('A', vec![Param::Int(0), Param::Float(1.0)]),
        ];
        let productions = vec![
            Production::new(
                'A',
                vec![
                    ProductionLetter::with_params('A', vec![Param::Int(1), Param::Float(0.0)]),
                    ProductionLetter::new('B'),
                ]
            ),
        ];

        let expected = vec![
            Letter::with_params('A', vec![Param::Int(1), Param::Float(0.0)]),
            Letter::new('B'),
        ];

        println!("{:?}", super::expand_lsystem(&axiom, &productions, 1));
        assert!(words_eq(&expected, &super::expand_lsystem(&axiom, &productions, 1)));
    }

    #[test]
    fn expand_lsystem_condition_test() {
        let axiom = vec![
            Letter::with_params('A', vec![Param::Int(0)]),
        ];
        let productions = vec![
            Production::with_condition(
                'A',
                Rc::new(|params| params[0].int().unwrap() == 0),
                vec![
                    ProductionLetter::with_params('A', vec![Param::Int(0)]),
                    ProductionLetter::with_params('B', vec![Param::Int(0)]),
                ]
            ),
            Production::with_condition(
                'B',
                Rc::new(|params| params[0].int().unwrap() == 1),
                vec![
                    ProductionLetter::new('C'),
                ]
            ),
        ];

        let expected = vec![
            Letter::with_params('A', vec![Param::Int(0)]),
            Letter::with_params('B', vec![Param::Int(0)]),
            Letter::with_params('B', vec![Param::Int(0)]),
        ];

        assert!(words_eq(&expected, &super::expand_lsystem(&axiom, &productions, 2)));
    }

    #[test]
    fn expand_lsystem_transform_test() {
        let axiom = vec![
            Letter::with_params('A', vec![Param::Int(0)]),
        ];
        let productions = vec![
            Production::new(
                'A',
                vec![
                    ProductionLetter::with_transform('A', Rc::new(|p| vec![Param::Int(p[0].int().unwrap() + 1)])),
                ]
            ),
        ];

        let expected = vec![
            Letter::with_params('A', vec![Param::Int(4)]),
        ];

        assert!(words_eq(&expected, &super::expand_lsystem(&axiom, &productions, 4)));
    }
}
