use std::{mem, ptr, fmt, slice};
use std::ops::{Index, IndexMut};

use serde::{Serialize, Serializer};
use serde::ser::{SerializeMap};

use common::Instruction;
use common::CommandMap;
use common::map_word_to_instructions;
use common::MAX_ALPHABET_SIZE;
use common::Rewriter;

pub struct RuleMap {
    map: [String; MAX_ALPHABET_SIZE],
}

impl RuleMap {
    pub fn new() -> RuleMap {
        let mut rules: [String; MAX_ALPHABET_SIZE] = unsafe { mem::uninitialized() };

        for (i, v) in rules.iter_mut().enumerate() {
            let mut rule = String::with_capacity(1);
            rule.push(i as u8 as char);
            unsafe { ptr::write(v, rule); }
        }

        RuleMap {
            map: rules,
        }
    }

    pub fn iter<'a>(&'a self) -> slice::Iter<'a, String> {
        self.map.iter()
    }

    pub fn iter_mut<'a>(&'a mut self) -> slice::IterMut<'a, String> {
        self.map.iter_mut()
    }
}

impl Index<u8> for RuleMap {
    type Output = String;

    fn index<'a>(&'a self, index: u8) -> &'a String {
        &self.map[index as usize]
    }
}

impl Index<char> for RuleMap {
    type Output = String;

    fn index<'a>(&'a self, index: char) -> &'a String {
        &self.map[index as usize]
    }
}

impl IndexMut<u8> for RuleMap {
    fn index_mut<'a>(&'a mut self, index: u8) -> &'a mut String {
        &mut self.map[index as usize]
    }
}

impl IndexMut<char> for RuleMap {
    fn index_mut<'a>(&'a mut self, index: char) -> &'a mut String {
        &mut self.map[index as usize]
    }
}

impl fmt::Display for RuleMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (pred, succ) in self.map.iter().enumerate() {
            if !(succ.len() == 1 && succ.as_bytes()[0] == pred as u8) {
                write!(f, "{} -> {}\n", pred as u8 as char, succ)?;
            }
        }

        Ok(())
    }
}

impl Serialize for RuleMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;

        for (pred, succ) in self.map.iter().enumerate() {
            if !(succ.len() == 1 && succ.as_bytes()[0] == pred as u8) {
                map.serialize_entry(&(pred as u8 as char), succ)?;
            }
        }

        map.end()
    }
}

fn expand_lsystem(axiom: &str, rules: &RuleMap, iterations: u32) -> String {
    let mut lword = String::from(axiom);

    for _ in 0..iterations {
        let mut expanded_lword = String::with_capacity(lword.len());
        for lchar in lword.bytes() {
            let expanded_lchar = &rules[lchar];
            expanded_lword.push_str(&mut expanded_lchar.clone());
        }
        lword = expanded_lword;
    }

    lword
}

fn remove_redundancy(from: &str) -> String {
    let mut trimmed = from.to_string();
    let mut prev_len = 0;

    while trimmed.len() != prev_len {
        prev_len = trimmed.len();
        trimmed = trimmed.replace("[]", "");
    }

    trimmed
}

#[derive(Serialize)]
pub struct LSystem {
    pub productions: RuleMap,
    pub axiom: String,
}

impl LSystem {
    pub fn new() -> LSystem {
        LSystem {
            productions: RuleMap::new(),
            axiom: String::new(),
        }
    }

    pub fn set_rule(&mut self, letter: char, expansion: &str) {
        self.productions[letter] = String::from(expansion);
    }

    pub fn remove_redundancy(&mut self) {
        self.axiom = remove_redundancy(&self.axiom);

        for successor in self.productions.iter_mut() {
            *successor = remove_redundancy(&successor);
        }
    }

    pub fn rewrite(&self, iterations: u32) -> String {
        expand_lsystem(&self.axiom, &self.productions, iterations)
    }
}

impl Rewriter for LSystem {
    fn instructions(&self, iterations: u32, command_map: &CommandMap) -> Vec<Instruction> {
        let lword = expand_lsystem(&self.axiom, &self.productions, iterations);
        map_word_to_instructions(&lword, &command_map)
    }
}

impl fmt::Display for LSystem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "w: ")?;
        for letter in self.axiom.as_bytes() {
            write!(f, "{}", *letter as char)?;
        }

        write!(f, "\n{}", self.productions)?;

        Ok(())
    }
}
