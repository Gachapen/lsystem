use std::mem;
use std::ptr;
use std::fmt;

use common::Command;
use common::Instruction;
use common::CommandMap;
use common::create_command_map;
use common::map_word_to_instructions;
use common::MAX_ALPHABET_SIZE;
use common::Rewriter;

pub type RuleMap = [String; MAX_ALPHABET_SIZE];

pub fn create_rule_map() -> RuleMap {
    let mut rules: RuleMap = unsafe { mem::uninitialized() };

    for (i, v) in rules.iter_mut().enumerate() {
        let mut rule = String::with_capacity(1);
        rule.push(i as u8 as char);
        unsafe { ptr::write(v, rule); }
    }

    rules
}

fn expand_lsystem(axiom: &str, rules: &RuleMap, iterations: u32) -> String {
    let mut lword = String::from(axiom);

    for _ in 0..iterations {
        let mut expanded_lword = String::with_capacity(lword.len());
        for lchar in lword.bytes() {
            let expanded_lchar = &rules[lchar as usize];
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

pub struct LSystem {
    pub command_map: CommandMap,
    pub rules: RuleMap,
    pub axiom: String,
}

impl LSystem {
    pub fn new() -> LSystem {
        LSystem {
            command_map: create_command_map(),
            rules: create_rule_map(),
            axiom: String::new(),
        }
    }

    pub fn map_command(&mut self, letter: char, command: Command) {
        self.command_map[letter as u8 as usize] = command;
    }

    pub fn set_rule(&mut self, letter: char, expansion: &str) {
        self.rules[letter as usize] = String::from(expansion);
    }

    pub fn remove_redundancy(&mut self) {
        self.axiom = remove_redundancy(&self.axiom);

        for successor in self.rules.iter_mut() {
            *successor = remove_redundancy(&successor);
        }
    }

    pub fn rewrite(&self, iterations: u32) -> String {
        expand_lsystem(&self.axiom, &self.rules, iterations)
    }
}

impl Rewriter for LSystem {
    fn instructions(&self, iterations: u32) -> Vec<Instruction> {
        let lword = expand_lsystem(&self.axiom, &self.rules, iterations);
        map_word_to_instructions(&lword, &self.command_map)
    }
}

impl fmt::Display for LSystem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "w: ")?;
        for letter in self.axiom.as_bytes() {
            write!(f, "{}", *letter as char)?;
        }

        for (pred, succ) in self.rules.iter().enumerate() {
            if !(succ.len() == 1 && succ.as_bytes()[0] == pred as u8) {
                write!(f, "\n{} -> {}", pred as u8 as char, succ)?;
            }
        }

        Ok(())
    }
}
