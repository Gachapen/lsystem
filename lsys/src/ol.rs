use std::mem;
use std::ptr;

use common::Command;
use common::Instruction;
use common::CommandMap;
use common::create_command_map;
use common::map_word_to_instructions;
use common::MAX_ALPHABET_SIZE;

type RuleMap = [String; MAX_ALPHABET_SIZE];

fn create_rule_map() -> RuleMap {
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
        //println!("{}: {}", i+1, lword);
    }

    lword
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

    pub fn instructions(&self, iterations: u32) -> Vec<Instruction> {
        let lword = expand_lsystem(&self.axiom, &self.rules, iterations);
        map_word_to_instructions(&lword, &self.command_map)
    }
}
