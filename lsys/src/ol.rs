use std::mem;
use std::ptr;

use common::Command;
use common::CommandMap;
use common::create_command_map;
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
    }

    lword
}

fn map_lword_to_commands(lword: &str, lchar_commands: &CommandMap) -> Vec<Command> {
    let mut commands = Vec::<Command>::with_capacity(lword.len());
    for lchar in lword.bytes() {
        let command = lchar_commands[lchar as usize];
        if command != Command::Noop {
            commands.push(command);
        }
    }
    commands
}

pub struct LSystem {
    pub command_map: CommandMap,
    pub rules: RuleMap,
    pub axiom: String,
    pub iterations: u32,
}

impl LSystem {
    pub fn new() -> LSystem {
        LSystem {
            command_map: create_command_map(),
            rules: create_rule_map(),
            axiom: String::new(),
            iterations: 0,
        }
    }

    pub fn map_command(&mut self, letter: char, command: Command) {
        self.command_map[letter as u8 as usize] = command;
    }

    pub fn set_rule(&mut self, letter: char, expansion: &str) {
        self.rules[letter as usize] = String::from(expansion);
    }

    pub fn instructions(&self) -> Vec<Command> {
        let lword = expand_lsystem(&self.axiom, &self.rules, self.iterations);
        map_lword_to_commands(&lword, &self.command_map)
    }
}
