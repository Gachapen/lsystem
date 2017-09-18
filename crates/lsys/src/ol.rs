use std::{mem, ptr, fmt, slice};
use std::ops::{Index, IndexMut};

use serde::{Serialize, Serializer, Deserialize, Deserializer, de};
use serde::ser::SerializeMap;

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
        Default::default()
    }

    pub fn iter(&self) -> slice::Iter<String> {
        self.map.iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<String> {
        self.map.iter_mut()
    }
}

impl Default for RuleMap {
    fn default() -> RuleMap {
        let mut rules: [String; MAX_ALPHABET_SIZE] = unsafe { mem::uninitialized() };

        for (i, v) in rules.iter_mut().enumerate() {
            let mut rule = String::with_capacity(1);
            rule.push(i as u8 as char);
            unsafe {
                ptr::write(v, rule);
            }
        }

        RuleMap { map: rules }
    }
}

impl Index<u8> for RuleMap {
    type Output = String;

    fn index(&self, index: u8) -> &String {
        &self.map[index as usize]
    }
}

impl Index<char> for RuleMap {
    type Output = String;

    fn index(&self, index: char) -> &String {
        &self.map[index as usize]
    }
}

impl IndexMut<u8> for RuleMap {
    fn index_mut(&mut self, index: u8) -> &mut String {
        &mut self.map[index as usize]
    }
}

impl IndexMut<char> for RuleMap {
    fn index_mut(&mut self, index: char) -> &mut String {
        &mut self.map[index as usize]
    }
}

impl fmt::Display for RuleMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let rules = self.map
            .iter()
            .enumerate()
            .filter_map(|(pred, succ)| if !(succ.len() == 1 && succ.as_bytes()[0] == pred as u8) {
                            Some(format!("{} -> {}", pred as u8 as char, succ))
                        } else {
                            None
                        })
            .collect::<Vec<_>>();

        for (i, rule) in rules.iter().enumerate() {
            write!(f, "{}", rule)?;
            if i != rules.len() - 1 {
                writeln!(f)?;
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

impl<'de> Deserialize<'de> for RuleMap {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(RuleMapVisitor {})
    }
}

struct RuleMapVisitor {}

impl<'de> de::Visitor<'de> for RuleMapVisitor {
    type Value = RuleMap;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an ol::RuleMap")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where M: de::MapAccess<'de>
    {
        let mut values = RuleMap::new();

        while let Some((key, value)) = access.next_entry::<char, String>()? {
            values[key] = value;
        }

        Ok(values)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
        where E: de::Error
    {
        Ok(RuleMap::new())
    }
}

fn expand_lsystem(axiom: &str, rules: &RuleMap, iterations: u32) -> String {
    let mut lword = String::from(axiom);

    for _ in 0..iterations {
        let mut expanded_lword = String::with_capacity(lword.len());
        for lchar in lword.bytes() {
            let expanded_lchar = &rules[lchar];
            expanded_lword.push_str(&expanded_lchar.clone());
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

#[derive(Serialize, Deserialize)]
pub struct LSystem {
    pub productions: RuleMap,
    pub axiom: String,
}

impl LSystem {
    pub fn new() -> LSystem {
        Default::default()
    }

    pub fn set_rule(&mut self, letter: char, expansion: &str) {
        self.productions[letter] = String::from(expansion);
    }

    pub fn remove_redundancy(&mut self) {
        self.axiom = remove_redundancy(&self.axiom);

        for successor in self.productions.iter_mut() {
            *successor = remove_redundancy(successor);
        }
    }

    pub fn rewrite(&self, iterations: u32) -> String {
        expand_lsystem(&self.axiom, &self.productions, iterations)
    }

    pub fn instructions_iter<'a, 'b>(&'a self,
                                     iterations: u32,
                                     command_map: &'b CommandMap)
                                     -> InstructionsIter<'a, 'b> {
        InstructionsIter::new(self, command_map, iterations)
    }
}

impl Default for LSystem {
    fn default() -> LSystem {
        LSystem {
            productions: RuleMap::new(),
            axiom: String::new(),
        }
    }
}

impl Rewriter for LSystem {
    fn instructions(&self, iterations: u32, command_map: &CommandMap) -> Vec<Instruction> {
        let lword = expand_lsystem(&self.axiom, &self.productions, iterations);
        map_word_to_instructions(&lword, command_map)
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

#[derive(Clone)]
pub struct InstructionsIter<'a, 'b> {
    lsystem: &'a LSystem,
    command_map: &'b CommandMap,
    num_iterations: u32,
    visit_stack: Vec<(u32, u8)>,
}

impl<'a, 'b> InstructionsIter<'a, 'b> {
    pub fn new(lsystem: &'a LSystem,
               command_map: &'b CommandMap,
               iterations: u32)
               -> InstructionsIter<'a, 'b> {
        let mut iter = InstructionsIter {
            lsystem: lsystem,
            command_map: command_map,
            num_iterations: iterations,
            visit_stack: Vec::with_capacity(lsystem.axiom.len()),
        };

        for symbol in iter.lsystem.axiom.as_bytes().iter().rev() {
            iter.visit_stack.push((0, *symbol));
        }

        iter
    }
}

impl<'a, 'b> Iterator for InstructionsIter<'a, 'b> {
    type Item = Instruction;

    fn next(&mut self) -> Option<Instruction> {
        let mut top = self.visit_stack.pop();

        while top.is_some() && top.unwrap().0 < self.num_iterations {
            let (lvl, sym) = top.unwrap();

            let successor = &self.lsystem.productions[sym];
            let next_lvl = lvl + 1;

            let mut succ_iter = successor.as_bytes().iter();
            top = Some((next_lvl, *succ_iter.next().unwrap()));

            self.visit_stack.reserve(successor.len() - 1);
            self.visit_stack.extend(
                succ_iter.rev().map(|sym| (next_lvl, *sym)),
            );
        }

        if let Some((_, sym)) = top {
            let command = self.command_map[sym];
            Some(Instruction::new(command))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use common::create_command_map;

    #[test]
    fn test_instructions_iter() {
        let mut lsystem = LSystem::new();
        lsystem.axiom = "F+F".to_string();
        lsystem.set_rule('F', "+F");
        lsystem.set_rule('+', "F+");

        let command_map = create_command_map();

        let instructions = lsystem
            .instructions_iter(5, &command_map)
            .collect::<Vec<_>>();
        assert_eq!(instructions, lsystem.instructions(5, &command_map))
    }
}
