use std::{fmt, mem, ptr, slice};
use std::ops::{Index, IndexMut};
use std::f32::consts::{FRAC_PI_2, PI};

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde::ser::SerializeMap;
use na::{Point3, Rotation3, UnitQuaternion, Vector3};

use common::{Instruction, CommandMap, map_word_to_instructions, MAX_ALPHABET_SIZE, Rewriter,
             Settings, Command};

#[derive(Clone)]
pub struct RuleMap([String; MAX_ALPHABET_SIZE]);

impl RuleMap {
    pub fn new() -> RuleMap {
        Default::default()
    }

    pub fn iter(&self) -> slice::Iter<String> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<String> {
        self.0.iter_mut()
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

        RuleMap(rules)
    }
}

impl Index<u8> for RuleMap {
    type Output = String;

    fn index(&self, index: u8) -> &String {
        &self.0[index as usize]
    }
}

impl Index<char> for RuleMap {
    type Output = String;

    fn index(&self, index: char) -> &String {
        &self.0[index as usize]
    }
}

impl IndexMut<u8> for RuleMap {
    fn index_mut(&mut self, index: u8) -> &mut String {
        &mut self.0[index as usize]
    }
}

impl IndexMut<char> for RuleMap {
    fn index_mut(&mut self, index: char) -> &mut String {
        &mut self.0[index as usize]
    }
}

impl fmt::Display for RuleMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let rules = self.0
            .iter()
            .enumerate()
            .filter_map(|(pred, succ)| {
                if !(succ.len() == 1 && succ.as_bytes()[0] == pred as u8) {
                    Some(format!("{} -> {}", pred as u8 as char, succ))
                } else {
                    None
                }
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

impl fmt::Debug for RuleMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.0
            .iter()
            .enumerate()
            .filter_map(|(letter, word)| if word.len() != 1 || word.as_bytes()[0] != letter as u8 {
                Some((letter as u8 as char, word))
            } else {
                None
            })
        ).finish()
    }
}

impl Serialize for RuleMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;

        for (pred, succ) in self.0.iter().enumerate() {
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
    where
        M: de::MapAccess<'de>,
    {
        let mut values = RuleMap::new();

        while let Some((key, value)) = access.next_entry::<char, String>()? {
            values[key] = value;
        }

        Ok(values)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

    pub fn instructions_iter<'a, 'b>(
        &'a self,
        iterations: u32,
        command_map: &'b CommandMap,
    ) -> InstructionsIter<'a, 'b> {
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
    pub fn new(
        lsystem: &'a LSystem,
        command_map: &'b CommandMap,
        iterations: u32,
    ) -> InstructionsIter<'a, 'b> {
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
            self.visit_stack
                .extend(succ_iter.rev().map(|sym| (next_lvl, *sym)));
        }

        if let Some((_, sym)) = top {
            let command = self.command_map[sym];
            Some(Instruction::new(command))
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct Skeleton {
    pub points: Vec<Point3<f32>>,
    pub edges: Vec<Vec<usize>>,
}

impl Skeleton {
    pub fn build_with_limits(
        instructions: InstructionsIter,
        settings: &Settings,
        size_limit: Option<usize>,
        instruction_limit: Option<usize>,
    ) -> Option<Skeleton> {
        let segment_length = settings.step;

        let mut points = Vec::new();
        points.push(Point3::new(0.0, 0.0, 0.0));

        let mut edges = Vec::new();

        let mut position = Point3::new(0.0, 0.0, 0.0);
        let mut rotation = UnitQuaternion::from_euler_angles(FRAC_PI_2, 0.0, 0.0);
        let mut parent = 0usize;
        let mut filling = false;

        let mut states = Vec::<(Point3<f32>, UnitQuaternion<f32>, usize)>::new();

        for (iteration, instruction) in instructions.enumerate() {
            if let Some(size_limit) = size_limit {
                if points.len() > size_limit {
                    return None;
                }
            }

            if let Some(instruction_limit) = instruction_limit {
                if iteration >= instruction_limit {
                    return None;
                }
            }

            let command = instruction.command;
            match command {
                Command::Forward => {
                    let segment_length = {
                        if let Some(ref args) = instruction.args {
                            args[0]
                        } else {
                            segment_length
                        }
                    };

                    if !filling {
                        let direction = rotation * Vector3::new(0.0, 0.0, -1.0);
                        position += direction * segment_length;

                        let index = points.len();
                        points.push(position);
                        edges.push(Vec::new());

                        edges[parent].push(index);
                        parent = index;
                    }
                }
                Command::YawRight => {
                    let angle = {
                        if let Some(ref args) = instruction.args {
                            args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation *= Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * -angle);
                }
                Command::YawLeft => {
                    let angle = {
                        if let Some(ref args) = instruction.args {
                            args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation *= Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * angle);
                }
                Command::UTurn => {
                    let angle = PI;
                    rotation *= Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * angle);
                }
                Command::PitchUp => {
                    let angle = {
                        if let Some(ref args) = instruction.args {
                            args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation *= Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * angle);
                }
                Command::PitchDown => {
                    let angle = {
                        if let Some(ref args) = instruction.args {
                            args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation *= Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * -angle);
                }
                Command::RollRight => {
                    let angle = {
                        if let Some(ref args) = instruction.args {
                            args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation *= Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * -angle);
                }
                Command::RollLeft => {
                    let angle = {
                        if let Some(ref args) = instruction.args {
                            args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation *= Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * angle);
                }
                Command::Shrink |
                Command::Grow |
                Command::Width |
                Command::NextColor |
                Command::Noop => {}
                Command::Push => {
                    states.push((position, rotation, parent));
                }
                Command::Pop => {
                    if let Some((stored_position, stored_rotation, stored_parent)) = states.pop() {
                        position = stored_position;
                        rotation = stored_rotation;
                        parent = stored_parent;
                    } else {
                        panic!("Tried to pop empty state stack");
                    }
                }
                Command::BeginSurface => {
                    filling = true;
                    states.push((position, rotation, parent));
                }
                Command::EndSurface => {
                    if let Some((stored_position, stored_rotation, stored_parent)) = states.pop() {
                        position = stored_position;
                        rotation = stored_rotation;
                        parent = stored_parent;
                    } else {
                        panic!("Tried to pop empty state stack");
                    }

                    filling = false;
                }
            };
        }

        Some(Skeleton {
            points: points,
            edges: edges,
        })
    }

    pub fn build_unlimited(
        instructions: InstructionsIter,
        settings: &Settings,
    ) -> Skeleton {
        // Safe to unwrap because we set no limits.
        Self::build_with_limits(instructions, settings, None, None).unwrap()
    }


    pub fn build(
        instructions: InstructionsIter,
        settings: &Settings,
    ) -> Option<Skeleton> {
        const DEFAULT_SKELETON_LIMIT: usize = 10_000;
        const DEFAULT_INSTRUCTION_LIMIT: usize = DEFAULT_SKELETON_LIMIT * 50;
        Self::build_with_limits(
            instructions,
            settings,
            Some(DEFAULT_SKELETON_LIMIT),
            Some(DEFAULT_INSTRUCTION_LIMIT)
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use common::CommandMap;

    #[test]
    fn test_instructions_iter() {
        let mut lsystem = LSystem::new();
        lsystem.axiom = "F+F".to_string();
        lsystem.set_rule('F', "+F");
        lsystem.set_rule('+', "F+");

        let command_map = CommandMap::default();

        let instructions = lsystem
            .instructions_iter(5, &command_map)
            .collect::<Vec<_>>();
        assert_eq!(instructions, lsystem.instructions(5, &command_map))
    }
}
