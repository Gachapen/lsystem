use std::{fmt, slice};
use std::ops::{Index, IndexMut};
use serde::{Serialize, Serializer, Deserialize, Deserializer, de};
use serde::ser::SerializeMap;

#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum Command {
    Forward,
    YawRight,
    YawLeft,
    PitchUp,
    PitchDown,
    RollRight,
    RollLeft,
    UTurn,
    Shrink,
    Grow,
    Width,
    Push,
    Pop,
    BeginSurface,
    EndSurface,
    NextColor,
    Noop,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Instruction {
    pub command: Command,
    pub args: Option<Vec<f32>>,
}

impl Instruction {
    pub fn new(command: Command) -> Instruction {
        Instruction {
            command: command,
            args: None,
        }
    }
}

pub const MAX_ALPHABET_SIZE: usize = 128;

pub struct CommandMap {
    map: [Command; MAX_ALPHABET_SIZE],
}

impl CommandMap {
    pub fn new() -> CommandMap {
        Default::default()
    }

    pub fn iter(&self) -> slice::Iter<Command> {
        self.map.iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<Command> {
        self.map.iter_mut()
    }
}

impl Default for CommandMap {
    fn default() -> CommandMap {
        let mut map: [Command; MAX_ALPHABET_SIZE] = [Command::Noop; MAX_ALPHABET_SIZE];

        map['F' as usize] = Command::Forward;
        map['+' as usize] = Command::YawLeft;
        map['-' as usize] = Command::YawRight;
        map['<' as usize] = Command::RollLeft;
        map['>' as usize] = Command::RollRight;
        map['^' as usize] = Command::PitchUp;
        map['&' as usize] = Command::PitchDown;
        map['[' as usize] = Command::Push;
        map[']' as usize] = Command::Pop;
        map['!' as usize] = Command::Shrink;
        map['#' as usize] = Command::Width;
        map['|' as usize] = Command::UTurn;
        map['{' as usize] = Command::BeginSurface;
        map['}' as usize] = Command::EndSurface;
        map['\'' as usize] = Command::NextColor;

        CommandMap { map: map }
    }
}

impl Index<u8> for CommandMap {
    type Output = Command;

    fn index(&self, index: u8) -> &Command {
        &self.map[index as usize]
    }
}

impl Index<char> for CommandMap {
    type Output = Command;

    fn index(&self, index: char) -> &Command {
        &self.map[index as usize]
    }
}

impl IndexMut<u8> for CommandMap {
    fn index_mut(&mut self, index: u8) -> &mut Command {
        &mut self.map[index as usize]
    }
}

impl IndexMut<char> for CommandMap {
    fn index_mut(&mut self, index: char) -> &mut Command {
        &mut self.map[index as usize]
    }
}

impl fmt::Display for CommandMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let rules = self.map
            .iter()
            .enumerate()
            .filter_map(|(letter, command)| if *command != Command::Noop {
                Some(format!("{} -> {:?}", letter as u8 as char, command))
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

impl Serialize for CommandMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;

        for (letter, command) in self.map.iter().enumerate() {
            if *command != Command::Noop {
                map.serialize_entry(&(letter as u8 as char), command)?;
            }
        }

        map.end()
    }
}

impl<'de> Deserialize<'de> for CommandMap {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(CommandMapVisitor {})
    }
}

struct CommandMapVisitor {}

impl<'de> de::Visitor<'de> for CommandMapVisitor {
    type Value = CommandMap;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an ol::CommandMap")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        let mut values = CommandMap::new();

        while let Some((key, value)) = access.next_entry::<char, Command>()? {
            values[key] = value;
        }

        Ok(values)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(CommandMap::new())
    }
}

pub fn map_word_to_instructions(word: &str, command_map: &CommandMap) -> Vec<Instruction> {
    let mut instructions = Vec::<Instruction>::with_capacity(word.len());
    for letter in word.bytes() {
        let command = command_map[letter];
        if command != Command::Noop {
            instructions.push(Instruction::new(command));
        }
    }
    instructions
}

#[derive(Serialize, Deserialize)]
pub struct Settings {
    pub iterations: u32,
    pub angle: f32,
    pub step: f32,
    pub width: f32,
    pub shrink_rate: f32,
    pub colors: Vec<(f32, f32, f32)>,
    pub command_map: CommandMap,
}

impl Settings {
    pub fn new() -> Settings {
        Default::default()
    }

    pub fn map_command(&mut self, letter: char, command: Command) {
        self.command_map[letter] = command;
    }
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            iterations: 0,
            angle: 0.0,
            step: 0.2,
            width: 0.2,
            shrink_rate: 1.0,
            colors: vec![(50.0 / 255.0, 169.0 / 255.0, 18.0 / 255.0)],
            command_map: CommandMap::default(),
        }
    }
}

pub trait Rewriter {
    fn instructions(&self, iterations: u32, command_map: &CommandMap) -> Vec<Instruction>;
}
