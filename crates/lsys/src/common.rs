use std::{fmt, slice};
use std::ops::{Index, IndexMut};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde::ser::SerializeMap;
use na::{self, Point3, Vector2};

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

pub struct CommandMap([Command; MAX_ALPHABET_SIZE]);

impl CommandMap {
    pub fn new() -> CommandMap {
        Default::default()
    }

    pub fn iter(&self) -> slice::Iter<Command> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<Command> {
        self.0.iter_mut()
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

        CommandMap(map)
    }
}

impl Index<u8> for CommandMap {
    type Output = Command;

    fn index(&self, index: u8) -> &Command {
        &self.0[index as usize]
    }
}

impl Index<char> for CommandMap {
    type Output = Command;

    fn index(&self, index: char) -> &Command {
        &self.0[index as usize]
    }
}

impl IndexMut<u8> for CommandMap {
    fn index_mut(&mut self, index: u8) -> &mut Command {
        &mut self.0[index as usize]
    }
}

impl IndexMut<char> for CommandMap {
    fn index_mut(&mut self, index: char) -> &mut Command {
        &mut self.0[index as usize]
    }
}

impl fmt::Display for CommandMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let rules = self.0
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

impl fmt::Debug for CommandMap {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_map()
            .entries(self.0.iter().enumerate().filter_map(
                |(letter, command)| if *command != Command::Noop {
                    Some((letter as u8 as char, command))
                } else {
                    None
                },
            ))
            .finish()
    }
}

impl Serialize for CommandMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;

        for (letter, command) in self.0.iter().enumerate() {
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

#[derive(Serialize, Deserialize, Debug)]
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

#[derive(Debug, Clone)]
pub struct Skeleton {
    pub points: Vec<Point3<f32>>,
    pub children_map: Vec<Vec<usize>>,
    pub parent_map: Vec<usize>,
}

impl Skeleton {
    pub fn find_leaves(&self) -> Vec<usize> {
        self.children_map
            .iter()
            .enumerate()
            .filter_map(|(parent, children)| if children.is_empty() {
                Some(parent)
            } else {
                None
            })
            .collect()
    }

    /// Find the hightest point that the system reaches
    pub fn find_top(&self) -> f32 {
        self.points
            .iter()
            .map(|p| p.y)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Find the lowest point that the system reaches
    pub fn find_bottom(&self) -> f32 {
        self.points
            .iter()
            .map(|p| p.y)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    pub fn find_height(&self) -> f32 {
        self.find_top() - self.find_bottom()
    }

    /// Find the largest radius in the horizontal plane
    pub fn find_horizontal_radius(&self) -> f32 {
        self.points
            .iter()
            .map(|p| na::norm(&Vector2::new(p.x, p.z)))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    pub fn root(&self) -> usize {
        return 0;
    }
}

pub trait SkeletonBuilder: Sized {
    const DEFAULT_POINT_LIMIT: usize;
    const DEFAULT_INSTRUCTION_LIMIT: usize;

    fn build_skeleton_with_limits(
        self,
        settings: &Settings,
        size_limit: Option<usize>,
        instruction_limit: Option<usize>,
    ) -> Option<Skeleton>;

    fn build_skeleton(self, settings: &Settings) -> Option<Skeleton> {
        Self::build_skeleton_with_limits(
            self,
            settings,
            Some(Self::DEFAULT_POINT_LIMIT),
            Some(Self::DEFAULT_INSTRUCTION_LIMIT),
        )
    }

    fn build_skeleton_unlimited(self, settings: &Settings) -> Skeleton {
        Self::build_skeleton_with_limits(self, settings, None, None).unwrap()
    }
}
