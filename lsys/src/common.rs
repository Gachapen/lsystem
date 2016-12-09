#[derive(Copy, Clone, PartialEq, Debug)]
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

pub struct Instruction {
    pub command: Command,
    pub args: Vec<f32>,
}

impl Instruction {
    fn new(command: Command) -> Instruction {
        Instruction {
            command: command,
            args: vec![],
        }
    }
}

pub const MAX_ALPHABET_SIZE: usize = 128;
pub type CommandMap = [Command; MAX_ALPHABET_SIZE];

pub fn create_command_map() -> CommandMap {
    let mut lchar_commands: CommandMap = [Command::Noop; MAX_ALPHABET_SIZE];

    lchar_commands['F' as usize] = Command::Forward;
    lchar_commands['+' as usize] = Command::YawLeft;
    lchar_commands['-' as usize] = Command::YawRight;
    lchar_commands['<' as usize] = Command::RollLeft;
    lchar_commands['>' as usize] = Command::RollRight;
    lchar_commands['^' as usize] = Command::PitchUp;
    lchar_commands['&' as usize] = Command::PitchDown;
    lchar_commands['[' as usize] = Command::Push;
    lchar_commands[']' as usize] = Command::Pop;
    lchar_commands['!' as usize] = Command::Shrink;
    lchar_commands['#' as usize] = Command::Width;
    lchar_commands['|' as usize] = Command::UTurn;
    lchar_commands['{' as usize] = Command::BeginSurface;
    lchar_commands['}' as usize] = Command::EndSurface;
    lchar_commands['\'' as usize] = Command::NextColor;

    lchar_commands
}

pub fn map_word_to_instructions(word: &str, command_map: &CommandMap) -> Vec<Instruction> {
    let mut instructions = Vec::<Instruction>::with_capacity(word.len());
    for letter in word.bytes() {
        let command = command_map[letter as usize];
        if command != Command::Noop {
            instructions.push(Instruction::new(command));
        }
    }
    instructions
}

pub struct Settings {
    pub iterations: u32,
    pub angle: f32,
    pub width: f32,
    pub shrink_rate: f32,
    pub colors: Vec<(f32, f32, f32)>,
}

impl Settings {
    pub fn new() -> Settings {
        Settings {
            iterations: 0,
            angle: 0.0,
            width: 1.0,
            shrink_rate: 1.0,
            colors: vec![(50.0/255.0, 169.0/255.0, 18.0/255.0)],
        }
    }
}
