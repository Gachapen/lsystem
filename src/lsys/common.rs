#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Command {
    Forward,
    Backward,
    YawRight,
    YawLeft,
    PitchUp,
    PitchDown,
    RollRight,
    RollLeft,
    Shrink,
    Grow,
    Push,
    Pop,
    Noop,
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

    lchar_commands
}

pub struct Settings {
    pub angle: f32,
    pub width: f32,
    pub shrink_rate: f32,
}

impl Settings {
    pub fn new() -> Settings {
        Settings {
            angle: 0.0,
            width: 1.0,
            shrink_rate: 1.0,
        }
    }
}
