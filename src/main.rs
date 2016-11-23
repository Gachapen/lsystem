extern crate kiss3d;
extern crate nalgebra as na;

use na::{Vector3, Point3, Rotation3, Translate, BaseFloat};
use kiss3d::window::Window;
use kiss3d::light::Light;
use kiss3d::camera::ArcBall;

#[derive(Copy, Clone, PartialEq, Debug)]
enum Command {
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

const MAX_ALPHABET_SIZE: usize = 128;
type RuleMap = [String; MAX_ALPHABET_SIZE];
type CommandMap = [Command; MAX_ALPHABET_SIZE];

fn create_rule_map() -> RuleMap {
    let mut rules: RuleMap = unsafe { std::mem::uninitialized() };

    for (i, v) in rules.iter_mut().enumerate() {
        let mut rule = String::with_capacity(1);
        rule.push(i as u8 as char);
        unsafe { std::ptr::write(v, rule); }
    }

    rules
}

fn create_command_map() -> CommandMap {
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
        if (command != Command::Noop) {
            commands.push(command);
        }
    }
    commands
}

struct LSystem {
    command_map: CommandMap,
    rules: RuleMap,
    axiom: String,
    iterations: u32,
    angle: f32,
    width: f32,
    shrink_rate: f32,
}

impl LSystem {
    fn new() -> LSystem {
        LSystem {
            command_map: create_command_map(),
            rules: create_rule_map(),
            axiom: String::new(),
            iterations: 0,
            angle: 0.0,
            width: 1.0,
            shrink_rate: 1.0,
        }
    }

    fn set_rule(&mut self, letter: char, expansion: &str) {
        self.rules[letter as usize] = String::from(expansion);
    }

    fn instructions(&self) -> Vec<Command> {
        let lword = expand_lsystem(&self.axiom, &self.rules, self.iterations);
        map_lword_to_commands(&lword, &self.command_map)
    }
}

fn main() {
    let mut window = Window::new("lsystem");
    window.set_light(Light::StickToCamera);
    window.set_background_color(1.0, 1.0, 1.0);

    let mut camera = {
        let eye = Point3::new(0.0, 0.0, 20.0);
        let at = na::origin();
        ArcBall::new(eye, at)
    };

    let segment_length = 0.1;
    let system = make_plant1();

    println!("Expanding");
    let instructions = system.instructions();

    let mut tree = window.add_group();
    tree.append_rotation(&Vector3::new(f32::frac_pi_2(), 0.0, 0.0));
    //tree.append_translation(&Vector3::new(0.0, -10.0, 0.0));
    let mut position = Point3::new(0.0, 0.0, 0.0);
    let mut rotation = Rotation3::new(Vector3::new(0.0, 0.0, 0.0));
    let mut width = system.width;
    let mut states = Vec::<(Point3<f32>, Rotation3<f32>, f32)>::new();

    println!("Assembling");

    for command in &instructions {
        match command {
            &Command::Forward => {
                let mut segment = tree.add_cube(1.0 * width, 1.0 * width, segment_length);
                segment.append_translation(&Vector3::new(0.0, 0.0, -segment_length / 2.0));
                segment.append_transformation(
                    &na::Isometry3 {
                        translation: position.to_vector(),
                        rotation: rotation,
                    }
                );

                let direction = na::rotate(&rotation, &Vector3::new(0.0, 0.0, -1.0));
                position = (direction * segment_length).translate(&position);
            },
            &Command::Backward => {
            },
            &Command::YawRight => {
                rotation = Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * -system.angle) * rotation;
            },
            &Command::YawLeft => {
                rotation = Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * system.angle) * rotation;
            },
            &Command::PitchUp => {
                rotation = Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * system.angle) * rotation;
            },
            &Command::PitchDown => {
                rotation = Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * -system.angle) * rotation;
            }
            &Command::RollRight => {
                rotation = Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * -system.angle) * rotation;
            },
            &Command::RollLeft => {
                rotation = Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * system.angle) * rotation;
            },
            &Command::Shrink => {
                width = width / system.shrink_rate;
            },
            &Command::Grow => {
                width = width * system.shrink_rate;
            },
            &Command::Push => {
                states.push((position, rotation, width));
            },
            &Command::Pop => {
                if let Some((stored_position, stored_rotation, stored_width)) = states.pop() {
                    position = stored_position;
                    rotation = stored_rotation;
                    width = stored_width;
                } else {
                    panic!("Tried to pop empty state stack");
                }
            },
            &Command::Noop => {},
        };
    }

    println!("Completed");

    tree.set_color(50.0/255.0, 169.0/255.0, 18.0/255.0);

    while window.render_with_camera(&mut camera) {
        tree.append_rotation(&Vector3::new(0.0f32, 0.004, 0.0));
    }
}

//fn init_rules() -> Vec<Vec<Command>> {
//    let mut rules: Vec<Vec<Command>> = vec![vec![]; 17];
//    rules[Command::A as usize] = vec![Command::A];
//    rules[Command::Forward as usize] = vec![Command::Forward];
//    rules[Command::Backward as usize] = vec![Command::Backward];
//    rules[Command::YawRight as usize] = vec![Command::YawRight];
//    rules[Command::YawLeft as usize] = vec![Command::YawLeft];
//    rules[Command::PitchUp as usize] = vec![Command::PitchUp];
//    rules[Command::PitchDown as usize] = vec![Command::PitchDown];
//    rules[Command::RollRight as usize] = vec![Command::RollRight];
//    rules[Command::RollLeft as usize] = vec![Command::RollLeft];
//    rules[Command::Shrink as usize] = vec![Command::Shrink];
//    rules[Command::Grow as usize] = vec![Command::Grow];
//    rules[Command::Push as usize] = vec![Command::Push];
//    rules[Command::Pop as usize] = vec![Command::Pop];
//    rules[Command::A as usize] = vec![Command::A];
//    rules[Command::B as usize] = vec![Command::B];
//    rules[Command::C as usize] = vec![Command::C];
//    rules[Command::D as usize] = vec![Command::D];

//    rules
//}

//fn make_thing1() -> LSystem {
//    let mut rules = init_rules();

//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawRight,
//        Command::YawRight,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward
//    ];

//    LSystem {
//        axiom: vec![Command::Forward],
//        iterations: 4,
//        angle: f32::frac_pi_4(),
//        rules: rules,
//    }
//}

fn make_hilbert() -> LSystem {
    let mut system = LSystem::new();

    system.rules['A' as usize] = String::from("B-F+CFC+F-D&F^D-F+&&CFC+F+B>>");
    system.rules['B' as usize] = String::from("A&F^CFB^F^D^^-F-D^++F^B++FC^F^A>>");
    system.rules['C' as usize] = String::from("++D^++F^-F+C^F^A&&FA&F^C+F+B^F^D>>");
    system.rules['D' as usize] = String::from("++CFB-F+B++FA&F^A&&FB-F+B++FC>>");

    system.axiom = String::from("A");
    system.iterations = 2;
    system.angle = f32::to_radians(90.0);
    system.width = 0.01;

    system
}

//fn make_koch1() -> LSystem {
//    let mut rules = init_rules();

//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawRight,
//        Command::Forward,
//    ];

//    LSystem {
//        axiom: vec![
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//        ],
//        iterations: 4,
//        angle: f32::frac_pi_2(),
//        rules: rules,
//    }
//}

//fn make_koch2() -> LSystem {
//    let mut rules = init_rules();

//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//    ];

//    LSystem {
//        axiom: vec![
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//        ],
//        iterations: 4,
//        angle: f32::frac_pi_2(),
//        rules: rules,
//    }
//}

//fn make_koch3() -> LSystem {
//    let mut rules = init_rules();

//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::Forward,
//        Command::YawLeft,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//    ];

//    LSystem {
//        axiom: vec![
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//        ],
//        iterations: 5,
//        angle: f32::frac_pi_2(),
//        rules: rules,
//    }
//}

fn make_plant1() -> LSystem {
    let mut system = LSystem::new();

    system.command_map['X' as usize] = Command::Noop;

    system.rules['X' as usize] = String::from("F[+X][-X][&X][^X]FX");
    system.rules['F' as usize] = String::from("F!F!");

    system.axiom = String::from("X");
    system.iterations = 6;
    system.angle = 0.4485496;
    system.width = 0.03;
    system.shrink_rate = 1.01;

    system
}

//fn make_plant2() -> LSystem {
//    let mut rules = init_rules();

//    rules[Command::A as usize] = vec![
//        Command::Forward,
//        Command::Shrink,
//        Command::YawLeft,
//        Command::Push,
//        Command::Push,
//        Command::A,
//        Command::Pop,
//        Command::YawRight,
//        Command::A,
//        Command::Pop,
//        Command::YawRight,
//        Command::Forward,
//        Command::Shrink,
//        Command::Push,
//        Command::YawRight,
//        Command::Forward,
//        Command::Shrink,
//        Command::A,
//        Command::Pop,
//        Command::PitchDown,
//        Command::Push,
//        Command::Push,
//        Command::A,
//        Command::Pop,
//        Command::PitchUp,
//        Command::A,
//        Command::Pop,
//        Command::PitchUp,
//        Command::Forward,
//        Command::Shrink,
//        Command::Push,
//        Command::PitchUp,
//        Command::Forward,
//        Command::Shrink,
//        Command::A,
//        Command::Pop,
//        Command::Push,
//        Command::PitchDown,
//        Command::Forward,
//        Command::Shrink,
//        Command::A,
//        Command::Pop,
//        Command::YawLeft,
//        Command::A,
//    ];
//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::Forward,
//    ];

//    LSystem {
//        axiom: vec![Command::A],
//        iterations: 5,
//        angle: 0.3926991,
//        rules: rules,
//    }
//}

//fn make_wheat() -> LSystem {
//    let mut rules = init_rules();

//    rules[Command::A as usize] = vec![
//        Command::Forward,
//        Command::Push,
//        Command::Forward,
//        Command::Pop,
//        Command::Push,
//        Command::YawLeft,
//        Command::Forward,
//        Command::Pop,
//        Command::Push,
//        Command::YawRight,
//        Command::Forward,
//        Command::Pop,
//        Command::A,
//    ];

//    LSystem {
//        axiom: vec![Command::A],
//        iterations: 10,
//        angle: f32::frac_pi_4(),
//        rules: rules,
//    }
//}

//fn make_plant3() -> LSystem {
//    let mut rules = init_rules();

//    rules[Command::A as usize] = vec![
//        Command::Forward,

//        Command::Push,
//        Command::A,
//        Command::Pop,

//        Command::Push,
//        Command::YawLeft,
//        Command::A,
//        Command::Pop,

//        Command::Forward,
//        Command::Push,
//        Command::YawRight,
//        Command::A,
//        Command::Pop,

//        Command::Push,
//        Command::PitchDown,
//        Command::A,
//        Command::Pop,

//        Command::Push,
//        Command::PitchUp,
//        Command::A,
//        Command::Pop,
//    ];

//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::Shrink,
//        Command::Forward,
//    ];

//    LSystem {
//        axiom: vec![Command::A],
//        iterations: 6,
//        angle: f32::frac_pi_4(),
//        rules: rules,
//    }
//}

fn make_gosper_hexa() -> LSystem {
    let mut system = LSystem::new();

    system.command_map['l' as usize] = Command::Forward;
    system.command_map['r' as usize] = Command::Forward;

    system.rules['l' as usize] = String::from("l+r++r-l--ll-r+");
    system.rules['r' as usize] = String::from("-l+rr++r+l--l-r");

    system.axiom = String::from("l");
    system.iterations = 4;
    system.angle = f32::to_radians(60.0);
    system.width = 0.02;

    system
}
