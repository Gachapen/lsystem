extern crate kiss3d;
extern crate nalgebra as na;
extern crate lsys;

use na::{Vector3, Point3, Rotation3, Translate, BaseFloat};
use kiss3d::window::Window;
use kiss3d::light::Light;
use kiss3d::camera::ArcBall;

use lsys::ol;
use lsys::il;
use lsys::Command;

fn main() {
    let mut window = Window::new("lsystem");
    window.set_light(Light::StickToCamera);
    window.set_background_color(1.0, 1.0, 1.0);

    let mut camera = {
        let eye = Point3::new(0.0, 0.0, 20.0);
        let at = na::origin();
        ArcBall::new(eye, at)
    };

    let segment_length = 0.2;
    let (system, settings) = make_hogeweg_e();

    println!("Expanding");
    let instructions = system.instructions();

    let mut tree = window.add_group();
    tree.append_rotation(&Vector3::new(f32::frac_pi_2(), 0.0, 0.0));
    //tree.append_translation(&Vector3::new(0.0, -10.0, 0.0));
    let mut position = Point3::new(0.0, 0.0, 0.0);
    let mut rotation = Rotation3::new(Vector3::new(0.0, 0.0, 0.0));
    let mut width = settings.width;
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
                rotation = Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * -settings.angle) * rotation;
            },
            &Command::YawLeft => {
                rotation = Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * settings.angle) * rotation;
            },
            &Command::PitchUp => {
                rotation = Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * settings.angle) * rotation;
            },
            &Command::PitchDown => {
                rotation = Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * -settings.angle) * rotation;
            }
            &Command::RollRight => {
                rotation = Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * -settings.angle) * rotation;
            },
            &Command::RollLeft => {
                rotation = Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * settings.angle) * rotation;
            },
            &Command::Shrink => {
                width = width / settings.shrink_rate;
            },
            &Command::Grow => {
                width = width * settings.shrink_rate;
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

//fn make_thing1() -> ol::LSystem {
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

//    ol::LSystem {
//        axiom: vec![Command::Forward],
//        iterations: 4,
//        angle: f32::frac_pi_4(),
//        rules: rules,
//    }
//}

fn make_hilbert() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.rules['A' as usize] = String::from("B-F+CFC+F-D&F^D-F+&&CFC+F+B>>");
    system.rules['B' as usize] = String::from("A&F^CFB^F^D^^-F-D^++F^B++FC^F^A>>");
    system.rules['C' as usize] = String::from("++D^++F^-F+C^F^A&&FA&F^C+F+B^F^D>>");
    system.rules['D' as usize] = String::from("++CFB-F+B++FA&F^A&&FB-F+B++FC>>");

    system.axiom = String::from("A");
    system.iterations = 2;

    let settings = lsys::Settings {
        angle: f32::to_radians(90.0),
        width: 0.01,
        ..lsys::Settings::new()
    };

    (system, settings)
}

//fn make_koch1() -> ol::LSystem {
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

//    ol::LSystem {
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

//fn make_koch2() -> ol::LSystem {
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

//    ol::LSystem {
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

//fn make_koch3() -> ol::LSystem {
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

//    ol::LSystem {
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

fn make_plant1() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.command_map['X' as usize] = Command::Noop;

    system.rules['X' as usize] = String::from("F[+X][-X][&X][^X]FX");
    system.rules['F' as usize] = String::from("F!F!");

    system.axiom = String::from("X");
    system.iterations = 6;

    let settings = lsys::Settings {
        angle: 0.4485496,
        width: 0.03,
        shrink_rate: 1.01,
        ..lsys::Settings::new()
    };

    (system, settings)
}

//fn make_plant2() -> ol::LSystem {
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

//    ol::LSystem {
//        axiom: vec![Command::A],
//        iterations: 5,
//        angle: 0.3926991,
//        rules: rules,
//    }
//}

//fn make_wheat() -> ol::LSystem {
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

//    ol::LSystem {
//        axiom: vec![Command::A],
//        iterations: 10,
//        angle: f32::frac_pi_4(),
//        rules: rules,
//    }
//}

//fn make_plant3() -> ol::LSystem {
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

//    ol::LSystem {
//        axiom: vec![Command::A],
//        iterations: 6,
//        angle: f32::frac_pi_4(),
//        rules: rules,
//    }
//}

fn make_gosper_hexa() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.command_map['l' as usize] = Command::Forward;
    system.command_map['r' as usize] = Command::Forward;

    system.rules['l' as usize] = String::from("l+r++r-l--ll-r+");
    system.rules['r' as usize] = String::from("-l+rr++r+l--l-r");

    system.axiom = String::from("l");
    system.iterations = 4;

    let settings = lsys::Settings {
        angle: f32::to_radians(60.0),
        width: 0.02,
        ..lsys::Settings::new()
    };

    (system, settings)
}

fn make_2012xuequiang() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    // 3D
    //system.set_rule('X', "F&[[^F^Y]&F&Y]^[[&F&Y]^F^Y]-[[+F+Y]-F-Y]+F[+FX+Y]-X");
    //system.set_rule('Y', "F[&Y]F[^Y]F[+Y]F[-Y]+Y");

    // 2D
    system.set_rule('X', "F-[[+F+Y]-F-Y]+F[+FX+Y]-X");
    system.set_rule('Y', "F[+Y]F[-Y]+Y");
    system.set_rule('F', "FF");

    system.axiom = String::from("X");
    system.iterations = 5;

    let settings = lsys::Settings {
        angle: f32::to_radians(23.5),
        width: 0.02,
        ..lsys::Settings::new()
    };

    (system, settings)
}

fn make_hogeweg_b() -> (il::LSystem, lsys::Settings) {
    let mut sys = il::LSystem::new();

    sys.axiom = "F1F1F1".to_string();
    sys.ignore_from_context("+-F");
    sys.productions = vec![
        il::Production::with_context('0', '0', '0', "1"),
        il::Production::with_context('0', '0', '1', "1[-F1F1]"),
        il::Production::with_context('0', '1', '0', "1"),
        il::Production::with_context('0', '1', '1', "1"),
        il::Production::with_context('1', '0', '0', "0"),
        il::Production::with_context('1', '0', '1', "1F1"),
        il::Production::with_context('1', '1', '0', "1"),
        il::Production::with_context('1', '1', '1', "0"),
        il::Production::without_context('+', "-"),
        il::Production::without_context('-', "+"),
    ];
    sys.iterations = 30;

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.02,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_hogeweg_a() -> (il::LSystem, lsys::Settings) {
    let mut sys = il::LSystem::new();

    sys.axiom = "F1F1F1".to_string();
    sys.ignore_from_context("+-F");
    sys.productions = vec![
        il::Production::with_context('0', '0', '0', "0"),
        il::Production::with_context('0', '0', '1', "1[+F1F1]"),
        il::Production::with_context('0', '1', '0', "1"),
        il::Production::with_context('0', '1', '1', "1"),
        il::Production::with_context('1', '0', '0', "0"),
        il::Production::with_context('1', '0', '1', "1F1"),
        il::Production::with_context('1', '1', '0', "0"),
        il::Production::with_context('1', '1', '1', "0"),
        il::Production::without_context('+', "-"),
        il::Production::without_context('-', "+"),
    ];
    sys.iterations = 30;

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.02,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_hogeweg_c() -> (il::LSystem, lsys::Settings) {
    let mut sys = il::LSystem::new();

    sys.axiom = "F1F1F1".to_string();
    sys.ignore_from_context("+-F");
    sys.productions = vec![
        il::Production::with_context('0', '0', '0', "0"),
        il::Production::with_context('0', '0', '1', "1"),
        il::Production::with_context('0', '1', '0', "0"),
        il::Production::with_context('0', '1', '1', "1[+F1F1]"),
        il::Production::with_context('1', '0', '0', "0"),
        il::Production::with_context('1', '0', '1', "1F1"),
        il::Production::with_context('1', '1', '0', "0"),
        il::Production::with_context('1', '1', '1', "0"),
        il::Production::without_context('+', "-"),
        il::Production::without_context('-', "+"),
    ];
    sys.iterations = 26;

    let settings = lsys::Settings {
        angle: f32::to_radians(22.75),
        width: 0.02,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_hogeweg_d() -> (il::LSystem, lsys::Settings) {
    let mut sys = il::LSystem::new();

    sys.axiom = "F0F1F1".to_string();
    sys.ignore_from_context("+-F");
    sys.productions = vec![
        il::Production::with_context('0', '0', '0', "1"),
        il::Production::with_context('0', '0', '1', "0"),
        il::Production::with_context('0', '1', '0', "0"),
        il::Production::with_context('0', '1', '1', "1F1"),
        il::Production::with_context('1', '0', '0', "1"),
        il::Production::with_context('1', '0', '1', "1[+F1F1]"),
        il::Production::with_context('1', '1', '0', "1"),
        il::Production::with_context('1', '1', '1', "0"),
        il::Production::without_context('+', "-"),
        il::Production::without_context('-', "+"),
    ];
    sys.iterations = 24;

    let settings = lsys::Settings {
        angle: f32::to_radians(22.75),
        width: 0.02,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_hogeweg_e() -> (il::LSystem, lsys::Settings) {
    let mut sys = il::LSystem::new();

    sys.axiom = "F1F1F1".to_string();
    sys.ignore_from_context("+-F");
    sys.productions = vec![
        il::Production::with_context('0', '0', '0', "0"),
        il::Production::with_context('0', '0', '1', "1[-F1F1]"),
        il::Production::with_context('0', '1', '0', "1"),
        il::Production::with_context('0', '1', '1', "1"),
        il::Production::with_context('1', '0', '0', "0"),
        il::Production::with_context('1', '0', '1', "1F1"),
        il::Production::with_context('1', '1', '0', "1"),
        il::Production::with_context('1', '1', '1', "0"),
        il::Production::without_context('+', "-"),
        il::Production::without_context('-', "+"),
    ];
    sys.iterations = 30;

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.02,
        ..lsys::Settings::new()
    };

    (sys, settings)
}
