extern crate kiss3d;
extern crate nalgebra as na;
extern crate time;
extern crate lsys;

use std::rc::Rc;

use na::{Vector3, Point3, Rotation3, Translate, BaseFloat};
use kiss3d::window::Window;
use kiss3d::light::Light;
use kiss3d::camera::ArcBall;

use lsys::Command;
use lsys::ol;
use lsys::il;
use lsys::param;
use lsys::param::Param;
use lsys::param::WordFromString;

fn main() {
    let mut window = Window::new("lsystem");
    window.set_light(Light::StickToCamera);
    window.set_background_color(1.0, 1.0, 1.0);
    window.set_framerate_limit(Some(60));

    let mut camera = {
        let eye = Point3::new(0.0, 0.0, 20.0);
        let at = na::origin();
        ArcBall::new(eye, at)
    };

    let segment_length = 0.2;
    let (system, settings) = make_anim_tree();

    let mut tree = window.add_group();

    let mut word = system.axiom.clone();
    let mut time = time::precise_time_s();

    while window.render_with_camera(&mut camera) {
        let prev_time = time;
        time = time::precise_time_s();
        let dt = time - prev_time;

        word = param::step(&word, &system.productions, dt as f32 * 1.0);
        let instructions = param::map_word_to_instructions(&word, &system.command_map);

        tree.unlink();
        tree = window.add_group();
        tree.append_rotation(&Vector3::new(f32::frac_pi_2(), 0.0, 0.0));

        let mut position = Point3::new(0.0, 0.0, 0.0);
        let mut rotation = Rotation3::new(Vector3::new(0.0, 0.0, 0.0));
        let mut width = settings.width;
        let mut states = Vec::<(Point3<f32>, Rotation3<f32>, f32)>::new();

        for instruction in &instructions {
            let command = instruction.command;
            match command {
                Command::Forward => {
                    let segment_length = {
                        if !instruction.args.is_empty() {
                           instruction.args[0]
                        } else {
                            segment_length
                        }
                    };

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
                Command::YawRight => {
                    let angle = {
                        if !instruction.args.is_empty() {
                           instruction.args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation = rotation * Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * -angle);
                },
                Command::YawLeft => {
                    let angle = {
                        if !instruction.args.is_empty() {
                           instruction.args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation = rotation * Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * angle);
                },
                Command::PitchUp => {
                    let angle = {
                        if !instruction.args.is_empty() {
                           instruction.args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation = rotation * Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * angle);
                },
                Command::PitchDown => {
                    let angle = {
                        if !instruction.args.is_empty() {
                           instruction.args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation = rotation * Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * -angle);
                }
                Command::RollRight => {
                    let angle = {
                        if !instruction.args.is_empty() {
                           instruction.args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation = rotation * Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * -angle);
                },
                Command::RollLeft => {
                    let angle = {
                        if !instruction.args.is_empty() {
                           instruction.args[0]
                        } else {
                            settings.angle
                        }
                    };
                    rotation = rotation * Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * angle);
                },
                Command::Shrink => {
                    let rate = {
                        if !instruction.args.is_empty() {
                           instruction.args[0]
                        } else {
                            settings.shrink_rate
                        }
                    };
                    width = width / rate;
                },
                Command::Grow => {
                    let rate = {
                        if !instruction.args.is_empty() {
                           instruction.args[0]
                        } else {
                            settings.shrink_rate
                        }
                    };
                    width = width * rate;
                },
                Command::Width => {
                    width = instruction.args[0];
                },
                Command::Push => {
                    states.push((position, rotation, width));
                },
                Command::Pop => {
                    if let Some((stored_position, stored_rotation, stored_width)) = states.pop() {
                        position = stored_position;
                        rotation = stored_rotation;
                        width = stored_width;
                    } else {
                        panic!("Tried to pop empty state stack");
                    }
                },
                Command::Noop => {},
            };
        }

        tree.append_rotation(&Vector3::new(0.0f32, 0.004, 0.0));
        tree.set_color(50.0/255.0, 169.0/255.0, 18.0/255.0);
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

    let settings = lsys::Settings {
        angle: f32::to_radians(90.0),
        width: 0.01,
        iterations: 2,
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

    let settings = lsys::Settings {
        angle: 0.4485496,
        width: 0.03,
        shrink_rate: 1.01,
        iterations: 6,
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

    let settings = lsys::Settings {
        angle: f32::to_radians(60.0),
        width: 0.02,
        iterations: 4,
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

    let settings = lsys::Settings {
        angle: f32::to_radians(23.5),
        width: 0.02,
        iterations: 5,
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

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.02,
        iterations: 30,
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

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.02,
        iterations: 30,
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

    let settings = lsys::Settings {
        angle: f32::to_radians(22.75),
        width: 0.02,
        iterations: 26,
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

    let settings = lsys::Settings {
        angle: f32::to_radians(22.75),
        width: 0.02,
        iterations: 24,
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

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.02,
        iterations: 30,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_bush() -> (ol::LSystem, lsys::Settings) {
    let mut sys = ol::LSystem::new();

    sys.axiom = "A".to_string();
    sys.set_rule('A', "[&F!A]>>>>>[&F!A]>>>>>>>[&F!A]");
    sys.set_rule('F', "S>>>>>F");
    sys.set_rule('S', "F");

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.1,
        shrink_rate: 1.5,
        iterations: 7,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_antenna() -> (param::LSystem, lsys::Settings) {
    let mut sys = param::LSystem::new();

    sys.axiom = param::Word::from_str("A");

    let r = 1.456;
    sys.productions = vec![
        param::Production::new(
            'A',
            vec![
                param::ProductionLetter::with_params('F', vec![Param::Float(1.0)]),
                param::ProductionLetter::new('['),
                param::ProductionLetter::new('+'),
                param::ProductionLetter::new('A'),
                param::ProductionLetter::new(']'),
                param::ProductionLetter::new('['),
                param::ProductionLetter::new('-'),
                param::ProductionLetter::new('A'),
                param::ProductionLetter::new(']'),
            ]
        ),
        param::Production::new(
            'F',
            vec![
                param::ProductionLetter::with_transform('F', Rc::new(move |p,_| vec![Param::Float(p[0].float().unwrap() * r)])),
            ]
        ),
    ];

    let settings = lsys::Settings {
        angle: f32::to_radians(85.0),
        width: 0.05,
        iterations: 10,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

//fn make_tree() -> (param::LSystem, lsys::Settings) {
//    let mut sys = param::LSystem::new();

//    sys.axiom = vec![
//        param::Letter::with_params('#', vec![Param::Float(1.0/10.0)]),
//        param::Letter::with_params('F', vec![Param::Float(5.0)]),
//        param::Letter::with_params('>', vec![Param::Float(f32::to_radians(45.0))]),
//        param::Letter::new('A'),
//    ];

//    let d1 = f32::to_radians(94.74);
//    let d2 = f32::to_radians(132.63);
//    let a = f32::to_radians(18.95);
//    let lr = 1.309;
//    let vr = 1.732 / 10.0;

//    sys.productions = vec![
//        param::Production::new(
//            'A',
//            vec![
//                param::ProductionLetter::with_transform('#', Rc::new(move |_| vec![Param::Float(vr)])),
//                param::ProductionLetter::with_params('F', vec![Param::Float(2.0)]),
//                param::ProductionLetter::new('['),
//                param::ProductionLetter::with_transform('&', Rc::new(move |_| vec![Param::Float(a)])),
//                param::ProductionLetter::with_params('F', vec![Param::Float(2.0)]),
//                param::ProductionLetter::new('A'),
//                param::ProductionLetter::new(']'),
//                param::ProductionLetter::with_transform('>', Rc::new(move |_| vec![Param::Float(d1)])),
//                param::ProductionLetter::new('['),
//                param::ProductionLetter::with_transform('&', Rc::new(move |_| vec![Param::Float(a)])),
//                param::ProductionLetter::with_params('F', vec![Param::Float(2.0)]),
//                param::ProductionLetter::new('A'),
//                param::ProductionLetter::new(']'),
//                param::ProductionLetter::with_transform('>', Rc::new(move |_| vec![Param::Float(d2)])),
//                param::ProductionLetter::new('['),
//                param::ProductionLetter::with_transform('&', Rc::new(move |_| vec![Param::Float(a)])),
//                param::ProductionLetter::with_params('F', vec![Param::Float(2.0)]),
//                param::ProductionLetter::new('A'),
//                param::ProductionLetter::new(']'),
//            ]
//        ),
//        param::Production::new(
//            'F',
//            vec![
//                param::ProductionLetter::with_transform('F', Rc::new(move |p| vec![Param::Float(p[0].float().unwrap() * lr)])),
//            ]
//        ),
//        param::Production::new(
//            '#',
//            vec![
//                param::ProductionLetter::with_transform('#', Rc::new(move |p| vec![Param::Float(p[0].float().unwrap() * vr)])),
//            ]
//        ),
//    ];

//    let settings = lsys::Settings {
//        width: 0.05,
//        iterations: 7,
//        ..lsys::Settings::new()
//    };

//    (sys, settings)
//}

fn make_anim_tree() -> (param::LSystem, lsys::Settings) {
    let mut sys = param::LSystem::new();

    sys.axiom = vec![
        param::Letter::with_params('#', vec![Param::Float(0.01)]),
        param::Letter::with_params('F', vec![Param::Float(0.0)]),
        param::Letter::with_params('>', vec![Param::Float(f32::to_radians(45.0))]),
        param::Letter::with_params('A', vec![Param::Float(0.0)]),
    ];

    let d1 = f32::to_radians(94.74);
    let d2 = f32::to_radians(132.63);
    let a = f32::to_radians(18.95);
    let lr = 1.309;
    let vr = 1.732 / 10.0;

    sys.productions = vec![
        param::Production::with_condition(
            'A',
            Rc::new(|p| p[0].float().unwrap() < 1.0),
            vec![
                param::ProductionLetter::with_transform('A', Rc::new(|p,dt| vec![Param::Float(p[0].float().unwrap() + dt)])),
            ]
        ),
        param::Production::with_condition(
            'A',
            Rc::new(|p| p[0].float().unwrap() >= 1.0),
            vec![
                param::ProductionLetter::with_transform('#', Rc::new(move |_,_| vec![Param::Float(vr)])),
                param::ProductionLetter::with_params('F', vec![Param::Float(0.0)]),
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_transform('&', Rc::new(move |_,_| vec![Param::Float(a)])),
                param::ProductionLetter::with_params('F', vec![Param::Float(0.0)]),
                param::ProductionLetter::with_params('A', vec![Param::Float(0.0)]),
                param::ProductionLetter::new(']'),
                param::ProductionLetter::with_transform('>', Rc::new(move |_,_| vec![Param::Float(d1)])),
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_transform('&', Rc::new(move |_,_| vec![Param::Float(a)])),
                param::ProductionLetter::with_params('F', vec![Param::Float(0.0)]),
                param::ProductionLetter::with_params('A', vec![Param::Float(0.0)]),
                param::ProductionLetter::new(']'),
                param::ProductionLetter::with_transform('>', Rc::new(move |_,_| vec![Param::Float(d2)])),
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_transform('&', Rc::new(move |_,_| vec![Param::Float(a)])),
                param::ProductionLetter::with_params('F', vec![Param::Float(0.0)]),
                param::ProductionLetter::with_params('A', vec![Param::Float(0.0)]),
                param::ProductionLetter::new(']'),
            ]
        ),
        param::Production::new(
            'F',
            vec![
                param::ProductionLetter::with_transform('F', Rc::new(move |p,dt| vec![Param::Float(p[0].float().unwrap() + dt * lr)])),
            ]
        ),
        param::Production::new(
            '#',
            vec![
                param::ProductionLetter::with_transform('#', Rc::new(move |p,dt| vec![Param::Float(p[0].float().unwrap() + dt * vr)])),
            ]
        ),
    ];

    let settings = lsys::Settings {
        width: 0.05,
        iterations: 7,
        ..lsys::Settings::new()
    };

    (sys, settings)
}
