use lsys;
use lsys::Command;
use lsys::ol;
use lsys::il;
use lsys::param;
use lsys::param::Param;
use lsys::param::WordFromString;

#[allow(dead_code)]
pub fn make_hilbert() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.productions['A'] = String::from("B-F+CFC+F-D&F^D-F+&&CFC+F+B>>");
    system.productions['B'] = String::from("A&F^CFB^F^D^^-F-D^++F^B++FC^F^A>>");
    system.productions['C'] = String::from("++D^++F^-F+C^F^A&&FA&F^C+F+B^F^D>>");
    system.productions['D'] = String::from("++CFB-F+B++FA&F^A&&FB-F+B++FC>>");

    system.axiom = String::from("A");

    let settings = lsys::Settings {
        angle: f32::to_radians(90.0),
        width: 0.01,
        iterations: 2,
        ..lsys::Settings::new()
    };

    (system, settings)
}

#[allow(dead_code)]
pub fn make_plant1() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.productions['X'] = String::from("F[+X][-X][&X][^X]FX");
    system.productions['F'] = String::from("F!F!");

    system.axiom = String::from("X");

    let mut settings = lsys::Settings {
        angle: 0.4485496,
        width: 0.03,
        shrink_rate: 1.01,
        iterations: 6,
        ..lsys::Settings::new()
    };

    settings.command_map['X' as usize] = Command::Noop;

    (system, settings)
}

#[allow(dead_code)]
pub fn make_gosper_hexa() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.productions['l'] = String::from("l+r++r-l--ll-r+");
    system.productions['r'] = String::from("-l+rr++r+l--l-r");

    system.axiom = String::from("l");

    let mut settings = lsys::Settings {
        angle: f32::to_radians(60.0),
        width: 0.02,
        iterations: 4,
        ..lsys::Settings::new()
    };

    settings.command_map['l' as usize] = Command::Forward;
    settings.command_map['r' as usize] = Command::Forward;

    (system, settings)
}

#[allow(dead_code)]
pub fn make_2012xuequiang() -> (ol::LSystem, lsys::Settings) {
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

#[allow(dead_code)]
pub fn make_hogeweg_b() -> (il::LSystem, lsys::Settings) {
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

#[allow(dead_code)]
pub fn make_hogeweg_a() -> (il::LSystem, lsys::Settings) {
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

#[allow(dead_code)]
pub fn make_hogeweg_c() -> (il::LSystem, lsys::Settings) {
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

#[allow(dead_code)]
pub fn make_hogeweg_d() -> (il::LSystem, lsys::Settings) {
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

#[allow(dead_code)]
pub fn make_hogeweg_e() -> (il::LSystem, lsys::Settings) {
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

#[allow(dead_code)]
pub fn make_bush() -> (ol::LSystem, lsys::Settings) {
    let mut sys = ol::LSystem::new();

    sys.axiom = "A".to_string();
    sys.set_rule('A', "[&FL!A]>>>>>[&FL!A]>>>>>>>[&FL!A]");
    sys.set_rule('F', "S>>>>>F");
    sys.set_rule('S', "FL");
    sys.set_rule('L', "['^^{-f+f+f-|-f+f+f}]");

    let mut settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.1,
        shrink_rate: 1.5,
        iterations: 7,
        colors: vec![
            (193.0/255.0, 154.0/255.0, 107.0/255.0),
            (0.3, 1.0, 0.2),
        ],
        ..lsys::Settings::new()
    };

    settings.map_command('f', Command::Forward);

    (sys, settings)
}

#[allow(dead_code)]
pub fn make_flower() -> (ol::LSystem, lsys::Settings) {
    let mut sys = ol::LSystem::new();

    sys.axiom = "P".to_string();
    sys.set_rule('P', "I+[P+K]-->>[--L]I[++L]-[PK]++PK"); // Plant
    sys.set_rule('I', "FS[>>&&L][>>^^L]FS"); // Internode`
    sys.set_rule('S', "SFS"); // Segment
    sys.set_rule('L', "['{+F-FF-F+|+F-FF-F}]"); // Leaf
    sys.set_rule('K', "[&&&E''>W>>>>W>>>>W>>>>W>>>>W]"); // Flower
    sys.set_rule('E', "FF"); // Pedicel
    sys.set_rule('W', "['^F][&&&&{-F+F|-F+F}]"); // Wedge

    let settings = lsys::Settings {
        angle: f32::to_radians(18.0),
        width: 0.015,
        iterations: 5,
        colors: vec![
            (193.0/255.0, 154.0/255.0, 107.0/255.0),
            (0.3, 1.0, 0.2),
            (1.5, 1.5, 1.4),
            (1.5, 1.5, 0.5),
        ],
        ..lsys::Settings::new()
    };

    (sys, settings)
}

#[allow(dead_code)]
pub fn make_antenna() -> (param::LSystem, lsys::Settings) {
    let mut sys = param::LSystem::new();

    sys.axiom = param::Word::from_str("A");

    let r = 1.456;
    sys.productions = vec![
        param::Production::new(
            'A',
            vec![
                param::ProductionLetter::with_params('F', params_f![1.0]),
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
                param::ProductionLetter::with_transform('F', move |p,_| params_f![p[0].f() * r]),
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

#[allow(dead_code)]
pub fn make_anim_tree() -> (param::LSystem, lsys::Settings) {
    let mut sys = param::LSystem::new();

    sys.axiom = param::Word::from_str("#(0.01)F(0.0)>(0.593412)A(0.0)");

    let d1 = f32::to_radians(94.74);
    let d2 = f32::to_radians(132.63);
    let a = f32::to_radians(18.95);
    let lr = 1.309;
    let vr = 1.732 / 10.0;
    let ls = 0.1;

    sys.productions = vec![
        param::Production::with_condition(
            'A',
            |p| p[0].f() < 1.0,
            vec![
                param::ProductionLetter::with_transform('A', |p,dt| params_f![p[0].f() + dt]),
            ]
        ),
        param::Production::with_condition(
            'A',
            |p| p[0].f() >= 1.0,
            vec![
                param::ProductionLetter::new('['),
                param::ProductionLetter::new('L'),
                param::ProductionLetter::with_params('>', params_f![f32::to_radians(90.0)]),
                param::ProductionLetter::new('L'),
                param::ProductionLetter::with_params('>', params_f![f32::to_radians(90.0)]),
                param::ProductionLetter::new('L'),
                param::ProductionLetter::with_params('>', params_f![f32::to_radians(90.0)]),
                param::ProductionLetter::new('L'),
                param::ProductionLetter::new(']'),
                param::ProductionLetter::with_transform('#', move |_,_| params_f![vr]),
                param::ProductionLetter::with_params('F', params_f![0.0]),
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_transform('&', move |_,_| params_f![a]),
                param::ProductionLetter::with_params('F', params_f![0.0]),
                param::ProductionLetter::with_params('A', params_f![0.0]),
                param::ProductionLetter::new(']'),
                param::ProductionLetter::with_transform('>', move |_,_| params_f![d1]),
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_transform('&', move |_,_| params_f![a]),
                param::ProductionLetter::with_params('F', params_f![0.0]),
                param::ProductionLetter::with_params('A', params_f![0.0]),
                param::ProductionLetter::new(']'),
                param::ProductionLetter::with_transform('>', move |_,_| params_f![d2]),
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_transform('&', move |_,_| params_f![a]),
                param::ProductionLetter::with_params('F', params_f![0.0]),
                param::ProductionLetter::with_params('A', params_f![0.0]),
                param::ProductionLetter::new(']'),
            ]
        ),
        param::Production::new(
            'L',
            vec![
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_params('&', params_f![f32::to_radians(45.0)]),
                param::ProductionLetter::new('\''),
                param::ProductionLetter::new('{'),
                param::ProductionLetter::with_params('+', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.01]),
                param::ProductionLetter::with_params('-', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.05]),
                param::ProductionLetter::with_params('-', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.01]),
                param::ProductionLetter::with_params('+', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::new('|'),
                param::ProductionLetter::with_params('+', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.01]),
                param::ProductionLetter::with_params('-', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.05]),
                param::ProductionLetter::with_params('-', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.01]),
                param::ProductionLetter::new('}'),
                param::ProductionLetter::new(']'),
            ]
        ),
        param::Production::new(
            'f',
            vec![
                param::ProductionLetter::with_transform('f', move |p,dt| params_f![p[0].f() + (dt * ls) / p[0].f()]),
            ]
        ),
        param::Production::new(
            'F',
            vec![
                param::ProductionLetter::with_transform('F', move |p,dt| params_f![p[0].f() + dt * lr]),
            ]
        ),
        param::Production::new(
            '#',
            vec![
                param::ProductionLetter::with_transform('#', move |p,dt| params_f![p[0].f() + dt * vr]),
            ]
        ),
    ];

    let mut settings = lsys::Settings {
        width: 0.05,
        iterations: 7,
        colors: vec![
            (193.0/255.0, 154.0/255.0, 107.0/255.0),
            (0.3, 1.0, 0.2),
        ],
        ..lsys::Settings::new()
    };

    settings.command_map['f' as usize] = Command::Forward;

    (sys, settings)
}

#[allow(dead_code)]
pub fn make_straw_a() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.set_rule('F', "F[+F]F[-F]F");
    system.axiom = "F".to_string();

    let settings = lsys::Settings {
        angle: (25.7f32).to_radians(),
        iterations: 5,
        width: 0.03,
        ..lsys::Settings::new()
    };

    (system, settings)
}

#[allow(dead_code)]
pub fn make_straw_b() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.set_rule('F', "F[+F]F[-F][F]");
    system.axiom = "F".to_string();

    let settings = lsys::Settings {
        angle: (20.0f32).to_radians(),
        iterations: 5,
        width: 0.03,
        ..lsys::Settings::new()
    };

    (system, settings)
}

#[allow(dead_code)]
pub fn make_straw_c() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.set_rule('F', "FF-[-F+F+F]+[+F-F-F]");
    system.axiom = "F".to_string();

    let settings = lsys::Settings {
        angle: (22.5f32).to_radians(),
        iterations: 4,
        width: 0.03,
        ..lsys::Settings::new()
    };

    (system, settings)
}

#[allow(dead_code)]
pub fn make_straw_d() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.axiom = "X".to_string();
    system.set_rule('X', "F[+X]F[-X]+X");
    system.set_rule('F', "FF");

    let settings = lsys::Settings {
        angle: (20.0f32).to_radians(),
        iterations: 7,
        width: 0.03,
        ..lsys::Settings::new()
    };

    (system, settings)
}

#[allow(dead_code)]
pub fn make_straw_e() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.axiom = "X".to_string();
    system.set_rule('X', "F[+X][-X]FX");
    system.set_rule('F', "FF");

    let settings = lsys::Settings {
        angle: (25.7f32).to_radians(),
        iterations: 7,
        width: 0.03,
        ..lsys::Settings::new()
    };

    (system, settings)
}

#[allow(dead_code)]
pub fn make_straw_f() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.axiom = "X".to_string();
    system.set_rule('X', "F-[[X]+X]+F[+FX]-X");
    system.set_rule('F', "FF");

    let settings = lsys::Settings {
        angle: (22.5f32).to_radians(),
        iterations: 5,
        width: 0.03,
        ..lsys::Settings::new()
    };

    (system, settings)
}

#[allow(dead_code)]
pub fn make_open_raceme() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.axiom = "aA".to_string();
    system.set_rule('a', "I[+++L]I[---L]a");
    system.set_rule('A', "I[++K]I[--K]A");

    // Internode
    system.set_rule('I', "IF");

    // Leaf
    system.set_rule('L', "['>>>{+B-BB-B+|+B-BB-B}]");
    system.set_rule('B', "BF"); // Leaf border

    // Flower
    system.set_rule('K', "[E''>W>>>>W>>>>W>>>>W>>>>W]"); // Flower
    system.set_rule('E', "EF"); // Pedicel
    system.set_rule('W', "['^F][&&&&{-F+F|-F+F}]"); // Wedge

    let settings = lsys::Settings {
        angle: (18f32).to_radians(),
        step: 0.1,
        iterations: 10,
        width: 0.03,
        colors: vec![
            (193.0/255.0, 154.0/255.0, 107.0/255.0),
            (0.3, 1.0, 0.2),
            (1.5, 1.5, 1.4),
            (1.5, 1.5, 0.5),
        ],
        ..lsys::Settings::new()
    };

    (system, settings)
}
