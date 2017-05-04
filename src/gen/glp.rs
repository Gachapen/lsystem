use std::error::Error;
use std::fmt;

use na::UnitQuaternion;
use kiss3d::window::Window;
use kiss3d::camera::Camera;
use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use lsys;
use lsys::Rewriter;
use lsys::param;
use lsys::param::Param;
use lsys3d;

const LETTER_BEGIN: u8 = 65;
const LETTER_END: u8 = 75;

const LETTER_STACK_RATIO: f32 = 0.7;

const PARAM_BEGIN: f32 = 0.2;
const PARAM_END: f32 = 1.0;

const STRING_LEN_BEGIN: u32 = 2;
const STRING_LEN_END: u32 = 5;

const NUM_RULES_BEGIN: u32 = 4;
const NUM_RULES_END: u32 = 10;

#[derive(Debug)]
enum Item {
    LSystem { axiom: Box<Item>, rules: Box<Item> },
    LRule { pred: Box<Item>, succ: Box<Item> },
    LRules(Vec<Item>),
    Stack(Vec<Item>),
    Letter { letter: u8, params: Vec<Item> },
    Parameter(Option<f32>),
}

fn rand_lsystem<R: Rng>(rng: &mut R) -> Item {
    Item::LSystem {
        axiom: Box::new(rand_axiom(rng)),
        rules: Box::new(rand_rules(rng)),
    }
}

fn rand_axiom<R: Rng>(rng: &mut R) -> Item {
    Item::Stack(rand_string_stackless(rng))
}

fn rand_rules<R: Rng>(rng: &mut R) -> Item {
    let mut rules = vec![];
    let num_rules = Range::new(NUM_RULES_BEGIN, NUM_RULES_END).ind_sample(rng);

    for _ in 0..num_rules {
        rules.push(rand_rule(rng));
    }

    Item::LRules(rules)
}

fn rand_rule<R: Rng>(rng: &mut R) -> Item {
    Item::LRule {
        pred: Box::new(rand_letter_no_param(rng)),
        succ: Box::new(rand_stack(rng)),
    }
}

fn rand_string<R: Rng>(rng: &mut R) -> Vec<Item> {
    let mut items = vec![];
    let num_items = Range::new(STRING_LEN_BEGIN, STRING_LEN_END).ind_sample(rng);

    for _ in 0..num_items {
        if Range::new(0.0, 1.0).ind_sample(rng) < LETTER_STACK_RATIO {
            items.push(rand_letter(rng));
        } else {
            items.push(rand_stack(rng));
        }
    }

    items
}

fn rand_string_stackless<R: Rng>(rng: &mut R) -> Vec<Item> {
    let mut items = vec![];
    let num_items = Range::new(STRING_LEN_BEGIN, STRING_LEN_END).ind_sample(rng);

    for _ in 0..num_items {
        items.push(rand_letter(rng));
    }

    items
}

fn rand_stack<R: Rng>(rng: &mut R) -> Item {
    if Range::new(0.0, 1.0).ind_sample(rng) < 0.5 {
        Item::Stack(rand_string(rng))
    } else {
        Item::Stack(rand_string_stackless(rng))
    }
}

fn rand_letter<R: Rng>(rng: &mut R) -> Item {
    Item::Letter {
        letter: Range::new(LETTER_BEGIN, LETTER_END).ind_sample(rng),
        params: vec![rand_param(rng)],
    }
}

fn rand_param<R: Rng>(rng: &mut R) -> Item {
    Item::Parameter(Some(Range::new(PARAM_BEGIN, PARAM_END).ind_sample(rng)))
}

fn rand_letter_no_param<R: Rng>(rng: &mut R) -> Item {
    Item::Letter {
        letter: Range::new(LETTER_BEGIN, LETTER_END).ind_sample(rng),
        params: vec![Item::Parameter(None)],
    }
}

#[derive(Debug)]
struct ConvertGlpError {
    message: String,
}

impl Error for ConvertGlpError {
    fn description(&self) -> &str {
        &self.message
    }
}

impl<'a> From<&'a str> for ConvertGlpError {
    fn from(msg: &'a str) -> ConvertGlpError {
        ConvertGlpError { message: msg.to_string() }
    }
}

impl From<String> for ConvertGlpError {
    fn from(msg: String) -> ConvertGlpError {
        ConvertGlpError { message: msg }
    }
}

impl fmt::Display for ConvertGlpError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

fn parse_glp(root: &Item) -> Result<param::LSystem, ConvertGlpError> {
    match *root {
        Item::LSystem {
            ref axiom,
            ref rules,
        } => {
            Ok(param::LSystem {
                   axiom: parse_glp_stack(axiom)?,
                   productions: parse_glp_rules(rules)?,
               })
        }
        _ => Err(ConvertGlpError::from(format!("Expected LSystem, got {:?}", root))),
    }
}

fn parse_glp_stack(stack: &Item) -> Result<param::Word, ConvertGlpError> {
    match *stack {
        Item::Stack(ref items) => {
            let mut word = param::Word::new();
            for item in items {
                match *item {
                    Item::Letter { .. } => {
                        word.push(parse_glp_letter(item)?);
                    }
                    Item::Stack(_) => {
                        word.push(param::Letter::new('['));
                        word.extend(parse_glp_stack(item)?);
                        word.push(param::Letter::new(']'));
                    }
                    _ => {
                        let msg = format!("Expected Letter or Stack, got {:?}", item);
                        return Err(ConvertGlpError::from(msg));
                    }
                }
            }

            Ok(word)
        }
        _ => Err(ConvertGlpError::from(format!("Expected Stack, got {:?}", stack))),
    }
}

fn parse_glp_rules(rules: &Item) -> Result<Vec<param::Production>, ConvertGlpError> {
    match *rules {
        Item::LRules(ref rules) => {
            let mut productions = vec![];
            for rule in rules {
                productions.push(parse_glp_rule(rule)?);
            }

            Ok(productions)
        }
        _ => Err(ConvertGlpError::from(format!("Expected LRules, got {:?}", rules))),
    }
}

fn parse_glp_rule(rule: &Item) -> Result<param::Production, ConvertGlpError> {
    match *rule {
        Item::LRule { ref pred, ref succ } => {
            Ok(param::Production::new(parse_glp_letter(&*pred)?.character as char,
                                      parse_glp_stack(&*succ)?
                                          .iter()
                                          .map(param::ProductionLetter::from)
                                          .collect()))
        }
        _ => Err(ConvertGlpError::from(format!("Expected LRule, got {:?}", rule))),
    }
}

fn parse_glp_letter(letter: &Item) -> Result<param::Letter, ConvertGlpError> {
    match *letter {
        Item::Letter {
            ref letter,
            ref params,
        } => {
            Ok(param::Letter {
                   character: *letter,
                   params: parse_glp_params(params)?
                       .iter()
                       .filter_map(|p| *p)
                       .map(Param::F)
                       .collect(),
               })
        }
        _ => Err(ConvertGlpError::from(format!("Expected Letter, got {:?}", letter))),
    }
}

fn parse_glp_params(params: &[Item]) -> Result<Vec<Option<f32>>, ConvertGlpError> {
    let mut parsed_params = vec![];
    for param in params {
        parsed_params.push(parse_glp_param(param)?);
    }
    Ok(parsed_params)
}

fn parse_glp_param(param: &Item) -> Result<Option<f32>, ConvertGlpError> {
    match *param {
        Item::Parameter(p) => Ok(p),
        _ => Err(ConvertGlpError::from(format!("Expected Parameter, got {:?}", param))),
    }
}

pub fn run_generated(window: &mut Window, camera: &mut Camera) {
    let mut rng = rand::thread_rng();
    let root = rand_lsystem(&mut rng);

    let mut settings = lsys::Settings {
        width: 0.05,
        iterations: 7,
        ..lsys::Settings::new()
    };

    let system = match parse_glp(&root) {
        Ok(system) => system,
        Err(error) => panic!("Failed converting GLP: {}", error.description()),
    };

    println!("{}", system);

    settings.command_map[LETTER_BEGIN as usize] = lsys::Command::Forward;
    settings.command_map[(LETTER_BEGIN + 1) as usize] = lsys::Command::YawLeft;
    settings.command_map[(LETTER_BEGIN + 2) as usize] = lsys::Command::YawRight;
    settings.command_map[(LETTER_BEGIN + 3) as usize] = lsys::Command::RollLeft;
    settings.command_map[(LETTER_BEGIN + 4) as usize] = lsys::Command::RollRight;
    settings.command_map[(LETTER_BEGIN + 5) as usize] = lsys::Command::PitchUp;
    settings.command_map[(LETTER_BEGIN + 6) as usize] = lsys::Command::PitchDown;

    let instructions = system.instructions(settings.iterations, &settings.command_map);

    let mut model = lsys3d::build_model(instructions, &settings);
    window.scene_mut().add_child(model.clone());

    while window.render_with_camera(camera) {
        model.append_rotation(&UnitQuaternion::from_euler_angles(0.0f32, 0.004, 0.0));
    }
}
