use na::Vector3;
use kiss3d::window::Window;
use kiss3d::camera::Camera;
use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use lsys::Rewriter;
use lsys3d;
use lsystems;

#[derive(Debug)]
enum Item {
    LSystem{ axiom: Box<Item>, rules: Box<Item> },
    LRule{ pred: Box<Item>, succ: Box<Item> },
    Predecessor(Box<Item>),
    Successor(Vec<Item>),
    LRules(Vec<Item>),
    Axiom(Vec<Item>),
    Stack(Vec<Item>),
    Letter{ letter: u8, params: Vec<Item> },
    Parameter(Option<f32>)
}

fn rand_lsystem<R: Rng>(rng: &mut R) -> Item {
    Item::LSystem{
        axiom: Box::new(rand_axiom(rng)),
        rules: Box::new(rand_rules(rng)),
    }
}

fn rand_axiom<R: Rng>(rng: &mut R) -> Item {
    Item::Axiom(rand_string(rng))
}

fn rand_rules<R: Rng>(rng: &mut R) -> Item {
    let mut rules = vec![];
    let num_rules = Range::new(1, 5).ind_sample(rng);

    for _ in 0..num_rules {
        rules.push(rand_rule(rng));
    }

    Item::LRules(rules)
}

fn rand_rule<R: Rng>(rng: &mut R) -> Item {
    Item::LRule {
        pred: Box::new(rand_pred(rng)),
        succ: Box::new(rand_succ(rng)),
    }
}

fn rand_pred<R: Rng>(rng: &mut R) -> Item {
    Item::Predecessor(Box::new(rand_letter_no_param(rng)))
}

fn rand_succ<R: Rng>(rng: &mut R) -> Item {
    Item::Successor(rand_string(rng))
}

fn rand_string<R: Rng>(rng: &mut R) -> Vec<Item> {
    let mut items = vec![];
    let num_items = Range::new(1, 5).ind_sample(rng);
    let letter_stack_ratio = 0.7;

    for _ in 0..num_items {
        if Range::new(0.0, 1.0).ind_sample(rng) < letter_stack_ratio {
            items.push(rand_letter(rng));
        } else {
            items.push(rand_stack(rng));
        }
    }

    items
}

fn rand_string_stackless<R: Rng>(rng: &mut R) -> Vec<Item> {
    let mut items = vec![];
    let num_items = Range::new(1, 5).ind_sample(rng);

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
    Item::Letter{
        letter: Range::new(0, 128).ind_sample(rng),
        params: vec![rand_param(rng)],
    }
}

fn rand_param<R: Rng>(rng: &mut R) -> Item {
    Item::Parameter(Some(Range::new(0.0, 1.0).ind_sample(rng)))
}

fn rand_letter_no_param<R: Rng>(rng: &mut R) -> Item {
    Item::Letter{
        letter: Range::new(0, 128).ind_sample(rng),
        params: vec![Item::Parameter(None)],
    }
}

#[allow(dead_code, unused_variables)]
pub fn run_generated(window: &mut Window, camera: &mut Camera)
{
    let mut rng = rand::thread_rng();
    let root = rand_lsystem(&mut rng);
    println!("{:#?}", root);
}
