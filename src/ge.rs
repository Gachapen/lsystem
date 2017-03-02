use std::f32;
use rand;
use rand::distributions::{IndependentSample, Range};
use kiss3d::window::Window;
use kiss3d::camera::Camera;
use na::UnitQuaternion;

use abnf;
use abnf::expand::{SelectionStrategy, expand_list};
use lsys;
use lsys::ol;
use lsys::Rewriter;
use lsys3d;

pub fn run_ge(window: &mut Window, camera: &mut Camera) {
    let lsys_abnf = abnf::parse_file("lsys.abnf").expect("Could not parse ABNF file");
    //println!("{:#?}", lsys_abnf);

    let mut rng = rand::thread_rng();
    let gene_range = Range::new(u8::min_value(), u8::max_value());
    let gene_length = 100;

    let mut genes = Vec::with_capacity(gene_length);
    for _ in 0..gene_length {
        genes.push(gene_range.ind_sample(&mut rng));
    }

    let mut genotype = Genotype::new(genes);

    //let axiom = expand_list(&lsys_abnf["axiom"], 0, &lsys_abnf, &mut genotype);
    //let productions = expand_list(&lsys_abnf["productions"], 0, &lsys_abnf, &mut genotype);
    //println!("genotype: {:?}", genotype.genes);
    //println!("axiom: {}", axiom);
    //println!("productions:\n{}", productions);

    let settings = lsys::Settings {
        width: 0.05,
        angle: f32::consts::PI / 8.0,
        iterations: 3,
        ..lsys::Settings::new()
    };

    let mut system = ol::LSystem {
        axiom: expand_list(&lsys_abnf["axiom"], 0, &lsys_abnf, &mut genotype),
        rules: expand_productions(&lsys_abnf["productions"], 0, &lsys_abnf, &mut genotype),
        ..ol::LSystem::new()
    };

    system.remove_redundancy();

    println!("Commands:");
    println!("{:?}", system.command_map['+' as usize]);
    println!("");
    println!("LSystem:");
    println!("{}", system);
    println!("");
    println!("Rewritten: {}", system.rewrite(settings.iterations));

    let instructions = system.instructions(settings.iterations);

    let mut model = lsys3d::build_model(&instructions, &settings);
    window.scene_mut().add_child(model.clone());

    while window.render_with_camera(camera) {
        model.append_rotation(&UnitQuaternion::from_euler_angles(0.0f32, 0.004, 0.0));
    }
}

struct Genotype {
    genes: Vec<u8>,
    index: usize,
}

impl Genotype {
    fn new(genes: Vec<u8>) -> Genotype {
        Genotype {
            genes: genes,
            index: 0,
        }
    }

    fn use_next_gene(&mut self) -> u8 {
        assert!(self.index < self.genes.len(), "Genotype index overflows gene list");

        let gene = self.genes[self.index];
        self.index = (self.index + 1) % self.genes.len();

        gene
    }
}

impl SelectionStrategy for Genotype {
    fn select_alternative(&mut self, num: usize) -> usize {
        let max_value = u8::max_value() as usize;
        let num = num % max_value;

        self.use_next_gene() as usize % num
    }

    fn select_repetition(&mut self, min: u32, max: u32) -> u32 {
        let max_value = u8::max_value() as u32;
        let range = (max - min + 1) % max_value;

        (self.use_next_gene() as u32 % range) + min
    }
}

// Somehow make the below expansion functions a genetic part of abnf::expand?

fn expand_productions<T>(list: &abnf::List, depth: u32, grammar: &abnf::Ruleset, strategy: &mut T) -> ol::RuleMap
    where T: SelectionStrategy
{
    let mut rules = ol::create_rule_map();

    if let abnf::List::Sequence(ref seq) = *list {
        assert!(seq.len() == 1);
        let item = &seq[0];

        if let abnf::Content::Symbol(ref prod_sym) = item.content {
            let repeat = item.repeat.unwrap_or_default();
            let min = repeat.min.unwrap_or(0);
            let max = repeat.max.unwrap_or(u32::max_value());
            let num = strategy.select_repetition(min, max);

            for _ in 0..num {
                let (pred, succ) = expand_production(&grammar[prod_sym], depth + 1, grammar, strategy);
                rules[pred as usize] = succ;
            }
        }
    }

    rules
}

fn expand_production<T>(list: &abnf::List, depth: u32, grammar: &abnf::Ruleset, strategy: &mut T) -> (char, String)
    where T: SelectionStrategy
{
    if let abnf::List::Sequence(ref seq) = *list {
        assert!(seq.len() == 2);
        let pred_item = &seq[0];
        let succ_item = &seq[1];

        let pred = if let abnf::Content::Symbol(ref pred_sym) = pred_item.content {
            expand_predicate(&grammar[pred_sym], depth + 1, grammar, strategy)
        } else {
            0 as char
        };

        let succ = if let abnf::Content::Symbol(ref succ_sym) = succ_item.content {
            expand_successor(&grammar[succ_sym], depth + 1, grammar, strategy)
        } else {
            String::new()
        };

        (pred, succ)
    } else {
        (0 as char, String::new())
    }
}

fn expand_predicate<T>(list: &abnf::List, depth: u32, grammar: &abnf::Ruleset, strategy: &mut T) -> char
    where T: SelectionStrategy
{
    let value = expand_list(list, depth, grammar, strategy);
    assert!(value.len() == 1);
    value.as_bytes()[0] as char
}

fn expand_successor<T>(list: &abnf::List, depth: u32, grammar: &abnf::Ruleset, strategy: &mut T) -> String
    where T: SelectionStrategy
{
    expand_list(list, depth, grammar, strategy)
}
