use std::f32;
use rand;
use rand::distributions::{IndependentSample, Range};
use kiss3d::window::Window;
use kiss3d::camera::Camera;
use na::UnitQuaternion;

use abnf;
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

    let mut genes = vec![];
    genes.reserve(gene_length);
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

trait AbnfSelectionStrategy {
    fn select_alternative(&mut self, num: usize) -> usize;
    fn select_repetition(&mut self, min: u32, max: u32) -> u32;
}

struct Genotype {
    genes: Vec<u8>,
    index: usize,
}

const MAX_DEPTH: u32 = 20;

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

impl AbnfSelectionStrategy for Genotype {
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

fn expand_list<T>(list: &abnf::List, depth: u32, grammar: &abnf::Ruleset, strategy: &mut T) -> String
    where T: AbnfSelectionStrategy
{
    match *list {
        abnf::List::Sequence(ref sequence) => {
            let mut string = String::new();

            for item in sequence {
                string.push_str(&expand_item(item, depth + 1, grammar, strategy));
            }

            string
        },
        abnf::List::Alternatives(ref alternatives) => {
            let index = strategy.select_alternative(alternatives.len());
            expand_item(&alternatives[index], depth + 1, grammar, strategy)
        },
    }
}

fn expand_item<T>(item: &abnf::Item, depth: u32, grammar: &abnf::Ruleset, strategy: &mut T) -> String
    where T: AbnfSelectionStrategy
{
    if depth > MAX_DEPTH {
        return String::new();
    }

    let times = match item.repeat {
        Some(ref repeat) => {
            let min = repeat.min.unwrap_or(0);
            let max = repeat.max.unwrap_or(u32::max_value());

            strategy.select_repetition(min, max)
        },
        None => 1,
    };

    let mut string = String::new();

    for _ in 0..times {
        let expanded = match item.content {
            abnf::Content::Value(ref value) => {
                value.clone()
            },
            abnf::Content::Symbol(ref symbol) => {
                expand_list(&grammar[symbol], depth, grammar, strategy)
            },
            abnf::Content::Group(ref group) => {
                expand_list(group, depth, grammar, strategy)
            },
            abnf::Content::Core(rule) => {
                let content = abnf::expand_core_rule(rule);
                expand_item(&abnf::Item::new(content), depth + 1, grammar, strategy)
            },
            abnf::Content::Range(min, max) => {
                let index = strategy.select_alternative(max as usize - min as usize);
                let character = (index + min as usize) as u8 as char;

                // Probably not the most efficient...
                let mut string = String::new();
                string.push(character);
                string
            }
        };

        string.push_str(&expanded);
    };

    string
}

fn expand_productions<T>(list: &abnf::List, depth: u32, grammar: &abnf::Ruleset, strategy: &mut T) -> ol::RuleMap
    where T: AbnfSelectionStrategy
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
    where T: AbnfSelectionStrategy
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
    where T: AbnfSelectionStrategy
{
    let value = expand_list(list, depth, grammar, strategy);
    assert!(value.len() == 1);
    value.as_bytes()[0] as char
}

fn expand_successor<T>(list: &abnf::List, depth: u32, grammar: &abnf::Ruleset, strategy: &mut T) -> String
    where T: AbnfSelectionStrategy
{
    expand_list(list, depth, grammar, strategy)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyStrategy {}

    impl AbnfSelectionStrategy for DummyStrategy {
        #[allow(unused_variables)]
        fn select_alternative(&mut self, num: usize) -> usize {
            0
        }

        #[allow(unused_variables)]
        fn select_repetition(&mut self, min: u32, max: u32) -> u32 {
            1
        }
    }

    #[test]
    fn test_expand_value() {
        let rules = abnf::Ruleset::new();
        let mut strategy = DummyStrategy {};
        let item = abnf::Item::new(abnf::Content::Value("value".to_string()));

        assert_eq!(expand_item(&item, 0, &rules, &mut strategy), "value".to_string());
    }

    #[test]
    fn test_expand_symbol() {
        let mut rules = abnf::Ruleset::new();
        rules.insert(
            "symbol".to_string(),
            abnf::List::Sequence(vec![
                abnf::Item::new(abnf::Content::Value("value".to_string())),
            ])
        );

        let mut strategy = DummyStrategy {};
        let item = abnf::Item::new(abnf::Content::Symbol("symbol".to_string()));

        assert_eq!(expand_item(&item, 0, &rules, &mut strategy), "value".to_string());
    }

    #[test]
    fn test_expand_core() {
        let rules = abnf::Ruleset::new();
        let mut strategy = DummyStrategy {};
        let item = abnf::Item::new(abnf::Content::Core(abnf::CoreRule::Alpha));

        assert_eq!(expand_item(&item, 0, &rules, &mut strategy), "A".to_string());
    }

    #[test]
    fn test_expand_range() {
        let rules = abnf::Ruleset::new();
        let mut strategy = DummyStrategy {};
        let item = abnf::Item::new(abnf::Content::Range('X', 'Z'));

        assert_eq!(expand_item(&item, 0, &rules, &mut strategy), "X".to_string());
    }

    #[test]
    fn test_expand_sequence() {
        let rules = abnf::Ruleset::new();
        let mut strategy = DummyStrategy {};
        let list = abnf::List::Sequence(vec![
            abnf::Item::new(abnf::Content::Value("value".to_string())),
            abnf::Item::new(abnf::Content::Value("value".to_string())),
        ]);

        assert_eq!(expand_list(&list, 0, &rules, &mut strategy), "valuevalue".to_string());
    }

    #[test]
    fn test_expand_alternatives() {
        let rules = abnf::Ruleset::new();
        let mut strategy = DummyStrategy {};
        let list = abnf::List::Alternatives(vec![
            abnf::Item::new(abnf::Content::Value("one".to_string())),
            abnf::Item::new(abnf::Content::Value("two".to_string())),
        ]);

        assert_eq!(expand_list(&list, 0, &rules, &mut strategy), "one".to_string());
    }

    #[test]
    fn test_expand_group() {
        let rules = abnf::Ruleset::new();
        let mut strategy = DummyStrategy {};
        let item = abnf::Item::new(abnf::Content::Group(
            abnf::List::Sequence(vec![
                abnf::Item::new(abnf::Content::Value("value".to_string())),
                abnf::Item::new(abnf::Content::Value("value".to_string())),
            ])
        ));

        assert_eq!(expand_item(&item, 0, &rules, &mut strategy), "valuevalue".to_string());
    }
}
