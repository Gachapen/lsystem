use std::f32;
use rand;
use rand::distributions::{IndependentSample, Range};
use kiss3d::window::Window;
use kiss3d::camera::Camera;
use na::UnitQuaternion;

use abnf;
use abnf::expand::{SelectionStrategy, expand_grammar, expand_list};
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
        axiom: expand_grammar(&lsys_abnf, "axiom", &mut genotype),
        rules: expand_productions(&lsys_abnf, "productions", &mut genotype),
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
}

// Somehow make the below expansion functions a genetic part of abnf::expand?

fn expand_productions<T>(grammar: &abnf::Ruleset, root: &str, strategy: &mut T) -> ol::RuleMap
    where T: SelectionStrategy
{
    let mut rules = ol::create_rule_map();
    let list = &grammar[root];

    if let abnf::List::Sequence(ref seq) = *list {
        assert!(seq.len() == 1);
        let item = &seq[0];

        if let abnf::Content::Symbol(ref prod_sym) = item.content {
            let repeat = item.repeat.unwrap_or_default();
            let min = repeat.min.unwrap_or(0);
            let max = repeat.max.unwrap_or(u32::max_value());
            let num = strategy.select_repetition(min, max);

            for _ in 0..num {
                let (pred, succ) = expand_production(&grammar[prod_sym], 1, grammar, strategy);
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

fn infer_selections(expanded: &str, grammar: &abnf::Ruleset, root: &str) -> Result<Vec<usize>, String> {
    let selection = infer_list_selections(&grammar[root], 0, expanded, grammar);

    match selection {
        Ok((list, index)) => {
            if index == expanded.len() {
                Ok(list)
            } else {
                Err(format!("Expanded string does not fully match grammar. The first {} characters matched", index))
            }
        },
        Err(_) => {
            Err("Expanded string does not match grammar".to_string())
        },
    }
}

// TODO: Need to be able to try new non-tested alternatives/repetitions if a previously matched
// alternative/repetition results in a mismatch later.
fn infer_list_selections(list: &abnf::List, mut index: usize, expanded: &str, grammar: &abnf::Ruleset) -> Result<(Vec<usize>, usize), ()> {
    use abnf::List;

    match *list {
        List::Sequence(ref sequence) => {
            let mut selections = vec![];

            for item in sequence {
                let (item_selections, updated_index) = infer_item_selections(item, index, expanded, grammar)?;
                index = updated_index;
                selections.extend(item_selections);
            }

            Ok((selections, index))
        },
        List::Alternatives(ref alternatives) => {
            let mut selections = Vec::with_capacity(1);

            for (alternative, item) in alternatives.iter().enumerate() {
                if let Ok((item_selections, updated_index)) = infer_item_selections(item, index, expanded, grammar) {
                    selections.push(alternative);
                    selections.extend(item_selections);
                    index = updated_index;

                    break;
                }
            }

            if selections.is_empty() {
                Err(())
            } else {
                Ok((selections, index))
            }
        },
    }
}

// TODO: Need to be able to try new non-tested alternatives/repetitions if a previously matched
// alternative/repetition results in a mismatch later.
fn infer_item_selections(item: &abnf::Item, mut index: usize, expanded: &str, grammar: &abnf::Ruleset) -> Result<(Vec<usize>, usize), ()> {
    use abnf::Content;
    use abnf::Item;

    let repeat = match item.repeat {
        Some(ref repeat) => {
            let min = repeat.min.unwrap_or(0);
            let max = repeat.max.unwrap_or(u32::max_value());

            Some((min, max))
        },
        None => None,
    };

    let (min_repeat, max_repeat) = match repeat {
        Some((min, max)) => (min, max),
        None => (1, 1),
    };

    let mut selections = vec![];
    let mut matched = false;
    let mut times_repeated = 0;

    for _ in 0..max_repeat {
        matched = false;

        match item.content {
            Content::Value(ref value) => {
                let index_end = index + value.len();
                if *value.as_str() == expanded[index..index_end] {
                    index += value.len();
                    matched = true;
                }
            },
            Content::Symbol(ref symbol) => {
                let child_result = infer_list_selections(&grammar[symbol], index, expanded, grammar);
                if let Ok((child_selections, child_index)) = child_result {
                    selections.extend(child_selections);
                    index = child_index;
                    matched = true;
                }
            },
            Content::Group(ref group) => {
                let child_result = infer_list_selections(group, index, expanded, grammar);
                if let Ok((child_selections, child_index)) = child_result {
                    selections.extend(child_selections);
                    index = child_index;
                    matched = true;
                }
            },
            Content::Core(rule) => {
                let content = abnf::expand::expand_core_rule(rule);
                let child_result = infer_item_selections(&Item::new(content), index, expanded, grammar);
                if let Ok((child_selections, child_index)) = child_result {
                    selections.extend(child_selections);
                    index = child_index;
                    matched = true;
                }
            },
            Content::Range(_, _) => {
                panic!("Content::Range not implemented for infer_item_selections");
                //let index = strategy.select_alternative(max as usize - min as usize);
                //let character = (index + min as usize) as u8 as char;
            }
        };

        if !matched {
            break;
        } else {
            times_repeated += 1;

            if index == expanded.len() {
                break;
            }
        }
    }

    if matched || times_repeated >= min_repeat {
        if repeat.is_some() {
            selections.insert(0, (times_repeated - min_repeat) as usize);
        }

        Ok((selections, index))
    } else {
        Err(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use abnf::{Item, Content, Ruleset, List, Repeat};

    #[test]
    fn test_infer_selections_value() {
        let item = Item::new(Content::Value("value".to_string()));

        let mut grammar = Ruleset::new();
        grammar.insert(
            "symbol".to_string(),
            List::Sequence(vec![
                item.clone(),
            ])
        );

        assert_eq!(
            infer_item_selections(&item, 0, "value", &grammar),
            Ok((vec![], 5))
        );
    }

    #[test]
    fn test_infer_selections_repeat_limits() {
        let item = Item::repeated(
            Content::Value("value".to_string()),
            Repeat::with_limits(2, 4)
        );

        let mut grammar = Ruleset::new();
        grammar.insert(
            "symbol".to_string(),
            List::Sequence(vec![
                item.clone(),
            ])
        );

        assert_eq!(
            infer_item_selections(&item, 0, "valuevaluevalue", &grammar),
            Ok((vec![1], 15))
        );
    }

    #[test]
    fn test_infer_selections_alternatives() {
        let list = List::Alternatives(vec![
            Item::new(Content::Value("1".to_string())),
            Item::new(Content::Value("2".to_string())),
            Item::new(Content::Value("3".to_string())),
        ]);

        let mut grammar = Ruleset::new();
        grammar.insert(
            "symbol".to_string(),
            list.clone()
        );

        assert_eq!(
            infer_list_selections(&list, 0, "2", &grammar),
            Ok((vec![1], 1))
        );
    }

    #[test]
    fn test_infer_selections_match() {
        let item = Item::new(Content::Value("value".to_string()));

        let mut grammar = Ruleset::new();
        grammar.insert(
            "symbol".to_string(),
            List::Sequence(vec![
                item.clone(),
            ])
        );

        assert_eq!(
            infer_selections("value", &grammar, "symbol"),
            Ok(vec![])
        );
    }

    #[test]
    fn test_infer_selections_match_alternatives() {
        let list = List::Alternatives(vec![
            Item::new(Content::Value("1".to_string())),
            Item::new(Content::Value("2".to_string())),
            Item::new(Content::Value("3".to_string())),
        ]);

        let mut grammar = Ruleset::new();
        grammar.insert(
            "symbol".to_string(),
            list.clone()
        );

        assert_eq!(
            infer_selections("2", &grammar, "symbol"),
            Ok(vec![1])
        );
    }

    #[test]
    fn test_infer_selections_mismatch() {
        let item = Item::new(Content::Value("value".to_string()));

        let mut grammar = Ruleset::new();
        grammar.insert(
            "symbol".to_string(),
            List::Sequence(vec![
                item.clone(),
            ])
        );

        assert_eq!(
            infer_selections("notvalue", &grammar, "symbol"),
            Err("Expanded string does not match grammar".to_string())
        );
    }

    #[test]
    fn test_infer_selections_missing() {
        let item = Item::new(Content::Value("value".to_string()));

        let mut grammar = Ruleset::new();
        grammar.insert(
            "symbol".to_string(),
            List::Sequence(vec![
                item.clone(),
            ])
        );

        assert_eq!(
            infer_selections("valueextra", &grammar, "symbol"),
            Err("Expanded string does not fully match grammar. The first 5 characters matched".to_string())
        );
    }
}
