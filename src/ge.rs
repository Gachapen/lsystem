use std::f32::consts::PI;
use std::{cmp, fs};
use std::collections::HashMap;
use std::io::{BufWriter, BufReader};
use std::fs::File;
use std::path::Path;

use rand;
use rand::distributions::{IndependentSample, Range};
use rand::Rng;
use kiss3d::window::Window;
use kiss3d::camera::{Camera, ArcBall};
use na::{UnitQuaternion, Point3};
use num;
use num::{Unsigned, NumCast};
use serde_yaml;
use time;
use glfw::{Key, WindowEvent, Action};

use abnf;
use abnf::expand::{SelectionStrategy, expand_grammar};
use lsys;
use lsys::ol;
use lsys::Rewriter;
use lsys3d;
use lsystems;

#[allow(dead_code, unused_variables)]
pub fn run_ge(window: &mut Window, camera: &mut Camera) {
    //run_print_abnf();
    //run_random_genes(window);
    run_with_distribution(window);
    //run_bush_inferred(window, camera);
}

type GenePrimitive = u32;

#[allow(dead_code)]
fn generate_genome< R: Rng>(rng: &mut R, len: usize) -> Vec<GenePrimitive> {
    let gene_range = Range::new(GenePrimitive::min_value(), GenePrimitive::max_value());

    let mut genes = Vec::with_capacity(len);
    for _ in 0..len {
        genes.push(gene_range.ind_sample(rng));
    }

    genes
}

#[allow(dead_code)]
fn generate_system<G>(grammar: &abnf::Ruleset, genotype: &mut G) -> ol::LSystem
    where G: SelectionStrategy
{
    let mut system = ol::LSystem {
        axiom: expand_grammar(grammar, "axiom", genotype),
        productions: expand_productions(grammar, genotype),
    };

    system.remove_redundancy();

    system
}

#[allow(dead_code)]
fn is_nothing(lsystem: &ol::LSystem) -> bool {
    // 'F' is rewritten as something not containing 'F'.
    // Technically it can still be something if the iteration stops right after something has been
    // expanded to 'F'. But it would still only be one to many single-length lines, and only at
    // that number of iterations.
    if !lsystem.productions['F'].as_bytes().iter().any(|s| *s == 'F' as u32 as u8) {
        return true;
    }

    let mut visited = Vec::new();
    let mut visit_stack = Vec::new();

    // If some symbol in the axiom is 'F', then it draws something.
    for symbol in lsystem.axiom.as_bytes() {
        if *symbol as char == 'F' {
            return false;
        } else if !visited.iter().any(|s| *s == *symbol) {
            visited.push(*symbol);
            visit_stack.push(*symbol);
        }
    }

    // If some symbol in the used productions is 'F', then it draws something.
    while !visit_stack.is_empty() {
        let predicate = visit_stack.pop().unwrap();
        let string = &lsystem.productions[predicate];

        for symbol in string.as_bytes() {
            if *symbol == 'F' as u32 as u8 {
                return false;
            } else if !visited.iter().any(|s| *s == *symbol) {
                visited.push(*symbol);
                visit_stack.push(*symbol);
            }
        }
    }

    true
}

#[allow(dead_code)]
fn run_with_distribution(window: &mut Window) {
    let grammar = abnf::parse_file("lsys.abnf").expect("Could not parse ABNF file");
    let genome_length = 100;

    let mut rng = rand::thread_rng();

    let mut genotype = {
        let mut distribution = Distribution::new();
        distribution.set_default_weights("productions", 0, &[0.0, 1.0]);
        distribution.set_default_weights("string", 0, &[1.0, 2.0, 2.0, 2.0, 1.0, 1.0]);
        distribution.set_default_weights("string", 1, &[1.0, 0.0]);

        distribution.set_weights(0, "string", 0, &[1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
        distribution.set_weights(0, "string", 1, &[1.0, 1.0]);

        distribution.set_weights(1, "string", 1, &[10.0, 1.0]);

        WeightedGenotype::new(vec![], distribution)
    };

    let settings = lsys::Settings {
        width: 0.05,
        angle: PI / 8.0,
        iterations: 5,
        ..lsys::Settings::new()
    };

    let mut system = ol::LSystem::new();
    while is_nothing(&system) {
        genotype.genotype.genes = generate_genome(&mut rng, genome_length);
        system = generate_system(&grammar, &mut genotype);
    }

    println!("Genes: {:?}", genotype.genotype.genes);
    println!("");

    println!("LSystem:");
    println!("{}", system);

    let mut model = {
        let instructions = system.instructions(settings.iterations, &settings.command_map);
        lsys3d::build_model(&instructions, &settings)
    };

    window.scene_mut().add_child(model.clone());

    let mut camera = {
        let eye = Point3::new(0.0, 0.0, 5.0);
        let at = Point3::new(0.0, 1.0, 0.0);
        ArcBall::new(eye, at)
    };

    let model_dir = Path::new("model");
    fs::create_dir_all(model_dir).unwrap();
    let mut model_index = 0;

    while window.render_with_camera(&mut camera) {
        model.append_rotation(&UnitQuaternion::from_euler_angles(0.0f32, 0.004, 0.0));

        for event in window.events().iter() {
            match event.value {
                WindowEvent::Key(Key::S, _, Action::Release, _) => {
                    let filename = format!("{}.yaml", time::now().rfc3339());
                    let path = model_dir.join(filename);

                    let file = File::create(&path).unwrap();
                    serde_yaml::to_writer(&mut BufWriter::new(file), &system).unwrap();

                    println!("Saved to {}", path.to_str().unwrap());
                },
                WindowEvent::Key(Key::Space, _, Action::Release, _) => {
                    let genes = generate_genome(&mut rng, 100);

                    println!("Genes: {:?}", genes);
                    println!("");

                    system = ol::LSystem::new();
                    while is_nothing(&system) {
                        genotype.genotype.genes = generate_genome(&mut rng, genome_length);
                        system = generate_system(&grammar, &mut genotype);
                    }

                    println!("LSystem:");
                    println!("{}", system);

                    let instructions = system.instructions(settings.iterations, &settings.command_map);
                    window.remove(&mut model);
                    model = lsys3d::build_model(&instructions, &settings);
                    window.scene_mut().add_child(model.clone());

                    model_index = 0;
                },
                WindowEvent::Key(Key::L, _, Action::Release, _) => {
                    let mut models = fs::read_dir(model_dir).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>();
                    models.sort();
                    let models = models;

                    if model_index >= models.len() {
                        model_index = 0;
                    }

                    if !models.is_empty() {
                        let path = &models[model_index];
                        let file = File::open(&path).unwrap();
                        system = serde_yaml::from_reader(&mut BufReader::new(file)).unwrap();

                        println!("Loaded {}", path.to_str().unwrap());

                        println!("LSystem:");
                        println!("{}", system);

                        let instructions = system.instructions(settings.iterations, &settings.command_map);
                        window.remove(&mut model);
                        model = lsys3d::build_model(&instructions, &settings);
                        window.scene_mut().add_child(model.clone());
                    }

                    model_index += 1;
                },
                _ => {}
            }
        }
    }
}

#[allow(dead_code)]
fn run_print_abnf() {
    let lsys_abnf = abnf::parse_file("lsys.abnf").expect("Could not parse ABNF file");
    println!("{:#?}", lsys_abnf);
}

#[allow(dead_code)]
fn run_random_genes(window: &mut Window) {
    let lsys_abnf = abnf::parse_file("lsys.abnf").expect("Could not parse ABNF file");

    let mut genotype = {
        let mut rng = rand::thread_rng();
        let genome = generate_genome(&mut rng, 100);
        Genotype::new(genome)
    };

    println!("Genes: {:?}", genotype.genes);
    println!("");

    let settings = lsys::Settings {
        width: 0.05,
        angle: PI / 8.0,
        iterations: 5,
        ..lsys::Settings::new()
    };

    let mut system = ol::LSystem {
        axiom: expand_grammar(&lsys_abnf, "axiom", &mut genotype),
        productions: expand_productions(&lsys_abnf, &mut genotype),
    };

    system.remove_redundancy();

    println!("LSystem:");
    println!("{}", system);

    let instructions = system.instructions(settings.iterations, &settings.command_map);

    let mut model = lsys3d::build_model(&instructions, &settings);
    window.scene_mut().add_child(model.clone());

    let mut camera = {
        let eye = Point3::new(0.0, 0.0, 5.0);
        let at = Point3::new(0.0, 1.0, 0.0);
        ArcBall::new(eye, at)
    };

    while window.render_with_camera(&mut camera) {
        model.append_rotation(&UnitQuaternion::from_euler_angles(0.0f32, 0.004, 0.0));
    }
}

#[allow(dead_code)]
fn run_bush_inferred(window: &mut Window, camera: &mut Camera) {
    let lsys_abnf = abnf::parse_file("bush.abnf").expect("Could not parse ABNF file");

    let (system, settings) = {
        let (system, mut settings) = lsystems::make_bush();

        let axiom_gen = infer_selections(&system.axiom, &lsys_abnf, "axiom").unwrap();
        let mut axiom_geno = Genotype::new(axiom_gen.iter().map(|g| *g as u8).collect());

        let a_gen = infer_selections(&system.productions['A'], &lsys_abnf, "successor").unwrap();
        let mut a_geno = Genotype::new(a_gen.iter().map(|g| *g as u8).collect());

        let f_gen = infer_selections(&system.productions['F'], &lsys_abnf, "successor").unwrap();
        let mut f_geno = Genotype::new(f_gen.iter().map(|g| *g as u8).collect());

        let s_gen = infer_selections(&system.productions['S'], &lsys_abnf, "successor").unwrap();
        let mut s_geno = Genotype::new(s_gen.iter().map(|g| *g as u8).collect());

        let l_gen = infer_selections(&system.productions['L'], &lsys_abnf, "successor").unwrap();
        let mut l_geno = Genotype::new(l_gen.iter().map(|g| *g as u8).collect());

        let mut new_system = ol::LSystem {
            axiom: expand_grammar(&lsys_abnf, "axiom", &mut axiom_geno),
            ..ol::LSystem::new()
        };

        new_system.set_rule('A', &expand_grammar(&lsys_abnf, "successor", &mut a_geno));
        new_system.set_rule('F', &expand_grammar(&lsys_abnf, "successor", &mut f_geno));
        new_system.set_rule('S', &expand_grammar(&lsys_abnf, "successor", &mut s_geno));
        new_system.set_rule('L', &expand_grammar(&lsys_abnf, "successor", &mut l_geno));

        settings.map_command('f', lsys::Command::Forward);

        (new_system, settings)
    };

    let instructions = system.instructions(settings.iterations, &settings.command_map);

    let mut model = lsys3d::build_model(&instructions, &settings);
    window.scene_mut().add_child(model.clone());

    while window.render_with_camera(camera) {
        model.append_rotation(&UnitQuaternion::from_euler_angles(0.0f32, 0.004, 0.0));
    }
}

trait MaxValue<V> {
    fn max_value() -> V;
}

macro_rules! impl_max_value {
    ($t:ident) => {
        impl MaxValue<Self> for $t {
            fn max_value() -> Self {
                ::std::$t::MAX
            }
        }
    };
}

impl_max_value!(i8);
impl_max_value!(i16);
impl_max_value!(i32);
impl_max_value!(i64);
impl_max_value!(isize);
impl_max_value!(u8);
impl_max_value!(u16);
impl_max_value!(u32);
impl_max_value!(u64);
impl_max_value!(usize);
impl_max_value!(f32);
impl_max_value!(f64);

trait Gene: Unsigned + NumCast + Copy + MaxValue<Self> {}

impl Gene for u8 {}
impl Gene for u16 {}
impl Gene for u32 {}
impl Gene for u64 {}
impl Gene for usize {}

struct Genotype<G> {
    genes: Vec<G>,
    index: usize,
}

impl<G: Gene> Genotype<G> {
    fn new(genes: Vec<G>) -> Genotype<G> {
        Genotype {
            genes: genes,
            index: 0,
        }
    }

    fn use_next_gene(&mut self) -> G {
        assert!(self.index < self.genes.len(), "Genotype index overflows gene list");

        let gene = self.genes[self.index];
        self.index = (self.index + 1) % self.genes.len();

        gene
    }

    fn max_selection_value<T: Gene>(num: T) -> G {
        let rep_max_value = num::cast::<_, u64>(G::max_value()).unwrap();
        let res_max_value = num::cast::<_, u64>(T::max_value()).unwrap();
        let max_value = num::cast::<_, G>(cmp::min(rep_max_value, res_max_value)).unwrap();

        num::cast::<_, G>(num).unwrap() % max_value
    }
}

impl<G: Gene> SelectionStrategy for Genotype<G> {
    fn select_alternative(&mut self, num: usize, _: &Vec<&str>, _: u32) -> usize {
        let limit = Self::max_selection_value(num);
        let gene = self.use_next_gene();

        num::cast::<_, usize>(gene % limit).unwrap()
    }

    fn select_repetition(&mut self, min: u32, max: u32, _: &Vec<&str>, _: u32) -> u32 {
        let limit = Self::max_selection_value(max - min + 1);
        let gene = self.use_next_gene();

        num::cast::<_, u32>(gene % limit).unwrap() + min
    }
}

fn weighted_selection(weights: &[f32], selector: f32) -> usize {
    let total_weight = weights.iter().fold(0.0, |acc, weight| acc + weight);
    let selector = selector * total_weight;

    let mut weight_acc = 0.0;
    let mut selected = weights.len() - 1;
    for (i, weight) in weights.iter().enumerate() {
        weight_acc += *weight;

        if selector < weight_acc {
            selected = i;
            break;
        }
    }

    selected
}

struct Distribution {
    depths: Vec<HashMap<String, Vec<Vec<f32>>>>,
    defaults: HashMap<String, Vec<Vec<f32>>>,
}

impl Distribution {
    fn new() -> Distribution {
        Distribution {
            depths: vec![],
            defaults: HashMap::new(),
        }
    }

    fn get_weights(&self, depth: usize, rule: &str, choice_num: u32) -> Option<&[f32]> {
        if depth < self.depths.len() {
            let rules = &self.depths[depth];
            if let Some(choices) = rules.get(rule) {
                if (choice_num as usize) < choices.len() {
                    let weights = &choices[choice_num as usize];
                    if !weights.is_empty() {
                        return Some(weights);
                    }
                }
            }
        }

        if let Some(choices) = self.defaults.get(rule) {
            if (choice_num as usize) < choices.len() {
                let weights = &choices[choice_num as usize];
                if !weights.is_empty() {
                    return Some(weights);
                }
            }
        }

        None
    }

    fn set_weights(&mut self, depth: usize, rule: &str, choice: u32, weights: &[f32]) {
        while self.depths.len() < depth + 1 {
            self.depths.push(HashMap::new());
        }

        let choices = self.depths[depth].entry(rule.to_string()).or_insert_with(Vec::new);
        let choice = choice as usize;
        while choices.len() < choice + 1 {
            choices.push(Vec::new());
        }

        choices[choice] = weights.to_vec();
    }

    fn set_default_weights(&mut self, rule: &str, choice: u32, weights: &[f32]) {
        let choices = self.defaults.entry(rule.to_string()).or_insert_with(Vec::new);
        let choice = choice as usize;
        while choices.len() < choice + 1 {
            choices.push(Vec::new());
        }

        choices[choice] = weights.to_vec();
    }
}

struct WeightedGenotype<G> {
    genotype: Genotype<G>,
    distribution: Distribution,
}

impl<G: Gene> WeightedGenotype<G> {
    fn new(genes: Vec<G>, distribution: Distribution) -> WeightedGenotype<G> {
        WeightedGenotype {
            genotype: Genotype::new(genes),
            distribution: distribution,
        }
    }

    fn find_depth(rulechain: &[&str]) -> usize {
        rulechain.iter().fold(0, |acc, r| if *r == "stack" { acc + 1 } else { acc })
    }
}

impl<G: Gene> SelectionStrategy for WeightedGenotype<G> {
    fn select_alternative(&mut self, num: usize, rulechain: &Vec<&str>, choice: u32) -> usize {
        let gene = self.genotype.use_next_gene();

        let depth = Self::find_depth(rulechain);
        let rule = rulechain.last().unwrap();
        let weights = self.distribution.get_weights(depth, rule, choice);

        if let Some(weights) = weights {
            assert_eq!(weights.len(), num, "Number of weights does not match number of alternatives");

            let gene_frac = num::cast::<_, f32>(gene).unwrap() / num::cast::<_, f32>(G::max_value()).unwrap();
            weighted_selection(weights, gene_frac)
        } else {
            self.genotype.select_alternative(num, rulechain, choice)
        }
    }

    fn select_repetition(&mut self, min: u32, max: u32, rulechain: &Vec<&str>, choice: u32) -> u32 {
        let gene = self.genotype.use_next_gene();

        let num = max - min + 1;

        let depth = Self::find_depth(rulechain);
        let rule = rulechain.last().unwrap();
        let weights = self.distribution.get_weights(depth, rule, choice);

        if let Some(weights) = weights {
            assert_eq!(weights.len(), num as usize, "Number of weights does not match number of repetition alternatives");

            let gene_frac = num::cast::<_, f32>(gene).unwrap() / num::cast::<_, f32>(G::max_value()).unwrap();
            min + weighted_selection(weights, gene_frac) as u32
        } else {
            self.genotype.select_repetition(min, max, rulechain, choice)
        }
    }
}

// Somehow make the below expansion functions a generic part of abnf::expand?

fn expand_productions<T>(grammar: &abnf::Ruleset, strategy: &mut T) -> ol::RuleMap
    where T: SelectionStrategy
{
    let mut rules = ol::RuleMap::new();
    let list = &grammar["productions"];

    if let abnf::List::Sequence(ref seq) = *list {
        assert_eq!(seq.len(), 1);
        let item = &seq[0];

        if let abnf::Content::Symbol(_) = item.content {
            let repeat = item.repeat.unwrap_or_default();
            let min = repeat.min.unwrap_or(0);
            let max = repeat.max.unwrap_or(u32::max_value());
            let num = strategy.select_repetition(min, max, &vec!["productions"], 0);

            for _ in 0..num {
                let (pred, succ) = expand_production(grammar, strategy);
                rules[pred] = succ;
            }
        }
    }

    rules
}

fn expand_production<T>(grammar: &abnf::Ruleset, strategy: &mut T) -> (char, String)
    where T: SelectionStrategy
{
    let list = &grammar["production"];

    if let abnf::List::Sequence(ref seq) = *list {
        assert_eq!(seq.len(), 2);
        let pred_item = &seq[0];
        let succ_item = &seq[1];

        let pred = if let abnf::Content::Symbol(_) = pred_item.content {
            expand_predecessor(grammar, strategy)
        } else {
            0 as char
        };

        let succ = if let abnf::Content::Symbol(_) = succ_item.content {
            expand_successor(grammar, strategy)
        } else {
            String::new()
        };

        (pred, succ)
    } else {
        (0 as char, String::new())
    }
}

fn expand_predecessor<T>(grammar: &abnf::Ruleset, strategy: &mut T) -> char
    where T: SelectionStrategy
{
    let value = expand_grammar(grammar, "predecessor", strategy);
    assert_eq!(value.len(), 1);
    value.as_bytes()[0] as char
}

fn expand_successor<T>(grammar: &abnf::Ruleset, strategy: &mut T) -> String
    where T: SelectionStrategy
{
    expand_grammar(grammar, "successor", strategy)
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

    #[test]
    fn test_lsystem_ge_expansion() {
        let grammar = abnf::parse_file("lsys.abnf").expect("Could not parse ABNF file");
        let genes = vec![
           2u8, // repeat 3 - "F[FX]X"
           0, // symbol - "F"
           0, // variable - "F"
           0, // "F"
           1, // stack - "[FX]"
           1, // repeat - "FX"
           0, // symbol - "F"
           0, // variable - "F"
           0, // "F"
           0, // symbol - "X"
           0, // variable - "X"
           1, // "X"
           0, // symbol - "X"
           0, // variable - "X"
           1, // "X"
        ];
        let mut genotype = Genotype::new(genes);

        assert_eq!(
            expand_grammar(&grammar, "axiom", &mut genotype),
            "F[FX]X"
        );
    }

    #[test]
    fn test_lsystem_ge_inference() {
        let grammar = abnf::parse_file("lsys.abnf").expect("Could not parse ABNF file");
        let genes = vec![
           2, // repeat - "F[FX]X"
           0, // symbol - "F"
           0, // variable - "F"
           0, // "F"
           1, // stack - "[FX]"
           1, // repeat - "FX"
           0, // symbol - "F"
           0, // variable - "F"
           0, // "F"
           0, // symbol - "X"
           0, // variable - "X"
           1, // "X"
           0, // symbol - "X"
           0, // variable - "X"
           1, // "X"
        ];

        assert_eq!(
            infer_selections("F[FX]X", &grammar, "axiom"),
            Ok(genes)
        );
    }
}
