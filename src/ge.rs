use std::f32::consts::{PI, FRAC_PI_2};
use std::f32;
use std::{cmp, fs, fmt, io};
use std::collections::HashMap;
use std::io::{BufWriter, BufReader, Write};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::ops::Add;

use rand::{self, Rng, XorShiftRng, SeedableRng};
use rand::distributions::{IndependentSample, Range};
use na::{self, Unit, UnitQuaternion, Point2, Point3, Vector2, Vector3, Translation3, Rotation3};
use kiss3d::camera::ArcBall;
use kiss3d::scene::SceneNode;
use num::{self, Unsigned, NumCast};
use glfw::{Key, WindowEvent, Action};
use serde_yaml;
use time;
use ncu;
use num_cpus;
use futures::{Future, future};
use futures_cpupool::CpuPool;
use bincode;
use crossbeam;
use clap::{App, SubCommand, Arg, ArgMatches};

use abnf;
use abnf::expand::{SelectionStrategy, expand_grammar};
use lsys::{self, ol, Command};
use lsys3d;
use lsystems;
use yobun::*;
use super::setup_window;

pub fn get_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("ge")
        .about("Run random plant generation using GE")
        .subcommand(SubCommand::with_name("abnf")
            .about("Print the parsed ABNF structure")
        )
        .subcommand(SubCommand::with_name("random")
            .about("Randomly generate plant based on random genes and ABNF")
        )
        .subcommand(SubCommand::with_name("inferred")
            .about("Run program that infers the genes of an L-system")
        )
        .subcommand(SubCommand::with_name("distribution")
            .about("Generate plants based on a predefined distribution")
            .arg(Arg::with_name("distribution")
                .short("d")
                .long("distribution")
                .takes_value(true)
                .help("Distribution file to use. Otherwise default distribution is used.")
            )
            .arg(Arg::with_name("num-samples")
                .short("n")
                .long("num-samples")
                .takes_value(true)
                .default_value("64")
                .help("Number of samples to generate before visualizing the best")
            )
        )
        .subcommand(SubCommand::with_name("sampling")
            .about("Run random sampling program until you type 'quit'")
            .arg(Arg::with_name("distribution")
                .short("d")
                .long("distribution")
                .takes_value(true)
                .help("Distribution file to use. Otherwise default distribution is used.")
            )
        )
        .subcommand(SubCommand::with_name("sampling-dist")
            .about("Take samples from directory and output a distribution CSV file")
            .arg(Arg::with_name("samples")
                .short("s")
                .long("samples")
                .takes_value(true)
                .default_value("sample")
                .help("Directory of samples to use")
            )
            .arg(Arg::with_name("csv")
                .long("csv")
                .takes_value(true)
                .default_value("distribution.csv")
                .help("Name of the output CSV file")
            )
            .arg(Arg::with_name("bin")
                .long("bin")
                .takes_value(true)
                .default_value("distribution.bin")
                .help("Name of the output bincode file")
            )
        )
}

pub fn run_ge(matches: &ArgMatches) {
    if matches.subcommand_matches("abnf").is_some() {
        run_print_abnf();
    } else if matches.subcommand_matches("random").is_some() {
        run_random_genes();
    } else if matches.subcommand_matches("inferred").is_some() {
        run_bush_inferred();
    } else if let Some(matches) = matches.subcommand_matches("distribution") {
        run_with_distribution(matches);
    } else if let Some(matches) = matches.subcommand_matches("sampling") {
        run_random_sampling(matches);
    } else if let Some(matches) = matches.subcommand_matches("sampling-dist") {
        run_sampling_distribution(matches);
    } else {
        println!("A subcommand must be specified. See help by passing -h.");
    }
}

type GenePrimitive = u32;

fn generate_genome<R: Rng>(rng: &mut R, len: usize) -> Vec<GenePrimitive> {
    let gene_range = Range::new(GenePrimitive::min_value(), GenePrimitive::max_value());

    let mut genes = Vec::with_capacity(len);
    for _ in 0..len {
        genes.push(gene_range.ind_sample(rng));
    }

    genes
}

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

fn is_nothing(lsystem: &ol::LSystem) -> bool {
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

pub struct Skeleton {
    pub points: Vec<Point3<f32>>,
    pub edges: Vec<Vec<usize>>,
}

impl Skeleton {
    pub fn new() -> Skeleton {
        Skeleton {
            points: Vec::new(),
            edges: Vec::new(),
        }
    }
}

pub fn build_skeleton(instructions: ol::InstructionsIter,
                      settings: &lsys::Settings,
                      size_limit: usize,
                      instruction_limit: usize)
                      -> Option<Skeleton> {
    let segment_length = settings.step;

    let mut skeleton = Skeleton::new();
    skeleton.points.push(Point3::new(0.0, 0.0, 0.0));

    let mut position = Point3::new(0.0, 0.0, 0.0);
    let mut rotation = UnitQuaternion::from_euler_angles(FRAC_PI_2, 0.0, 0.0);
    let mut parent = 0usize;
    let mut filling = false;

    let mut states = Vec::<(Point3<f32>, UnitQuaternion<f32>, usize)>::new();

    for (iteration, instruction) in instructions.enumerate() {
        if skeleton.points.len() > size_limit || iteration >= instruction_limit {
            return None;
        }

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

                if !filling {
                    let direction = rotation * Vector3::new(0.0, 0.0, -1.0);
                    position = position + (direction * segment_length);

                    let index = skeleton.points.len();
                    skeleton.points.push(position);
                    skeleton.edges.push(Vec::new());

                    skeleton.edges[parent].push(index);
                    parent = index;
                }
            }
            Command::YawRight => {
                let angle = {
                    if !instruction.args.is_empty() {
                        instruction.args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * -angle);
            }
            Command::YawLeft => {
                let angle = {
                    if !instruction.args.is_empty() {
                        instruction.args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * angle);
            }
            Command::UTurn => {
                let angle = PI;
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * -angle);
            }
            Command::PitchUp => {
                let angle = {
                    if !instruction.args.is_empty() {
                        instruction.args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * angle);
            }
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
            }
            Command::RollLeft => {
                let angle = {
                    if !instruction.args.is_empty() {
                        instruction.args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * angle);
            }
            Command::Shrink => {}
            Command::Grow => {}
            Command::Width => {}
            Command::Push => {
                states.push((position, rotation, parent));
            }
            Command::Pop => {
                if let Some((stored_position, stored_rotation, stored_parent)) = states.pop() {
                    position = stored_position;
                    rotation = stored_rotation;
                    parent = stored_parent;
                } else {
                    panic!("Tried to pop empty state stack");
                }
            }
            Command::BeginSurface => {
                filling = true;
                states.push((position, rotation, parent));
            }
            Command::EndSurface => {
                if let Some((stored_position, stored_rotation, stored_parent)) = states.pop() {
                    position = stored_position;
                    rotation = stored_rotation;
                    parent = stored_parent;
                } else {
                    panic!("Tried to pop empty state stack");
                }

                filling = false;
            }
            Command::NextColor => {}
            Command::Noop => {}
        };
    }

    Some(skeleton)
}

struct Properties {
    reach: f32,
    drop: f32,
    spread: f32,
    center: Point3<f32>,
    center_spread: f32,
    num_points: usize,
}

const SKELETON_LIMIT: usize = 20000;
const INSTRUCTION_LIMIT: usize = 10000000;

fn is_crap(lsystem: &ol::LSystem, settings: &lsys::Settings) -> bool {
    if is_nothing(lsystem) {
        return true;
    }

    let instruction_iter = lsystem.instructions_iter(settings.iterations, &settings.command_map);
    if let Some(skeleton) = build_skeleton(instruction_iter,
                                           settings,
                                           SKELETON_LIMIT,
                                           INSTRUCTION_LIMIT) {
        if skeleton.points.len() <= 1 {
            return true;
        }

        return false;
    } else {
        return true;
    }
}

fn fitness(lsystem: &ol::LSystem, settings: &lsys::Settings) -> (f32, Option<Properties>) {
    if is_nothing(lsystem) {
        return (f32::MIN, None);
    }

    let instruction_iter = lsystem.instructions_iter(settings.iterations, &settings.command_map);
    if let Some(skeleton) = build_skeleton(instruction_iter,
                                           settings,
                                           SKELETON_LIMIT,
                                           INSTRUCTION_LIMIT) {
        if skeleton.points.len() <= 1 {
            return (f32::MIN, None);
        }

        let reach = skeleton
            .points
            .iter()
            .max_by(|a, b| a.y.partial_cmp(&b.y).unwrap())
            .unwrap()
            .y;
        let drop = skeleton
            .points
            .iter()
            .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap())
            .unwrap()
            .y;

        let floor_points: Vec<_> = skeleton
            .points
            .iter()
            .map(|p| Point2::new(p.x, p.z))
            .collect();
        let floor_distances_iter = floor_points
            .iter()
            .map(|p| na::norm(&Vector2::new(p.x, p.y)));
        let spread = floor_distances_iter
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let center = ncu::center(&skeleton.points);
        let floor_center = Point2::new(center.x, center.z);
        let center_distance = na::norm(&Vector2::new(floor_center.x, floor_center.y));

        let closeness = skeleton
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i >= skeleton.edges.len() {
                    return 0.0;
                }

                let edges = skeleton.edges[i].iter().map(|e| skeleton.points[*e]);

                let segments: Vec<_> = edges.map(|e| (e - p).normalize()).collect();
                let closeness = segments
                    .iter()
                    .enumerate()
                    .map(|(a_i, a_s)| {
                        let mut closest = -1.0;
                        for (b_i, b_s) in segments.iter().enumerate() {
                            if b_i != a_i {
                                let dot = na::dot(a_s, b_s);
                                closest = *na::partial_max(&dot, &closest).unwrap();
                            }
                        }

                        const TRESHOLD: f32 = 0.9;
                        if closest < TRESHOLD {
                            0.0
                        } else {
                            (closest - TRESHOLD) * (1.0 / (1.0 - TRESHOLD))
                        }
                    })
                    .max_by(|a, b| a.partial_cmp(b).unwrap());

                if let Some(closeness) = closeness {
                    closeness
                } else {
                    0.0
                }
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let center_direction = Unit::new_normalize(Vector2::new(center.x, center.z));
        let center_spread = floor_points
            .iter()
            .map(|p| Vector2::new(p.x, p.y))
            .map(|p| project_onto(&p, &center_direction))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();


        // First point is root, which we don't want to measure, so skip 1.
        let branching_counts = skeleton
            .points
            .iter()
            .skip(1)
            .enumerate()
            .filter_map(|(i, _)| {
                if i >= skeleton.edges.len() {
                    return None;
                }

                if skeleton.edges[i].is_empty() {
                    return None;
                }

                Some(skeleton.edges[i].len())
            })
            .collect::<Vec<_>>();

        let total_branching_count = branching_counts.iter().fold(0, |total, b| total + b);
        let branching_complexity = {
            if !branching_counts.is_empty() {
                total_branching_count as f32 / branching_counts.len() as f32
            } else {
                0.0
            }
        };
        println!("Complexity: {}", branching_complexity);

        let branching_fitness = {
            if branching_complexity >= 1.0 && branching_complexity < 2.0 {
                interpolate_cos(0.0, 1.0, branching_complexity - 1.0)
            } else if branching_complexity < 3.0 {
                1.0
            } else if branching_complexity < 7.0 {
                let t = (branching_complexity - 3.0) / (7.0 - 3.0);
                interpolate_cos(1.0, -1.0, t)
            } else {
                // No branches, or 7 or more branches.
                -1.0
            }
        };

        let branching_reward = *na::partial_max(&branching_fitness, &0.0).unwrap();
        let branching_punishment = *na::partial_max(&-branching_fitness, &0.0).unwrap();

        let balance_fitness = (0.5 - (center_distance / center_spread)) * 2.0;

        let drop_fitness = -drop;

        println!("Branching fitness: {}", branching_fitness);
        println!("Balance: {}", balance_fitness);
        println!("Closeness: {}", closeness);
        println!("Drop: {}", drop_fitness);

        // let reward = (proportion_fitness + balance_fitness) / 2.0;
        let reward = (balance_fitness + branching_reward) / 2.0;
        let punishment = closeness + drop_fitness + branching_punishment;
        let fit = reward - punishment;

        let prop = Properties {
            reach: reach,
            drop: drop,
            spread: spread,
            center: center,
            center_spread: center_spread,
            num_points: skeleton.points.len(),
        };

        (fit, Some(prop))
    } else {
        (f32::MIN, None)
    }

    // TODO: balanced number of branches.
}

fn add_properties_rendering(node: &mut SceneNode, properties: &Properties) {
    const LINE_LEN: f32 = 1.0;
    const LINE_WIDTH: f32 = 0.02;

    let mut center = SceneNode::new_empty();
    center.add_cube(LINE_WIDTH, LINE_LEN, LINE_WIDTH);
    center.add_cube(LINE_LEN, LINE_WIDTH, LINE_WIDTH);
    center.add_cube(LINE_WIDTH, LINE_WIDTH, LINE_LEN);
    center.set_local_translation(Translation3::new(properties.center.x,
                                                   properties.center.y,
                                                   properties.center.z));
    node.add_child(center);

    let mut reach = SceneNode::new_empty();
    reach.add_cube(LINE_WIDTH, properties.reach, LINE_WIDTH);
    reach.set_local_translation(Translation3::new(0.0, properties.reach / 2.0, 0.0));
    node.add_child(reach);

    let mut drop = SceneNode::new_empty();
    drop.add_cube(LINE_WIDTH, properties.drop.abs(), LINE_WIDTH);
    drop.set_local_translation(Translation3::new(0.0, properties.drop / 2.0, 0.0));
    node.add_child(drop);

    let mut spread = SceneNode::new_empty();
    spread
        .add_cube(properties.spread * 2.0, LINE_WIDTH, LINE_WIDTH)
        .set_color(0.8, 0.1, 0.1);
    spread.add_cube(LINE_WIDTH, LINE_WIDTH, properties.spread * 2.0);
    node.add_child(spread);

    let mut balance = SceneNode::new_empty();
    let center_vector = Vector2::new(properties.center.x, -properties.center.z);
    let center_distance = na::norm(&Vector2::new(center_vector.x, center_vector.y));
    let center_direction = na::normalize(&center_vector);
    let center_angle = center_direction.y.atan2(center_direction.x);
    balance.append_rotation(&UnitQuaternion::from_euler_angles(0.0, center_angle, 0.0));

    let mut center_dist = balance.add_cube(center_distance, LINE_WIDTH * 1.2, LINE_WIDTH * 1.2);
    center_dist.set_color(0.1, 0.1, 0.8);
    center_dist.set_local_translation(Translation3::new(center_distance / 2.0, 0.0, 0.0));

    let mut center_imbalance = balance.add_cube(properties.center_spread / 2.0,
                                                LINE_WIDTH * 1.1,
                                                LINE_WIDTH * 1.1);
    center_imbalance.set_color(0.1, 0.8, 0.1);
    center_imbalance.set_local_translation(Translation3::new(properties.center_spread / 4.0,
                                                             0.0,
                                                             0.0));

    let mut center_spread = balance.add_cube(properties.center_spread, LINE_WIDTH, LINE_WIDTH);
    center_spread.set_local_translation(Translation3::new(properties.center_spread / 2.0,
                                                          0.0,
                                                          0.0));

    node.add_child(balance);
}

fn random_seed() -> [u32; 4] {
    [rand::thread_rng().gen::<u32>(),
     rand::thread_rng().gen::<u32>(),
     rand::thread_rng().gen::<u32>(),
     rand::thread_rng().gen::<u32>()]
}

const GENOME_LENGTH: usize = 100;

fn run_with_distribution(matches: &ArgMatches) {
    let (mut window, _) = setup_window();

    let grammar =
        Arc::new(abnf::parse_file("grammar/lsys2.abnf").expect("Could not parse ABNF file"));

    let distribution = match matches.value_of("distribution") {
        Some(filename) => {
            println!("Using distribution from {}", filename);
            let file = File::open(filename).unwrap();
            let d: Distribution =
                bincode::deserialize_from(&mut BufReader::new(file), bincode::Infinite).unwrap();
            Arc::new(d)
        }
        None => {
            let distribution = {
                let mut distribution = Distribution::new();

                // lsys2.abnf distribution
                distribution.set_weights(0, "string", 1, &[1.0, 1.0]);
                distribution.set_weights(1, "string", 1, &[1.0, 1.0]);
                distribution.set_weights(2, "string", 1, &[1.0, 1.0]);
                distribution.set_default_weights("string", 1, &[1.0, 0.0]);

                // lsys.abnf distribution
                //  distribution.set_default_weights("productions", 0, &[1.0, 1.0]);
                //  distribution.set_default_weights("string",
                //                                   0,
                //                                   &[1.0, 2.0, 2.0, 2.0, 1.0, 1.0]);
                //  distribution.set_default_weights("string", 1, &[1.0, 0.0]);
                //
                //  distribution.set_weights(0, "string", 0, &[1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
                //  distribution.set_weights(0, "string", 1, &[1.0, 1.0]);
                //
                //  distribution.set_weights(1, "string", 1, &[10.0, 1.0]);

                distribution
            };
            Arc::new(distribution)
        }
    };

    println!("Distribution:");
    println!("{}", distribution);

    let num_samples = usize::from_str_radix(matches.value_of("num-samples").unwrap(), 10).unwrap();

    let settings = Arc::new(lsys::Settings {
                                width: 0.05,
                                angle: PI / 8.0,
                                iterations: 5,
                                ..lsys::Settings::new()
                            });

    let mut system = ol::LSystem::new();
    let mut model = SceneNode::new_empty();

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
        //model.append_rotation(&UnitQuaternion::from_euler_angles(0.0f32, 0.004, 0.0));

        for event in window.events().iter() {
            match event.value {
                WindowEvent::Key(Key::S, _, Action::Release, _) => {
                    let filename = format!("{}.yaml", time::now().rfc3339());
                    let path = model_dir.join(filename);

                    let file = File::create(&path).unwrap();
                    serde_yaml::to_writer(&mut BufWriter::new(file), &system).unwrap();

                    println!("Saved to {}", path.to_str().unwrap());
                }
                WindowEvent::Key(Key::Space, _, Action::Release, _) => {
                    struct Sample {
                        seed: [u32; 4],
                        score: f32,
                    }

                    fn generate_sample(grammar: &abnf::Ruleset,
                                       distribution: &Distribution,
                                       settings: &lsys::Settings)
                                       -> Sample {
                        let seed = random_seed();
                        let genes = generate_genome(&mut XorShiftRng::from_seed(seed),
                                                    GENOME_LENGTH);
                        let system = generate_system(grammar,
                                                     &mut WeightedGenotype::new(genes,
                                                                                distribution));
                        let (score, _) = fitness(&system, settings);
                        Sample {
                            seed: seed,
                            score: score,
                        }
                    }

                    println!("Generating {} samples...", num_samples);
                    let start_time = time::now();

                    let mut samples = {
                        let workers = num_cpus::get() + 1;

                        if workers == 1 || num_samples <= 1 {
                            (0..num_samples)
                                .map(|_| generate_sample(&grammar, &distribution, &settings))
                                .collect::<Vec<_>>()
                        } else {
                            let pool = CpuPool::new(workers);
                            let mut tasks = Vec::with_capacity(num_samples);

                            for _ in 0..num_samples {
                                let distribution = distribution.clone();
                                let grammar = grammar.clone();
                                let settings = settings.clone();

                                tasks.push(pool.spawn_fn(move || {
                                                             let sample =
                                                                 generate_sample(&grammar,
                                                                                 &distribution,
                                                                                 &settings);
                                                             future::ok::<Sample, ()>(sample)
                                                         }));
                            }

                            future::join_all(tasks).wait().unwrap()
                        }
                    };

                    let end_time = time::now();
                    let duration = end_time - start_time;
                    println!("Duration: {}.{}",
                             duration.num_seconds(),
                             duration.num_milliseconds());

                    samples.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

                    let sample = samples.pop().unwrap();

                    println!("Found sample with score {}.", sample.score);
                    println!("Building model...");

                    window.remove(&mut model);

                    let genes = generate_genome(&mut XorShiftRng::from_seed(sample.seed),
                                                GENOME_LENGTH);
                    system = generate_system(&grammar,
                                             &mut WeightedGenotype::new(genes, &distribution));
                    let (_, properties) = fitness(&system, &settings);

                    if let Some(properties) = properties {
                        println!("{} points.", properties.num_points);
                        let instructions =
                            system.instructions_iter(settings.iterations, &settings.command_map);
                        model = lsys3d::build_model(instructions, &settings);
                        add_properties_rendering(&mut model, &properties);
                        window.scene_mut().add_child(model.clone());

                        println!("");
                        println!("LSystem:");
                        println!("{}", system);

                        model_index = 0;
                    } else {
                        println!("Plant was nothing or reached the limits.");
                    }
                }
                WindowEvent::Key(Key::L, _, Action::Release, _) => {
                    let mut models = fs::read_dir(model_dir)
                        .unwrap()
                        .map(|e| e.unwrap().path())
                        .collect::<Vec<_>>();
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

                        let instructions =
                            system.instructions_iter(settings.iterations, &settings.command_map);
                        let (score, properties) = fitness(&system, &settings);
                        println!("Score: {}", score);

                        if let Some(properties) = properties {
                            window.remove(&mut model);
                            model = lsys3d::build_model(instructions, &settings);
                            add_properties_rendering(&mut model, &properties);
                            window.scene_mut().add_child(model.clone());
                        }
                    }

                    model_index += 1;
                }
                WindowEvent::Key(Key::X, _, Action::Release, _) => {
                    let (system, settings) = lsystems::make_bush();
                    println!("LSystem:");
                    println!("{}", system);

                    let instructions =
                        system.instructions_iter(settings.iterations, &settings.command_map);
                    let (score, properties) = fitness(&system, &settings);
                    println!("Score: {}", score);

                    window.remove(&mut model);
                    model = lsys3d::build_model(instructions, &settings);
                    add_properties_rendering(&mut model, &properties.unwrap());
                    window.scene_mut().add_child(model.clone());

                    model_index = 0;
                }
                _ => {}
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SampleBatch {
    sample_count: usize,
    accepted: Vec<[u32; 4]>,
}

fn get_sample_setup() -> (abnf::Ruleset, Distribution) {
    const DEPTHS: usize = 4;

    let grammar = abnf::parse_file("grammar/lsys2.abnf").expect("Could not parse ABNF file");
    let distribution = {
        let mut distribution = Distribution::new();
        for d in 0..DEPTHS - 1 {
            distribution.set_weights(d, "string", 1, &[1.0, 1.0]);
        }
        distribution.set_default_weights("string", 1, &[1.0, 0.0]);

        distribution
    };

    (grammar, distribution)
}

fn run_random_sampling(matches: &ArgMatches) {
    struct Sample {
        seed: [u32; 4],
        crap: bool,
    }

    fn generate_sample(grammar: &abnf::Ruleset,
                       distribution: &Distribution,
                       settings: &lsys::Settings)
                       -> Sample {
        let seed = random_seed();
        let genes = generate_genome(&mut XorShiftRng::from_seed(seed), GENOME_LENGTH);
        let system = generate_system(grammar, &mut WeightedGenotype::new(genes, distribution));
        Sample {
            seed: seed,
            crap: is_crap(&system, settings),
        }
    }

    let (grammar, distribution) = get_sample_setup();

    let grammar = Arc::new(grammar);
    let distribution = match matches.value_of("distribution") {
        Some(filename) => {
            println!("Using distribution from {}", filename);
            let file = File::open(filename).unwrap();
            let d: Distribution =
                bincode::deserialize_from(&mut BufReader::new(file), bincode::Infinite).unwrap();
            Arc::new(d)
        }
        None => Arc::new(distribution),
    };

    println!("Distribution:");
    println!("{}", distribution);

    let settings = Arc::new(lsys::Settings {
                                width: 0.05,
                                angle: PI / 8.0,
                                iterations: 5,
                                ..lsys::Settings::new()
                            });

    // const BATCH_SIZE: usize = 1024 / 8;
    const BATCH_SIZE: usize = 16 / 8;
    const SEQUENCE_SIZE: usize = 16;

    let sample_dir = Path::new("sample").join(format!("{}", time::now().rfc3339()));
    fs::create_dir_all(&sample_dir).unwrap();

    println!("Starting random sampling...");

    let num_workers = num_cpus::get() + 1;
    let work = Arc::new(AtomicBool::new(true));
    let num_samples = Arc::new(AtomicUsize::new(0));
    let num_good_samples = Arc::new(AtomicUsize::new(0));

    let start_time = time::now();

    crossbeam::scope(|scope| {
        let sample_dir = &sample_dir;

        for worker_id in 0..num_workers {
            let distribution = distribution.clone();
            let grammar = grammar.clone();
            let settings = settings.clone();
            let work = work.clone();
            let num_samples = num_samples.clone();
            let num_good_samples = num_good_samples.clone();

            scope.spawn(move || {
                let dump_samples = |accepted_samples: &[Sample],
                                    sample_count: usize,
                                    batch: usize| {
                    let filename = format!("{}.{}.sample", worker_id, batch);
                    let path = sample_dir.join(filename);
                    let file = File::create(&path).unwrap();

                    let samples = SampleBatch {
                        sample_count: sample_count,
                        accepted: accepted_samples
                            .iter()
                            .map(|s| s.seed)
                            .collect::<Vec<_>>(),
                    };
                    bincode::serialize_into(&mut BufWriter::new(file), &samples, bincode::Infinite)
                        .unwrap();
                };

                // Room for 0.5% of samples.
                let mut accepted_samples = Vec::with_capacity(BATCH_SIZE / 200);
                let mut batch = 0;
                let mut batch_num_samples = 0;

                while work.load(Ordering::Relaxed) {
                    for _ in 0..SEQUENCE_SIZE {
                        let sample = generate_sample(&grammar, &distribution, &settings);
                        if !sample.crap {
                            accepted_samples.push(sample);
                        }
                    }

                    batch_num_samples += SEQUENCE_SIZE;

                    if accepted_samples.len() >= BATCH_SIZE {
                        dump_samples(&accepted_samples, batch_num_samples, batch);

                        num_samples.fetch_add(batch_num_samples, Ordering::Relaxed);

                        accepted_samples.clear();
                        batch += 1;
                        batch_num_samples = 0;
                    }
                }

                dump_samples(&accepted_samples, batch_num_samples, batch);
                num_samples.fetch_add(batch_num_samples, Ordering::Relaxed);
                num_good_samples.fetch_add(batch * BATCH_SIZE + accepted_samples.len(),
                                           Ordering::Relaxed);
            });
        }

        println!("Spawned {} threads.", num_workers);

        let mut input = String::new();
        while input != "quit" {
            input = String::new();
            print!("> ");
            io::stdout().flush().unwrap();
            io::stdin().read_line(&mut input).unwrap();
            input.pop().unwrap(); // remove '\n'.
        }

        println!("Quitting...");
        work.store(false, Ordering::Relaxed);
    });

    let end_time = time::now();
    let duration = end_time - start_time;
    println!("Duration: {}:{}:{}.{}",
             duration.num_hours(),
             duration.num_minutes() % 60,
             duration.num_seconds() % 60,
             duration.num_milliseconds() % 1000);

    let num_samples = Arc::try_unwrap(num_samples).unwrap().into_inner();
    let num_good_samples = Arc::try_unwrap(num_good_samples).unwrap().into_inner();
    println!("Good samples: {}/{} ({:.*}%)",
             num_good_samples,
             num_samples,
             1,
             num_good_samples as f32 / num_samples as f32 * 100.0);
}

fn run_sampling_distribution(matches: &ArgMatches) {
    let samples_path = Path::new(matches.value_of("samples").unwrap());
    let csv_path = Path::new(matches.value_of("csv").unwrap());
    let bin_path = Path::new(matches.value_of("bin").unwrap());

    println!("Reading samples from {}.", samples_path.to_str().unwrap());

    let sample_paths = read_dir_all(samples_path)
        .unwrap()
        .filter_map(|e| {
            let path = e.unwrap().path();
            if path.is_dir() {
                return None;
            }

            if let Some(extension) = path.extension() {
                if extension != "sample" {
                    return None;
                }
            } else {
                return None;
            }

            Some(path)
        });

    let mut sample_count = 0;
    let mut accepted_samples = Vec::new();

    for batch_path in sample_paths {
        let file = File::open(&batch_path).unwrap();
        let batch: SampleBatch =
            bincode::deserialize_from(&mut BufReader::new(file), bincode::Infinite).unwrap();

        accepted_samples.reserve(batch.accepted.len());
        for sample in batch.accepted {
            accepted_samples.push(sample);
        }
        sample_count += batch.sample_count;
    }

    println!("Read {} accepted samples from a total of {} samples.",
             accepted_samples.len(),
             sample_count);

    let (grammar, distribution) = get_sample_setup();
    let grammar = Arc::new(grammar);
    let distribution = Arc::new(distribution);

    let workers = num_cpus::get() + 1;
    let pool = CpuPool::new(workers);
    let mut tasks = Vec::with_capacity(accepted_samples.len());

    for seed in accepted_samples {
        let distribution = distribution.clone();
        let grammar = grammar.clone();

        tasks.push(pool.spawn_fn(move || {
            let genes = generate_genome(&mut XorShiftRng::from_seed(seed), GENOME_LENGTH);
            let mut stats_genotype = WeightedGenotypeStats::with_genes(genes, &distribution);
            expand_grammar(&grammar, "axiom", &mut stats_genotype);
            expand_productions(&grammar, &mut stats_genotype);

            future::ok::<SelectionStats, ()>(stats_genotype.take_stats())
        }));
    }

    let stats_collection = future::join_all(tasks).wait().unwrap();
    let stats = stats_collection
        .iter()
        .fold(SelectionStats::new(), |sum, stats| sum + stats);

    let mut csv_file = File::create(csv_path).unwrap();
    csv_file
        .write_all(stats.to_csv_normalized().as_bytes())
        .unwrap();

    let mut distribution = stats.to_distribution();
    distribution.set_default_weights("string", 1, &[1.0, 0.0]);
    let dist_file = File::create(bin_path).unwrap();
    bincode::serialize_into(&mut BufWriter::new(dist_file),
                            &distribution,
                            bincode::Infinite)
            .unwrap();
}

struct SelectionStats {
    data: Vec<HashMap<String, Vec<Vec<usize>>>>,
}

impl SelectionStats {
    fn new() -> SelectionStats {
        SelectionStats { data: vec![] }
    }

    fn make_room(&mut self, depth: usize, rule: &str, choice: u32, num: usize) {
        while self.data.len() <= depth {
            self.data.push(HashMap::new());
        }

        let choices = self.data[depth]
            .entry(rule.to_string())
            .or_insert_with(Vec::new);
        let choice = choice as usize;
        while choices.len() <= choice {
            choices.push(Vec::new());
        }

        let alternatives = &mut choices[choice];
        while alternatives.len() < num {
            alternatives.push(0);
        }
    }

    fn add_selection(&mut self, selection: usize, depth: usize, rule: &str, choice: u32) {
        assert!(depth < self.data.len(),
                format!("Depth {} is not available. Use make_room to make it.",
                        depth));
        let rules = &mut self.data[depth];

        let choices = &mut rules.entry(rule.to_string()).or_insert_with(Vec::new);

        let choice = choice as usize;
        assert!(choice < choices.len(),
                format!("Choice {} is not available. Use make_room to make it.",
                        choice));
        let alternatives = &mut choices[choice];

        assert!(selection < alternatives.len(),
                format!("Alternative {} is not available. Use make_room to make it.",
                        selection));
        alternatives[selection] += 1;
    }

    fn to_csv_normalized(&self) -> String {
        let mut csv = "depth,rule,choice,alternative,weight\n".to_string();
        for (depth, rules) in self.data.iter().enumerate() {
            for (rule, choices) in rules {
                for (choice, alternatives) in choices.iter().enumerate() {
                    let mut total = 0;
                    for count in alternatives {
                        total += *count;
                    }

                    for (alternative, count) in alternatives.iter().enumerate() {
                        let weight = *count as f32 / total as f32;
                        csv +=
                            &format!("{},{},{},{},{}\n", depth, rule, choice, alternative, weight);
                    }
                }
            }
        }

        csv
    }

    fn to_distribution(&self) -> Distribution {
        let mut distribution = Distribution::new();

        for (depth, rules) in self.data.iter().enumerate() {
            for (rule, choices) in rules {
                for (choice, alternatives) in choices.iter().enumerate() {
                    let mut total = 0;
                    for count in alternatives {
                        total += *count;
                    }

                    let mut weights = Vec::new();

                    for count in alternatives {
                        let weight = *count as f32 / total as f32;
                        weights.push(weight);
                    }

                    distribution.set_weights(depth, rule, choice as u32, &weights);
                }
            }
        }

        distribution
    }
}

impl Add for SelectionStats {
    type Output = SelectionStats;

    fn add(self, other: SelectionStats) -> SelectionStats {
        self.add(&other)
    }
}

impl<'a> Add<&'a SelectionStats> for SelectionStats {
    type Output = SelectionStats;

    fn add(mut self, other: &SelectionStats) -> SelectionStats {
        while self.data.len() < other.data.len() {
            self.data.push(HashMap::new());
        }

        for (depth, other_rules) in other.data.iter().enumerate() {
            for (rule, other_choices) in other_rules {
                let choices = self.data[depth]
                    .entry(rule.to_string())
                    .or_insert_with(Vec::new);
                while choices.len() < other_choices.len() {
                    choices.push(Vec::new());
                }

                for (choice, other_alternatives) in other_choices.iter().enumerate() {
                    let alternatives = &mut choices[choice];
                    while alternatives.len() < other_alternatives.len() {
                        alternatives.push(0);
                    }

                    for (alternative, count) in other_alternatives.iter().enumerate() {
                        alternatives[alternative] += *count;
                    }
                }
            }
        }

        self
    }
}

struct WeightedGenotypeStats<'a, G> {
    weighted_genotype: WeightedGenotype<'a, G>,
    stats: SelectionStats,
}

impl<'a, G: Gene> WeightedGenotypeStats<'a, G> {
    fn with_genes(genes: Vec<G>, distribution: &'a Distribution) -> WeightedGenotypeStats<G> {
        WeightedGenotypeStats {
            weighted_genotype: WeightedGenotype::new(genes, distribution),
            stats: SelectionStats::new(),
        }
    }

    fn new(distribution: &'a Distribution) -> WeightedGenotypeStats<G> {
        Self::with_genes(vec![], distribution)
    }

    fn take_stats(self) -> SelectionStats {
        self.stats
    }
}

impl<'a, G: Gene> SelectionStrategy for WeightedGenotypeStats<'a, G> {
    fn select_alternative(&mut self, num: usize, rulechain: &[&str], choice: u32) -> usize {
        let selection = self.weighted_genotype
            .select_alternative(num, rulechain, choice);

        let depth = WeightedGenotype::<'a, G>::find_depth(rulechain);
        let rule = rulechain.last().unwrap();

        self.stats.make_room(depth, rule, choice, num);
        self.stats.add_selection(selection, depth, rule, choice);

        selection
    }

    fn select_repetition(&mut self, min: u32, max: u32, rulechain: &[&str], choice: u32) -> u32 {
        let selection = self.weighted_genotype
            .select_repetition(min, max, rulechain, choice);

        let depth = WeightedGenotype::<'a, G>::find_depth(rulechain);
        let rule = rulechain.last().unwrap();
        let num = (max - min + 1) as usize;

        self.stats.make_room(depth, rule, choice, num);
        self.stats
            .add_selection((selection - min) as usize, depth, rule, choice);

        selection
    }
}

struct ReadDirAll {
    visit_stack: Vec<fs::ReadDir>,
}

impl Iterator for ReadDirAll {
    type Item = io::Result<fs::DirEntry>;

    fn next(&mut self) -> Option<io::Result<fs::DirEntry>> {
        let mut iter = match self.visit_stack.pop() {
            Some(iter) => iter,
            None => return None,
        };

        let mut entry = iter.next();
        while entry.is_none() {
            iter = match self.visit_stack.pop() {
                Some(iter) => iter,
                None => return None,
            };

            entry = iter.next();
        }

        let entry = match entry {
            Some(entry) => entry,
            None => return None,
        };

        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => return Some(Err(err)),
        };

        self.visit_stack.push(iter);

        let path = entry.path();
        if path.is_dir() {
            match fs::read_dir(path) {
                Ok(entries) => self.visit_stack.push(entries),
                Err(err) => return Some(Err(err)),
            }
        }

        Some(Ok(entry))
    }
}

fn read_dir_all<P: AsRef<Path>>(path: P) -> io::Result<ReadDirAll> {
    let top_dir = fs::read_dir(path)?;

    Ok(ReadDirAll { visit_stack: vec![top_dir] })
}

fn run_print_abnf() {
    let lsys_abnf = abnf::parse_file("grammar/lsys.abnf").expect("Could not parse ABNF file");
    println!("{:#?}", lsys_abnf);
}

fn run_random_genes() {
    let (mut window, _) = setup_window();

    let lsys_abnf = abnf::parse_file("grammar/lsys.abnf").expect("Could not parse ABNF file");

    let mut genotype = {
        let genome = generate_genome(&mut rand::thread_rng(), 100);
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

    let instructions = system.instructions_iter(settings.iterations, &settings.command_map);

    let mut model = lsys3d::build_model(instructions, &settings);
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

fn run_bush_inferred() {
    let (mut window, mut camera) = setup_window();

    let lsys_abnf = abnf::parse_file("grammar/bush.abnf").expect("Could not parse ABNF file");

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

    let instructions = system.instructions_iter(settings.iterations, &settings.command_map);

    let mut model = lsys3d::build_model(instructions, &settings);
    window.scene_mut().add_child(model.clone());

    while window.render_with_camera(&mut camera) {
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

#[derive(Clone)]
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
        assert!(self.index < self.genes.len(),
                "Genotype index overflows gene list");

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
    fn select_alternative(&mut self, num: usize, _: &[&str], _: u32) -> usize {
        let limit = Self::max_selection_value(num);
        let gene = self.use_next_gene();

        num::cast::<_, usize>(gene % limit).unwrap()
    }

    fn select_repetition(&mut self, min: u32, max: u32, _: &[&str], _: u32) -> u32 {
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

#[derive(Clone, Serialize, Deserialize)]
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

        let choices = self.depths[depth]
            .entry(rule.to_string())
            .or_insert_with(Vec::new);
        let choice = choice as usize;
        while choices.len() < choice + 1 {
            choices.push(Vec::new());
        }

        choices[choice] = weights.to_vec();
    }

    fn set_default_weights(&mut self, rule: &str, choice: u32, weights: &[f32]) {
        let choices = self.defaults
            .entry(rule.to_string())
            .or_insert_with(Vec::new);
        let choice = choice as usize;
        while choices.len() < choice + 1 {
            choices.push(Vec::new());
        }

        choices[choice] = weights.to_vec();
    }
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (depth, rules) in self.depths.iter().enumerate() {
            writeln!(f, "{}:", depth)?;

            for (rule, choices) in rules {
                let indent = "  ";
                writeln!(f, "{}{}:", indent, rule)?;

                for (choice, weights) in choices.iter().enumerate() {
                    let indent = indent.to_string() + "  ";
                    write!(f, "{}{}: ", indent, choice)?;

                    for weight in weights {
                        write!(f, "{}, ", weight)?;
                    }

                    writeln!(f)?;
                }
            }
        }

        writeln!(f, "Default:")?;
        for (rule, choices) in &self.defaults {
            let indent = "  ";
            writeln!(f, "{}{}:", indent, rule)?;

            for (choice, weights) in choices.iter().enumerate() {
                let indent = indent.to_string() + "  ";
                write!(f, "{}{}: ", indent, choice)?;

                for weight in weights {
                    write!(f, "{}, ", weight)?;
                }

                writeln!(f)?;
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
struct WeightedGenotype<'a, G> {
    genotype: Genotype<G>,
    distribution: &'a Distribution,
}

impl<'a, G: Gene> WeightedGenotype<'a, G> {
    fn new(genes: Vec<G>, distribution: &'a Distribution) -> WeightedGenotype<G> {
        WeightedGenotype {
            genotype: Genotype::new(genes),
            distribution: distribution,
        }
    }

    fn find_depth(rulechain: &[&str]) -> usize {
        rulechain
            .iter()
            .fold(0, |acc, r| if *r == "stack" { acc + 1 } else { acc })
    }
}

impl<'a, G: Gene> SelectionStrategy for WeightedGenotype<'a, G> {
    fn select_alternative(&mut self, num: usize, rulechain: &[&str], choice: u32) -> usize {
        let gene = self.genotype.use_next_gene();

        let depth = Self::find_depth(rulechain);
        let rule = rulechain.last().unwrap();
        let gene_frac = num::cast::<_, f32>(gene).unwrap() /
                        num::cast::<_, f32>(G::max_value()).unwrap();
        let weights = self.distribution.get_weights(depth, rule, choice);

        if let Some(weights) = weights {
            assert_eq!(weights.len(),
                       num,
                       "Number of weights does not match number of alternatives");
            weighted_selection(weights, gene_frac)
        } else {
            let weights = (0..num).map(|_| 1.0).collect::<Vec<_>>();
            weighted_selection(&weights, gene_frac)
        }
    }

    fn select_repetition(&mut self, min: u32, max: u32, rulechain: &[&str], choice: u32) -> u32 {
        let gene = self.genotype.use_next_gene();

        let num = max - min + 1;

        let depth = Self::find_depth(rulechain);
        let rule = rulechain.last().unwrap();
        let gene_frac = num::cast::<_, f32>(gene).unwrap() /
                        num::cast::<_, f32>(G::max_value()).unwrap();
        let weights = self.distribution.get_weights(depth, rule, choice);

        if let Some(weights) = weights {
            assert_eq!(weights.len(),
                       num as usize,
                       "Number of weights does not match number of repetition alternatives");
            min + weighted_selection(weights, gene_frac) as u32
        } else {
            let weights = (0..num).map(|_| 1.0).collect::<Vec<_>>();
            min + weighted_selection(&weights, gene_frac) as u32
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
            let num = strategy.select_repetition(min, max, &["productions"], 0);

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

fn infer_selections(expanded: &str,
                    grammar: &abnf::Ruleset,
                    root: &str)
                    -> Result<Vec<usize>, String> {
    let selection = infer_list_selections(&grammar[root], 0, expanded, grammar);

    match selection {
        Ok((list, index)) => {
            if index == expanded.len() {
                Ok(list)
            } else {
                Err(format!("Expanded string does not fully match grammar. \
                             The first {} characters matched",
                            index))
            }
        }
        Err(_) => Err("Expanded string does not match grammar".to_string()),
    }
}

// TODO: Need to be able to try new non-tested alternatives/repetitions if a previously matched
// alternative/repetition results in a mismatch later.
fn infer_list_selections(list: &abnf::List,
                         mut index: usize,
                         expanded: &str,
                         grammar: &abnf::Ruleset)
                         -> Result<(Vec<usize>, usize), ()> {
    use abnf::List;

    match *list {
        List::Sequence(ref sequence) => {
            let mut selections = vec![];

            for item in sequence {
                let (item_selections, updated_index) =
                    infer_item_selections(item, index, expanded, grammar)?;
                index = updated_index;
                selections.extend(item_selections);
            }

            Ok((selections, index))
        }
        List::Alternatives(ref alternatives) => {
            let mut selections = Vec::with_capacity(1);

            for (alternative, item) in alternatives.iter().enumerate() {
                if let Ok((item_selections, updated_index)) =
                    infer_item_selections(item, index, expanded, grammar) {
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
        }
    }
}

// TODO: Need to be able to try new non-tested alternatives/repetitions if a previously matched
// alternative/repetition results in a mismatch later.
fn infer_item_selections(item: &abnf::Item,
                         mut index: usize,
                         expanded: &str,
                         grammar: &abnf::Ruleset)
                         -> Result<(Vec<usize>, usize), ()> {
    use abnf::Content;

    let repeat = match item.repeat {
        Some(ref repeat) => {
            let min = repeat.min.unwrap_or(0);
            let max = repeat.max.unwrap_or(u32::max_value());

            Some((min, max))
        }
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
            }
            Content::Symbol(ref symbol) => {
                let child_result =
                    infer_list_selections(&grammar[symbol], index, expanded, grammar);
                if let Ok((child_selections, child_index)) = child_result {
                    selections.extend(child_selections);
                    index = child_index;
                    matched = true;
                }
            }
            Content::Group(ref group) => {
                let child_result = infer_list_selections(group, index, expanded, grammar);
                if let Ok((child_selections, child_index)) = child_result {
                    selections.extend(child_selections);
                    index = child_index;
                    matched = true;
                }
            }
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
        grammar.insert("symbol".to_string(), List::Sequence(vec![item.clone()]));

        assert_eq!(infer_item_selections(&item, 0, "value", &grammar),
                   Ok((vec![], 5)));
    }

    #[test]
    fn test_infer_selections_repeat_limits() {
        let item = Item::repeated(Content::Value("value".to_string()),
                                  Repeat::with_limits(2, 4));

        let mut grammar = Ruleset::new();
        grammar.insert("symbol".to_string(), List::Sequence(vec![item.clone()]));

        assert_eq!(infer_item_selections(&item, 0, "valuevaluevalue", &grammar),
                   Ok((vec![1], 15)));
    }

    #[test]
    fn test_infer_selections_alternatives() {
        let list = List::Alternatives(vec![Item::new(Content::Value("1".to_string())),
                                           Item::new(Content::Value("2".to_string())),
                                           Item::new(Content::Value("3".to_string()))]);

        let mut grammar = Ruleset::new();
        grammar.insert("symbol".to_string(), list.clone());

        assert_eq!(infer_list_selections(&list, 0, "2", &grammar),
                   Ok((vec![1], 1)));
    }

    #[test]
    fn test_infer_selections_match() {
        let item = Item::new(Content::Value("value".to_string()));

        let mut grammar = Ruleset::new();
        grammar.insert("symbol".to_string(), List::Sequence(vec![item.clone()]));

        assert_eq!(infer_selections("value", &grammar, "symbol"), Ok(vec![]));
    }

    #[test]
    fn test_infer_selections_match_alternatives() {
        let list = List::Alternatives(vec![Item::new(Content::Value("1".to_string())),
                                           Item::new(Content::Value("2".to_string())),
                                           Item::new(Content::Value("3".to_string()))]);

        let mut grammar = Ruleset::new();
        grammar.insert("symbol".to_string(), list.clone());

        assert_eq!(infer_selections("2", &grammar, "symbol"), Ok(vec![1]));
    }

    #[test]
    fn test_infer_selections_mismatch() {
        let item = Item::new(Content::Value("value".to_string()));

        let mut grammar = Ruleset::new();
        grammar.insert("symbol".to_string(), List::Sequence(vec![item.clone()]));

        assert_eq!(infer_selections("notvalue", &grammar, "symbol"),
                   Err("Expanded string does not match grammar".to_string()));
    }

    #[test]
    fn test_infer_selections_missing() {
        let item = Item::new(Content::Value("value".to_string()));

        let mut grammar = Ruleset::new();
        grammar.insert("symbol".to_string(), List::Sequence(vec![item.clone()]));

        assert_eq!(infer_selections("valueextra", &grammar, "symbol"),
                   Err("Expanded string does not fully match grammar. \
                 The first 5 characters matched"
                               .to_string()));
    }

    #[test]
    fn test_lsystem_ge_expansion() {
        let grammar = abnf::parse_file("grammar/lsys.abnf").expect("Could not parse ABNF file");
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

        assert_eq!(expand_grammar(&grammar, "axiom", &mut genotype), "F[FX]X");
    }

    #[test]
    fn test_lsystem_ge_inference() {
        let grammar = abnf::parse_file("grammar/lsys.abnf").expect("Could not parse ABNF file");
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

        assert_eq!(infer_selections("F[FX]X", &grammar, "axiom"), Ok(genes));
    }

    #[test]
    fn test_read_dir_all() {
        let entries_it = read_dir_all("testdata/read_dir_all").unwrap();
        let paths = entries_it.map(|e| e.unwrap().path()).collect::<Vec<_>>();
        let path_str = paths
            .iter()
            .map(|p| p.to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(path_str,
                   vec!["testdata/read_dir_all/a",
                        "testdata/read_dir_all/a/c",
                        "testdata/read_dir_all/a/c/d",
                        "testdata/read_dir_all/a/c/e",
                        "testdata/read_dir_all/a/b",
                        "testdata/read_dir_all/a/b/y",
                        "testdata/read_dir_all/a/b/z",
                        "testdata/read_dir_all/a/b/x"])
    }
}
