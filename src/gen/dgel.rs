use std::f32::consts::{PI, E};
use std::f32;
use std::{cmp, fmt, iter};
use std::io::{self, BufWriter, BufReader, Write, Read};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{self, AtomicBool, AtomicUsize};
use std::ops::Add;
use std::time::Instant;
use std::cmp::Ordering;
use std::cell::{Cell, RefCell};

use rand::{self, Rng, XorShiftRng, SeedableRng};
use rand::distributions::{IndependentSample, Range};
use rand::distributions::range::SampleRange;
use na::{UnitQuaternion, Point3};
use kiss3d::camera::ArcBall;
use kiss3d::scene::SceneNode;
use num::{self, Unsigned, NumCast};
use glfw::{Key, WindowEvent, Action};
use serde_yaml;
use num_cpus;
use futures::Future;
use futures::future::{self, FutureResult};
use futures_cpupool::CpuPool;
use bincode;
use crossbeam;
use clap::{App, SubCommand, Arg, ArgMatches};
use csv;
use chrono::prelude::*;

use abnf::{self, Grammar};
use abnf::expand::{SelectionStrategy, Rulechain, expand_grammar};
use lsys::{self, ol};
use lsys3d;
use lsystems;
use yobun::read_dir_all;
use setup_window;
use gen::fitness::{self, Fitness};

const DEPTHS: usize = 4;

pub const COMMAND_NAME: &'static str = "dgel";

pub fn get_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name(COMMAND_NAME)
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
            .arg(Arg::with_name("grammar")
                .short("g")
                .long("grammar")
                .takes_value(true)
                .default_value("grammar/lsys2.abnf")
                .help("Which ABNF grammar to use")
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
            .arg(Arg::with_name("threshold-type")
                .short("t")
                .long("threshold-type")
                .takes_value(true)
                .possible_values(&["crap", "zero"])
                .default_value("crap")
                .help("How to threshold which samples get accepted")
            )
            .arg(Arg::with_name("batch-size")
                .short("b")
                .long("batch_size")
                .takes_value(true)
                .default_value("128")
                .help("Number of accepted samples to accumulate before writing to file")
            )
        )
        .subcommand(SubCommand::with_name("sampling-dist")
            .about("Take samples from directory and output a distribution CSV file")
            .arg(Arg::with_name("SAMPLES")
                .required(true)
                .index(1)
                .help("Directory of samples to use")
            )
            .arg(Arg::with_name("distribution")
                .short("d")
                .long("distribution")
                .takes_value(true)
                .help("Distribution file to use. Otherwise default distribution is used.")
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
        .subcommand(SubCommand::with_name("stats")
            .about("Generate samples and dump stats to be analyzed")
            .arg(Arg::with_name("DISTRIBUTIONS")
                .required(true)
                .index(1)
                .multiple(true)
                .help("Distribution files to use")
            )
            .arg(Arg::with_name("csv")
                .long("csv")
                .takes_value(true)
                .default_value("stats.csv")
                .help("Name of the output CSV file")
            )
            .arg(Arg::with_name("num-samples")
                .short("n")
                .long("num-samples")
                .takes_value(true)
                .default_value("64")
                .help("Number of samples generated to create the stats")
            )
        )
        .subcommand(SubCommand::with_name("learning")
            .about("Run learning program until you type 'quit'")
            .arg(Arg::with_name("distribution")
                .short("d")
                .long("distribution")
                .takes_value(true)
                .help("Distribution file to use. Otherwise default distribution is used.")
            )
            .arg(Arg::with_name("csv")
                .long("csv")
                .takes_value(true)
                .default_value("distribution.csv")
                .help("Name of the output CSV file")
            )
            .arg(Arg::with_name("stats-csv")
                .long("stats-csv")
                .takes_value(true)
                .default_value("learning-stats.csv")
                .help("Name of the output stats CSV file")
            )
            .arg(Arg::with_name("bin")
                .long("bin")
                .takes_value(true)
                .default_value("distribution.bin")
                .help("Name of the output bincode file")
            )
            .arg(Arg::with_name("workers")
                .short("w")
                .long("workers")
                .takes_value(true)
                .help("How many workers (threads) to run. Default is number of CPU cores + 1.")
            )
            .arg(Arg::with_name("error-threshold")
                .long("error-threshold")
                .short("e")
                .takes_value(true)
                .default_value("0.002")
                .help("The required maximum standard error threshold for a distribution score \
                       to be accepted")
            )
            .arg(Arg::with_name("min-samples")
                .long("min-samples")
                .takes_value(true)
                .default_value("64")
                .help("Minimum number of samples to generate before calculating standard error")
            )
            .arg(Arg::with_name("fitness-scale")
                .long("fitness-scale")
                .short("s")
                .takes_value(true)
                .default_value("1")
                .help("A factor multiplied with the fitness when calculating the acceptance \
                       probability")
            )
            .arg(Arg::with_name("cooldown-rate")
                .long("cooldown-rate")
                .short("c")
                .takes_value(true)
                .default_value("0.1")
                .help("How fast to cool down the SA process. Higher values means faster cooldown. \
                       Range [0, 1).")
            )
            .arg(Arg::with_name("max-moves")
                .long("max-moves")
                .short("m")
                .takes_value(true)
                .default_value("1")
                .help("How many moves per dimension to perform until SA stops.")
            )
            .arg(Arg::with_name("no-dump")
                .long("no-dump")
                .help("Disable dumping of the distribution each iteration.")
            )
        )
        .subcommand(SubCommand::with_name("dist-csv-to-bin")
            .about("Convert a CSV distribution file to a bincode file")
            .arg(Arg::with_name("INPUT")
                .required(true)
                .index(1)
                .help("Distribution CSV file to convert")
            )
            .arg(Arg::with_name("output")
                .long("output")
                .short("o")
                .takes_value(true)
                .help("Name of the output bincode file. Defaults to INPUT with .csv extension")
            )
            .arg(Arg::with_name("version")
                .long("version")
                .short("v")
                .takes_value(true)
                .help("Version of Distribution structure to parse")
            )
            .arg(Arg::with_name("grammar")
                .long("grammar")
                .short("g")
                .takes_value(true)
                .required(true)
                .help("Grammar used for the Distribution, must be supplied for verion 1\
                       Distributions")
            )
        )
        .subcommand(SubCommand::with_name("sample-weight-space")
            .about("Sample a weight space and write points to file")
            .arg(Arg::with_name("DIMENSIONS")
                .required(true)
                .index(1)
                .help("The number of dimensions of the weight space to sample")
            )
            .arg(Arg::with_name("output")
                .long("output")
                .short("o")
                .takes_value(true)
                .default_value("weight-space.csv")
                .help("Name of the output csv file.")
            )
            .arg(Arg::with_name("num-samples")
                .long("num-samples")
                .short("n")
                .takes_value(true)
                .default_value("1000")
                .help("Number of samples to generate")
            )
            .arg(Arg::with_name("dividers")
                .long("dividers")
                .short("d")
                .takes_value(false)
                .help("Sample the dividers instead of the weights")
            )
        )
        .subcommand(SubCommand::with_name("ge")
            .about("Run grammatical evolution")
            .arg(Arg::with_name("distribution")
                .short("d")
                .long("distribution")
                .takes_value(true)
                .help("Distribution file to use. Otherwise default distribution is used.")
            )
            .arg(Arg::with_name("grammar")
                .short("g")
                .long("grammar")
                .takes_value(true)
                .default_value("grammar/lsys2.abnf")
                .help("Which ABNF grammar to use")
            )
        )
        .subcommand(SubCommand::with_name("bench")
            .about("Run benchmarks")
        )
}

pub fn run_dgel(matches: &ArgMatches) {
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
    } else if let Some(matches) = matches.subcommand_matches("stats") {
        run_stats(matches);
    } else if let Some(matches) = matches.subcommand_matches("learning") {
        run_learning(matches);
    } else if let Some(matches) = matches.subcommand_matches("dist-csv-to-bin") {
        run_distribution_csv_to_bin(matches);
    } else if let Some(matches) = matches.subcommand_matches("sample-weight-space") {
        run_sample_weight_space(matches);
    } else if let Some(matches) = matches.subcommand_matches("ge") {
        run_ge(matches);
    } else if let Some(matches) = matches.subcommand_matches("bench") {
        run_benchmark(matches);
    } else {
        println!("A subcommand must be specified. See help by passing -h.");
    }
}

type GenePrimitive = u32;
const CHROMOSOME_LEN: usize = 1024;

fn generate_chromosome<R: Rng>(rng: &mut R, len: usize) -> Vec<GenePrimitive> {
    let gene_range = Range::new(GenePrimitive::min_value(), GenePrimitive::max_value());

    let mut chromosome = Vec::with_capacity(len);
    for _ in 0..len {
        chromosome.push(gene_range.ind_sample(rng));
    }

    chromosome
}

fn generate_system<G>(grammar: &abnf::Grammar, genotype: &mut G) -> ol::LSystem
where
    G: SelectionStrategy,
{
    let mut system = ol::LSystem {
        axiom: expand_grammar(grammar, "axiom", genotype),
        productions: expand_productions(grammar, genotype),
    };

    system.remove_redundancy();

    system
}

fn random_seed() -> [u32; 4] {
    [
        rand::thread_rng().gen::<u32>(),
        rand::thread_rng().gen::<u32>(),
        rand::thread_rng().gen::<u32>(),
        rand::thread_rng().gen::<u32>(),
    ]
}

fn run_with_distribution(matches: &ArgMatches) {
    let (mut window, _) = setup_window();

    let (grammar, distribution) = get_sample_setup(matches.value_of("grammar").unwrap());
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
                    let filename = format!("{}.yaml", Local::now().to_rfc3339());
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

                    fn generate_sample(
                        grammar: &abnf::Grammar,
                        distribution: &Distribution,
                        settings: &lsys::Settings,
                    ) -> Sample {
                        let seed = random_seed();
                        let chromosome =
                            generate_chromosome(&mut XorShiftRng::from_seed(seed), CHROMOSOME_LEN);
                        let system = generate_system(
                            grammar,
                            &mut WeightedGenotype::new(
                                chromosome,
                                distribution,
                                grammar.symbol_index("stack").unwrap(),
                            ),
                        );
                        let (fit, _) = fitness::evaluate(&system, settings);
                        Sample {
                            seed: seed,
                            score: fit.score(),
                        }
                    }

                    println!("Generating {} samples...", num_samples);
                    let start_time = Instant::now();

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
                                        generate_sample(&grammar, &distribution, &settings);
                                    future::ok::<Sample, ()>(sample)
                                }));
                            }

                            future::join_all(tasks).wait().unwrap()
                        }
                    };

                    let end_time = Instant::now();
                    let duration = end_time - start_time;
                    println!(
                        "Duration: {}.{}",
                        duration.as_secs(),
                        duration.subsec_nanos()
                    );

                    samples.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

                    let sample = samples.pop().unwrap();

                    println!("Found sample with score {}.", sample.score);
                    println!("Seed: {:?}", sample.seed);
                    println!("Building model...");

                    window.remove(&mut model);

                    let chromosome = generate_chromosome(
                        &mut XorShiftRng::from_seed(sample.seed),
                        CHROMOSOME_LEN,
                    );
                    system = generate_system(
                        &grammar,
                        &mut WeightedGenotype::new(
                            chromosome,
                            &distribution,
                            grammar.symbol_index("stack").unwrap(),
                        ),
                    );
                    let (fit, properties) = fitness::evaluate(&system, &settings);

                    if let Some(properties) = properties {
                        println!("{} points.", properties.num_points);
                        let instructions =
                            system.instructions_iter(settings.iterations, &settings.command_map);
                        model = lsys3d::build_model(instructions, &settings);
                        fitness::add_properties_rendering(&mut model, &properties);
                        window.scene_mut().add_child(model.clone());

                        println!("");
                        println!("LSystem:");
                        println!("{}", system);

                        println!("{:#?}", properties);
                        println!("Fitness: {}", fit);

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
                        let (fit, properties) = fitness::evaluate(&system, &settings);
                        println!("{:#?}", properties);
                        println!("Fitness: {}", fit);

                        if let Some(properties) = properties {
                            window.remove(&mut model);
                            model = lsys3d::build_model(instructions, &settings);
                            fitness::add_properties_rendering(&mut model, &properties);
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
                    let (score, properties) = fitness::evaluate(&system, &settings);
                    println!("Score: {}", score);

                    window.remove(&mut model);
                    model = lsys3d::build_model(instructions, &settings);
                    fitness::add_properties_rendering(&mut model, &properties.unwrap());
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

fn get_sample_setup(grammar_path: &str) -> (abnf::Grammar, Distribution) {
    let grammar = abnf::parse_file(grammar_path).expect("Could not parse ABNF file");
    let string_index = grammar.symbol_index("string").expect(
        "Grammar does not contain 'string' symbol",
    );
    let distribution = {
        let mut distribution = Distribution::new();
        for d in 0..DEPTHS - 1 {
            distribution.set_weights(d, string_index, 1, &[1.0, 1.0]);
        }
        distribution.set_default_weights(string_index, 1, &[1.0, 0.0]);

        distribution
    };

    (grammar, distribution)
}

fn run_random_sampling(matches: &ArgMatches) {
    fn generate_sample(
        grammar: &abnf::Grammar,
        distribution: &Distribution,
    ) -> ([u32; 4], ol::LSystem) {
        let seed = random_seed();
        let chromosome = generate_chromosome(&mut XorShiftRng::from_seed(seed), CHROMOSOME_LEN);
        let system = generate_system(
            grammar,
            &mut WeightedGenotype::new(
                chromosome,
                distribution,
                grammar.symbol_index("stack").unwrap(),
            ),
        );
        (seed, system)
    }

    fn accept_not_crap(lsystem: &ol::LSystem, settings: &lsys::Settings) -> bool {
        !fitness::is_crap(lsystem, settings)
    }

    fn accept_above_zero(lsystem: &ol::LSystem, settings: &lsys::Settings) -> bool {
        fitness::evaluate(lsystem, settings).0.score() >= 0.0
    }

    let accept_sample = {
        let threshold_type = matches.value_of("threshold-type").unwrap();
        if threshold_type == "crap" {
            println!("Using not crap threshold.");
            accept_not_crap
        } else if threshold_type == "zero" {
            println!("Using above zero threshold.");
            accept_above_zero
        } else {
            panic!(format!("Unrecognized threshold-type: {}", threshold_type));
        }
    };

    let (grammar, distribution) = get_sample_setup("grammar/lsys2.abnf");

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

    let settings = Arc::new(lsys::Settings {
        width: 0.05,
        angle: PI / 8.0,
        iterations: 5,
        ..lsys::Settings::new()
    });

    let batch_size = usize::from_str_radix(matches.value_of("batch-size").unwrap(), 10).unwrap();
    println!("Using batch size {}.", batch_size);

    const SEQUENCE_SIZE: usize = 16;

    let sample_dir = Path::new("sample").join(Local::now().to_rfc3339());
    fs::create_dir_all(&sample_dir).unwrap();

    println!("Starting random sampling...");

    let num_workers = num_cpus::get() + 1;
    let work = Arc::new(AtomicBool::new(true));
    let num_samples = Arc::new(AtomicUsize::new(0));
    let num_good_samples = Arc::new(AtomicUsize::new(0));

    let start_time = Instant::now();

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
                let dump_samples = |accepted_samples: &[[u32; 4]],
                                    sample_count: usize,
                                    batch: usize| {
                    let filename = format!("{}.{}.sample", worker_id, batch);
                    let path = sample_dir.join(filename);
                    let file = File::create(&path).unwrap();

                    let samples = SampleBatch {
                        sample_count: sample_count,
                        accepted: accepted_samples.to_vec(),
                    };
                    bincode::serialize_into(&mut BufWriter::new(file), &samples, bincode::Infinite)
                        .unwrap();
                };

                // Room for 0.5% of samples.
                let mut accepted_samples = Vec::with_capacity(batch_size / 200);
                let mut batch = 0;
                let mut batch_num_samples = 0;

                while work.load(atomic::Ordering::Relaxed) {
                    for _ in 0..SEQUENCE_SIZE {
                        let (seed, lsystem) = generate_sample(&grammar, &distribution);
                        if accept_sample(&lsystem, &settings) {
                            accepted_samples.push(seed);
                        }
                    }

                    batch_num_samples += SEQUENCE_SIZE;

                    if accepted_samples.len() >= batch_size {
                        dump_samples(&accepted_samples, batch_num_samples, batch);

                        num_samples.fetch_add(batch_num_samples, atomic::Ordering::Relaxed);

                        accepted_samples.clear();
                        batch += 1;
                        batch_num_samples = 0;
                    }
                }

                dump_samples(&accepted_samples, batch_num_samples, batch);
                num_samples.fetch_add(batch_num_samples, atomic::Ordering::Relaxed);
                num_good_samples.fetch_add(
                    batch * batch_size + accepted_samples.len(),
                    atomic::Ordering::Relaxed,
                );
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
        work.store(false, atomic::Ordering::Relaxed);
    });

    let end_time = Instant::now();
    let duration = end_time - start_time;
    let seconds = duration.as_secs();
    let hours = seconds / (60 * 60);
    let minutes = seconds % (60 * 60) / 60;
    println!(
        "Duration: {}:{}:{}.{}",
        hours,
        minutes,
        seconds,
        duration.subsec_nanos()
    );

    let num_samples = Arc::try_unwrap(num_samples).unwrap().into_inner();
    let num_good_samples = Arc::try_unwrap(num_good_samples).unwrap().into_inner();
    println!(
        "Good samples: {}/{} ({:.*}%)",
        num_good_samples,
        num_samples,
        1,
        num_good_samples as f32 / num_samples as f32 * 100.0
    );
}

fn run_sampling_distribution(matches: &ArgMatches) {
    let samples_path = Path::new(matches.value_of("SAMPLES").unwrap());
    let csv_path = Path::new(matches.value_of("csv").unwrap());
    let bin_path = Path::new(matches.value_of("bin").unwrap());

    println!("Reading samples from {}.", samples_path.to_str().unwrap());

    let sample_paths = read_dir_all(samples_path).unwrap().filter_map(|e| {
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

    println!(
        "Read {} accepted samples from a total of {} samples.",
        accepted_samples.len(),
        sample_count
    );

    let (grammar, distribution) = get_sample_setup("grammar/lsys2.abnf");
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

    let workers = num_cpus::get() + 1;
    let pool = CpuPool::new(workers);
    let mut tasks = Vec::with_capacity(accepted_samples.len());

    for seed in accepted_samples {
        let distribution = distribution.clone();
        let grammar = grammar.clone();

        tasks.push(pool.spawn_fn(move || {
            let chromosome = generate_chromosome(&mut XorShiftRng::from_seed(seed), CHROMOSOME_LEN);
            let mut stats_genotype = WeightedGenotypeStats::new(
                chromosome,
                &distribution,
                grammar.symbol_index("stack").unwrap(),
            );
            expand_grammar(&grammar, "axiom", &mut stats_genotype);
            expand_productions(&grammar, &mut stats_genotype);

            future::ok::<SelectionStats, ()>(stats_genotype.take_stats())
        }));
    }

    let stats_collection = future::join_all(tasks).wait().unwrap();
    let stats = stats_collection.iter().fold(
        SelectionStats::new(),
        |sum, stats| sum + stats,
    );

    let mut csv_file = File::create(csv_path).unwrap();
    csv_file
        .write_all(stats.to_csv_normalized().as_bytes())
        .unwrap();

    let string_index = grammar.symbol_index("string").expect(
        "Grammar does not contain 'string' symbol",
    );
    let mut distribution = stats.to_distribution();
    distribution.set_default_weights(string_index, 1, &[1.0, 0.0]);

    // Remove the default weights from the depth weights.
    distribution.depths[DEPTHS - 1][string_index].pop();

    let dist_file = File::create(bin_path).unwrap();
    bincode::serialize_into(
        &mut BufWriter::new(dist_file),
        &distribution,
        bincode::Infinite,
    ).unwrap();
}

#[allow(dead_code)]
fn adjust_distribution(distribution: &mut Distribution, stats: &SelectionStats, factor: f32) {
    for (depth, rules) in stats.data.iter().enumerate() {
        for (rule_index, choices) in rules.iter().enumerate() {
            for (choice, options) in choices.iter().enumerate() {
                let weights = distribution.get_weights_mut(depth, rule_index, choice as u32);
                if let Some(weights) = weights {
                    assert_eq!(
                        weights.len(),
                        options.len(),
                        "Stats has different number of weights than distribution"
                    );
                    let total_count = options.iter().sum::<usize>() as f32;
                    for (option, count) in options.iter().enumerate() {
                        if *count > 0 {
                            let count_factor = *count as f32 / total_count;
                            let combined_factor = 1.0 + (factor - 1.0) * count_factor;
                            weights[option] *= combined_factor;

                            if weights[option].is_nan() {
                                panic!("NAN");
                            }
                        }
                    }
                }
            }
        }
    }
}

fn mutate_distribution<R>(distribution: &mut Distribution, rng: &mut R)
where
    R: Rng,
{
    for rules in &mut distribution.depths {
        for choices in rules {
            for mut options in choices {
                let i = Range::new(0, options.len()).ind_sample(rng);
                let u = Range::new(0.0, 1.0).ind_sample(rng);

                let old = options[i];
                let new = if u < 0.5 {
                    old + u * (1.0 - old)
                } else if u > 0.5 {
                    old - u * old
                } else {
                    old
                };

                let expected_remaining_sum = 1.0 - new;
                options[i] = new;
                let remaining_sum: f32 = options
                    .iter()
                    .take(i)
                    .chain(options.iter().skip(i + 1))
                    .sum();

                if remaining_sum == 0.0 {
                    // The expected remaining sum must be divided over the remaining weights.
                    let new_value = expected_remaining_sum / (options.len() - 1) as f32;
                    modify_remaining(&mut options, i, |w| *w = new_value);
                } else {
                    // The remaining options are normalized to sum up to remaining_sum.
                    let normalization_factor = expected_remaining_sum / remaining_sum;
                    modify_remaining(&mut options, i, |w| *w *= normalization_factor);
                }

                fn modify_remaining<F>(weights: &mut [f32], changed_index: usize, mut f: F)
                where
                    F: FnMut(&mut f32),
                {
                    // Can't chain mut iterators, so need to have two for loops.
                    for w in weights.iter_mut().take(changed_index) {
                        f(w);
                    }
                    for w in weights.iter_mut().skip(changed_index + 1) {
                        f(w);
                    }
                }
            }
        }
    }
}

type SelectionChoice = Vec<usize>;
type SelectionRule = Vec<SelectionChoice>;
type SelectionDepth = Vec<SelectionRule>;

#[derive(Clone)]
struct SelectionStats {
    data: Vec<SelectionDepth>,
}

impl SelectionStats {
    fn new() -> SelectionStats {
        SelectionStats { data: Vec::new() }
    }

    fn make_room(&mut self, depth: usize, rule_index: usize, choice: u32, num: usize) {
        while self.data.len() <= depth {
            self.data.push(Vec::new());
        }

        let rules = &mut self.data[depth];
        while rules.len() <= rule_index {
            rules.push(Vec::new());
        }

        let choices = &mut rules[rule_index];
        let choice = choice as usize;
        while choices.len() <= choice {
            choices.push(Vec::new());
        }

        let alternatives = &mut choices[choice];
        while alternatives.len() < num {
            alternatives.push(0);
        }
    }

    fn add_selection(&mut self, selection: usize, depth: usize, rule_index: usize, choice: u32) {
        assert!(
            depth < self.data.len(),
            format!(
                "Depth {} is not available. Use make_room to make it.",
                depth
            )
        );
        let rules = &mut self.data[depth];

        assert!(
            rule_index < rules.len(),
            format!(
                "Rule index {} is not available. Use make_room to make it.",
                rule_index
            )
        );

        let choices = &mut rules[rule_index];

        let choice = choice as usize;
        assert!(
            choice < choices.len(),
            format!(
                "Choice {} is not available. Use make_room to make it.",
                choice
            )
        );
        let alternatives = &mut choices[choice];

        assert!(
            selection < alternatives.len(),
            format!(
                "Alternative {} is not available. Use make_room to make it.",
                selection
            )
        );
        alternatives[selection] += 1;
    }

    fn to_csv_normalized(&self) -> String {
        let mut csv = "depth,rule,choice,alternative,weight\n".to_string();
        for (depth, rules) in self.data.iter().enumerate() {
            for (rule_index, choices) in rules.iter().enumerate() {
                for (choice, alternatives) in choices.iter().enumerate() {
                    let mut total = 0;
                    for count in alternatives {
                        total += *count;
                    }

                    for (alternative, count) in alternatives.iter().enumerate() {
                        let weight = *count as f32 / total as f32;
                        csv += &format!(
                            "{},{},{},{},{}\n",
                            depth,
                            rule_index,
                            choice,
                            alternative,
                            weight
                        );
                    }
                }
            }
        }

        csv
    }

    fn to_distribution(&self) -> Distribution {
        let mut distribution = Distribution::new();

        for (depth, rules) in self.data.iter().enumerate() {
            for (rule_index, choices) in rules.iter().enumerate() {
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

                    distribution.set_weights(depth, rule_index, choice as u32, &weights);
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
            self.data.push(Vec::new());
        }

        for (depth, other_rules) in other.data.iter().enumerate() {
            let rules = &mut self.data[depth];
            for (rule_index, other_choices) in other_rules.iter().enumerate() {
                while rules.len() < other_rules.len() {
                    rules.push(Vec::new());
                }
                let choices = &mut rules[rule_index];

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

impl iter::Sum for SelectionStats {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let mut sum = match iter.next() {
            Some(stats) => stats,
            None => return SelectionStats::new(),
        };

        for stats in iter {
            sum = sum + stats;
        }

        sum
    }
}

impl<'a> iter::Sum<&'a Self> for SelectionStats {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        let mut sum = match iter.next() {
            Some(stats) => stats.clone(),
            None => return SelectionStats::new(),
        };

        for stats in iter {
            sum = sum + stats;
        }

        sum
    }
}

struct WeightedGenotypeStats<'a, G> {
    weighted_genotype: WeightedGenotype<'a, G>,
    stats: SelectionStats,
}

impl<'a, G: Gene> WeightedGenotypeStats<'a, G> {
    fn new(
        chromosome: Vec<G>,
        distribution: &'a Distribution,
        stack_rule_index: usize,
    ) -> WeightedGenotypeStats<'a, G> {
        WeightedGenotypeStats {
            weighted_genotype: WeightedGenotype::new(chromosome, distribution, stack_rule_index),
            stats: SelectionStats::new(),
        }
    }

    fn take_stats(self) -> SelectionStats {
        self.stats
    }
}

impl<'a, G: Gene> SelectionStrategy for WeightedGenotypeStats<'a, G> {
    fn select_alternative(&mut self, num: usize, rulechain: &Rulechain, choice: u32) -> usize {
        let selection = self.weighted_genotype.select_alternative(
            num,
            rulechain,
            choice,
        );

        let depth = self.weighted_genotype.find_depth(rulechain);
        let rule = rulechain.last().unwrap();

        if self.weighted_genotype.distribution.has_weights(
            depth,
            rule.index.unwrap(),
            choice,
        )
        {
            self.stats.make_room(
                depth,
                rule.index.unwrap(),
                choice,
                num,
            );
            self.stats.add_selection(
                selection,
                depth,
                rule.index.unwrap(),
                choice,
            );
        }

        selection
    }

    fn select_repetition(&mut self, min: u32, max: u32, rulechain: &Rulechain, choice: u32) -> u32 {
        let selection = self.weighted_genotype.select_repetition(
            min,
            max,
            rulechain,
            choice,
        );

        let depth = self.weighted_genotype.find_depth(rulechain);
        let rule = rulechain.last().unwrap();
        let num = (max - min + 1) as usize;

        if self.weighted_genotype.distribution.has_weights(
            depth,
            rule.index.unwrap(),
            choice,
        )
        {
            self.stats.make_room(
                depth,
                rule.index.unwrap(),
                choice,
                num,
            );
            self.stats.add_selection(
                (selection - min) as usize,
                depth,
                rule.index.unwrap(),
                choice,
            );
        }

        selection
    }
}

fn run_print_abnf() {
    let lsys_abnf = abnf::parse_file("grammar/lsys.abnf").expect("Could not parse ABNF file");
    println!("{:#?}", lsys_abnf);
}

fn run_random_genes() {
    let (mut window, _) = setup_window();

    let lsys_abnf = abnf::parse_file("grammar/lsys.abnf").expect("Could not parse ABNF file");

    let mut genotype = {
        let chromosome = generate_chromosome(&mut rand::thread_rng(), 100);
        Genotype::new(chromosome)
    };

    println!("Genes: {:?}", genotype.chromosome);
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

trait MinValue<V> {
    fn min_value() -> V;
}

macro_rules! impl_min_value {
    ($t:ident) => {
        impl MinValue<Self> for $t {
            fn min_value() -> Self {
                ::std::$t::MIN
            }
        }
    };
}

impl_min_value!(i8);
impl_min_value!(i16);
impl_min_value!(i32);
impl_min_value!(i64);
impl_min_value!(isize);
impl_min_value!(u8);
impl_min_value!(u16);
impl_min_value!(u32);
impl_min_value!(u64);
impl_min_value!(usize);
impl_min_value!(f32);
impl_min_value!(f64);

trait Gene
    : Unsigned + NumCast + Clone + Copy + MaxValue<Self> + MinValue<Self> + SampleRange + PartialOrd
    {
}

impl Gene for u8 {}
impl Gene for u16 {}
impl Gene for u32 {}
impl Gene for u64 {}
impl Gene for usize {}

#[derive(Clone)]
struct Genotype<G> {
    chromosome: Vec<G>,
    index: usize,
}

impl<G: Gene> Genotype<G> {
    fn new(chromosome: Vec<G>) -> Genotype<G> {
        Genotype {
            chromosome: chromosome,
            index: 0,
        }
    }

    fn use_next_gene(&mut self) -> G {
        assert!(
            self.index < self.chromosome.len(),
            "Genotype index overflows gene list"
        );

        let gene = self.chromosome[self.index];
        self.index = (self.index + 1) % self.chromosome.len();

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
    fn select_alternative(&mut self, num: usize, _: &Rulechain, _: u32) -> usize {
        let limit = Self::max_selection_value(num);
        let gene = self.use_next_gene();

        num::cast::<_, usize>(gene % limit).unwrap()
    }

    fn select_repetition(&mut self, min: u32, max: u32, _: &Rulechain, _: u32) -> u32 {
        let limit = Self::max_selection_value(max - min + 1);
        let gene = self.use_next_gene();

        num::cast::<_, u32>(gene % limit).unwrap() + min
    }
}

fn weighted_selection(weights: &[f32], selector: f32) -> usize {
    let total_weight: f32 = weights.iter().sum();
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
    // [depth][rule][choice][option]
    depths: Vec<Vec<Vec<Vec<f32>>>>,
    // [rule][choice][option]
    defaults: Vec<Vec<Vec<f32>>>,
}

impl Distribution {
    fn new() -> Distribution {
        Distribution {
            depths: Vec::new(),
            defaults: Vec::new(),
        }
    }

    fn has_weights(&self, depth: usize, rule_index: usize, choice_num: u32) -> bool {
        if depth < self.depths.len() {
            let rules = &self.depths[depth];
            if rule_index < rules.len() {
                let choices = &rules[rule_index];
                if (choice_num as usize) < choices.len() {
                    let weights = &choices[choice_num as usize];
                    if !weights.is_empty() {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn get_weights_mut(
        &mut self,
        depth: usize,
        rule_index: usize,
        choice_num: u32,
    ) -> Option<&mut [f32]> {
        if depth < self.depths.len() {
            let rules = &mut self.depths[depth];
            if rule_index < rules.len() {
                let choices = &mut rules[rule_index];
                if (choice_num as usize) < choices.len() {
                    let weights = &mut choices[choice_num as usize];
                    if !weights.is_empty() {
                        return Some(weights);
                    }
                }
            }
        }

        None
    }

    fn get_weights(&self, depth: usize, rule_index: usize, choice_num: u32) -> Option<&[f32]> {
        if depth < self.depths.len() {
            let rules = &self.depths[depth];
            if rule_index < rules.len() {
                let choices = &rules[rule_index];
                if (choice_num as usize) < choices.len() {
                    let weights = &choices[choice_num as usize];
                    if !weights.is_empty() {
                        return Some(weights);
                    }
                }
            }
        }

        if rule_index < self.defaults.len() {
            let choices = &self.defaults[rule_index];
            if (choice_num as usize) < choices.len() {
                let weights = &choices[choice_num as usize];
                if !weights.is_empty() {
                    return Some(weights);
                }
            }
        }

        None
    }

    fn set_weight(
        &mut self,
        depth: usize,
        rule_index: usize,
        choice: u32,
        alternative: usize,
        weight: f32,
    ) {
        while self.depths.len() < depth + 1 {
            self.depths.push(Vec::new());
        }

        let rules = &mut self.depths[depth];
        while rules.len() < rule_index + 1 {
            rules.push(Vec::new());
        }

        let choices = &mut rules[rule_index];
        let choice = choice as usize;
        while choices.len() < choice + 1 {
            choices.push(Vec::new());
        }

        let alternatives = &mut choices[choice];
        while alternatives.len() < alternative + 1 {
            alternatives.push(f32::NAN);
        }

        alternatives[alternative] = weight;
    }

    fn set_weights(&mut self, depth: usize, rule_index: usize, choice: u32, weights: &[f32]) {
        while self.depths.len() < depth + 1 {
            self.depths.push(Vec::new());
        }

        let rules = &mut self.depths[depth];
        while rules.len() < rule_index + 1 {
            rules.push(Vec::new());
        }

        let choices = &mut rules[rule_index];
        let choice = choice as usize;
        while choices.len() < choice + 1 {
            choices.push(Vec::new());
        }

        choices[choice] = weights.to_vec();
    }

    fn set_default_weights(&mut self, rule_index: usize, choice: u32, weights: &[f32]) {
        while self.defaults.len() < rule_index + 1 {
            self.defaults.push(Vec::new());
        }

        let choices = &mut self.defaults[rule_index];
        let choice = choice as usize;
        while choices.len() < choice + 1 {
            choices.push(Vec::new());
        }

        choices[choice] = weights.to_vec();
    }

    #[allow(dead_code)]
    fn normalize(&mut self) {
        for rules in &mut self.depths {
            for choices in rules {
                for alternatives in choices {
                    let total: f32 = alternatives.iter().sum();
                    for weight in alternatives {
                        *weight /= total as f32;
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    fn into_normalized(mut self) -> Distribution {
        self.normalize();
        self
    }

    fn dimensions(&self) -> usize {
        self.depths.iter().fold(0, |count, ref rules| {
            count + rules.iter().fold(0, |count, ref choices| {
                count + choices.iter().fold(0, |count, ref alternatives| {
                    count + alternatives.len()
                })
            })
        })
    }

    fn to_csv(&self) -> String {
        let mut csv = "depth,rule,choice,alternative,weight\n".to_string();
        for (depth, rules) in self.depths.iter().enumerate() {
            for (rule, choices) in rules.iter().enumerate() {
                for (choice, alternatives) in choices.iter().enumerate() {
                    let mut total = 0.0;
                    for weight in alternatives {
                        total += *weight;
                    }

                    for (alternative, weight) in alternatives.iter().enumerate() {
                        let weight = *weight / total as f32;
                        csv +=
                            &format!("{},{},{},{},{}\n", depth, rule, choice, alternative, weight);
                    }
                }
            }
        }

        csv
    }

    fn from_csv(csv: &str) -> Distribution {
        let mut reader = csv::Reader::from_string(csv).has_headers(true);
        let mut dist = Distribution::new();

        for row in reader.decode() {
            let (depth, rule, choice, alternative, weight): (usize,
                                                             usize,
                                                             u32,
                                                             usize,
                                                             f32) = row.unwrap();
            dist.set_weight(depth, rule, choice, alternative, weight);
        }

        dist
    }

    fn from_csv_v1(csv: &str, grammar: &Grammar) -> Distribution {
        let mut reader = csv::Reader::from_string(csv).has_headers(true);
        let mut dist = Distribution::new();

        for row in reader.decode() {
            let (depth, rule, choice, alternative, weight): (usize,
                                                             String,
                                                             u32,
                                                             usize,
                                                             f32) = row.unwrap();
            let rule_index = grammar.symbol_index(&rule).expect(&format!(
                "Symbol '{}' not found in grammar",
                rule
            ));
            dist.set_weight(depth, rule_index, choice, alternative, weight);
        }

        dist
    }
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (depth, rules) in self.depths.iter().enumerate() {
            writeln!(f, "{}:", depth)?;

            for (rule, choices) in rules.iter().enumerate() {
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
        for (rule, choices) in self.defaults.iter().enumerate() {
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
    stack_rule_index: usize,
}

impl<'a, G: Gene> WeightedGenotype<'a, G> {
    fn new(
        chromosome: Vec<G>,
        distribution: &'a Distribution,
        stack_rule_index: usize,
    ) -> WeightedGenotype<G> {
        WeightedGenotype {
            genotype: Genotype::new(chromosome),
            distribution: distribution,
            stack_rule_index: stack_rule_index,
        }
    }

    fn find_depth(&self, rulechain: &Rulechain) -> usize {
        rulechain.iter().fold(
            0,
            |acc, r| if let Some(index) = r.index {
                if index == self.stack_rule_index {
                    acc + 1
                } else {
                    acc
                }
            } else {
                acc
            },
        )
    }
}

impl<'a, G: Gene> SelectionStrategy for WeightedGenotype<'a, G> {
    fn select_alternative(&mut self, num: usize, rulechain: &Rulechain, choice: u32) -> usize {
        let gene = self.genotype.use_next_gene();

        let depth = self.find_depth(rulechain);
        let rule = rulechain.last().unwrap();
        let gene_frac = num::cast::<_, f32>(gene).unwrap() /
            num::cast::<_, f32>(G::max_value()).unwrap();
        let weights = self.distribution.get_weights(
            depth,
            rule.index.unwrap(),
            choice,
        );

        if let Some(weights) = weights {
            assert_eq!(
                weights.len(),
                num,
                "Number of weights does not match number of alternatives"
            );
            weighted_selection(weights, gene_frac)
        } else {
            let weights = (0..num).map(|_| 1.0).collect::<Vec<_>>();
            weighted_selection(&weights, gene_frac)
        }
    }

    fn select_repetition(&mut self, min: u32, max: u32, rulechain: &Rulechain, choice: u32) -> u32 {
        let gene = self.genotype.use_next_gene();

        let num = max - min + 1;

        let depth = self.find_depth(rulechain);
        let rule = rulechain.last().unwrap();
        let gene_frac = num::cast::<_, f32>(gene).unwrap() /
            num::cast::<_, f32>(G::max_value()).unwrap();
        let weights = self.distribution.get_weights(
            depth,
            rule.index.unwrap(),
            choice,
        );

        if let Some(weights) = weights {
            assert_eq!(
                weights.len(),
                num as usize,
                "Number of weights does not match number of repetition alternatives"
            );
            min + weighted_selection(weights, gene_frac) as u32
        } else {
            let weights = (0..num).map(|_| 1.0).collect::<Vec<_>>();
            min + weighted_selection(&weights, gene_frac) as u32
        }
    }
}

// Somehow make the below expansion functions a generic part of abnf::expand?

fn expand_productions<T>(grammar: &abnf::Grammar, strategy: &mut T) -> ol::RuleMap
where
    T: SelectionStrategy,
{
    let productions_symbol = grammar.symbol("productions");
    let mut rules = ol::RuleMap::new();
    let list = grammar.map_rule(&productions_symbol).unwrap();

    if let abnf::List::Sequence(ref seq) = *list {
        assert_eq!(seq.len(), 1);
        let item = &seq[0];

        if let abnf::Content::Symbol(_) = item.content {
            let repeat = item.repeat.unwrap_or_default();
            let min = repeat.min.unwrap_or(0);
            let max = repeat.max.unwrap_or(u32::max_value());
            let num = strategy.select_repetition(min, max, &[&productions_symbol], 0);

            for _ in 0..num {
                let (pred, succ) = expand_production(grammar, strategy);
                rules[pred] = succ;
            }
        }
    }

    rules
}

fn expand_production<T>(grammar: &abnf::Grammar, strategy: &mut T) -> (char, String)
where
    T: SelectionStrategy,
{
    let list = grammar.map_rule_from_name("production").unwrap();

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

fn expand_predecessor<T>(grammar: &abnf::Grammar, strategy: &mut T) -> char
where
    T: SelectionStrategy,
{
    let value = expand_grammar(grammar, "predecessor", strategy);
    assert_eq!(value.len(), 1);
    value.as_bytes()[0] as char
}

fn expand_successor<T>(grammar: &abnf::Grammar, strategy: &mut T) -> String
where
    T: SelectionStrategy,
{
    expand_grammar(grammar, "successor", strategy)
}

fn infer_selections(
    expanded: &str,
    grammar: &abnf::Grammar,
    root: &str,
) -> Result<Vec<usize>, String> {
    let selection = infer_list_selections(
        grammar.map_rule_from_name(root).unwrap(),
        0,
        expanded,
        grammar,
    );

    match selection {
        Ok((list, index)) => {
            if index == expanded.len() {
                Ok(list)
            } else {
                Err(format!(
                    "Expanded string does not fully match grammar. \
                             The first {} characters matched",
                    index
                ))
            }
        }
        Err(_) => Err("Expanded string does not match grammar".to_string()),
    }
}

// TODO: Need to be able to try new non-tested alternatives/repetitions if a previously matched
// alternative/repetition results in a mismatch later.
fn infer_list_selections(
    list: &abnf::List,
    mut index: usize,
    expanded: &str,
    grammar: &abnf::Grammar,
) -> Result<(Vec<usize>, usize), ()> {
    use abnf::List;

    match *list {
        List::Sequence(ref sequence) => {
            let mut selections = Vec::new();

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
                    infer_item_selections(item, index, expanded, grammar)
                {
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
fn infer_item_selections(
    item: &abnf::Item,
    mut index: usize,
    expanded: &str,
    grammar: &abnf::Grammar,
) -> Result<(Vec<usize>, usize), ()> {
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

    let mut selections = Vec::new();
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
                let child_result = infer_list_selections(
                    grammar.map_rule(symbol).unwrap(),
                    index,
                    expanded,
                    grammar,
                );
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

fn run_stats(matches: &ArgMatches) {
    let grammar = Arc::new(abnf::parse_file("grammar/lsys2.abnf").expect(
        "Could not parse ABNF file",
    ));

    let distributions: Vec<Arc<Distribution>> = {
        let paths = matches.values_of("DISTRIBUTIONS").expect(
            "No distributions specified",
        );

        println!("Distributions: {:?}", paths.clone().collect::<Vec<_>>());

        paths
            .map(|p| {
                let file = File::open(p).unwrap();
                let distribution =
                    bincode::deserialize_from(&mut BufReader::new(file), bincode::Infinite)
                        .unwrap();
                Arc::new(distribution)
            })
            .collect()
    };

    let num_samples = usize::from_str_radix(matches.value_of("num-samples").unwrap(), 10).unwrap();
    let csv_path = Path::new(matches.value_of("csv").unwrap());

    let settings = Arc::new(lsys::Settings {
        width: 0.05,
        angle: PI / 8.0,
        iterations: 5,
        ..lsys::Settings::new()
    });

    fn generate_sample(
        grammar: &abnf::Grammar,
        distribution: &Distribution,
        settings: &lsys::Settings,
    ) -> Fitness {
        let seed = random_seed();
        let chromosome = generate_chromosome(&mut XorShiftRng::from_seed(seed), CHROMOSOME_LEN);
        let system = generate_system(
            grammar,
            &mut WeightedGenotype::new(
                chromosome,
                distribution,
                grammar.symbol_index("stack").unwrap(),
            ),
        );
        let (fit, _) = fitness::evaluate(&system, settings);
        fit
    }

    println!(
        "Generating {} samples...",
        num_samples * distributions.len()
    );
    let start_time = Instant::now();

    let samples = {
        let workers = num_cpus::get() + 1;

        let pool = CpuPool::new(workers);
        let mut tasks = Vec::with_capacity(num_samples);

        for (d, distribution) in distributions.iter().enumerate() {
            for _ in 0..num_samples {
                let distribution = distribution.clone();
                let grammar = grammar.clone();
                let settings = settings.clone();

                tasks.push(pool.spawn_fn(move || {
                    let sample = generate_sample(&grammar, &distribution, &settings);
                    future::ok::<(usize, Fitness), ()>((d, sample))
                }));
            }
        }

        future::join_all(tasks).wait().unwrap()
    };

    let end_time = Instant::now();
    let duration = end_time - start_time;
    println!(
        "Duration: {}.{}",
        duration.as_secs(),
        duration.subsec_nanos()
    );

    print!("Append to existing file? (Y/n): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.pop().unwrap(); // remove '\n'.

    if input != "n" && input != "N" {
        println!("Appending to file.");
    } else {
        println!("Overwriting file.");
        match fs::remove_file(csv_path) {
            Err(ref error) if error.kind() != io::ErrorKind::NotFound => {
                panic!("Failed removing file: {}", error);
            }
            Ok(_) | Err(_) => {}
        }
    }

    let mut csv = String::new();

    let mut csv_file = match OpenOptions::new().append(true).open(csv_path) {
        Ok(file) => file,
        Err(_) => {
            csv += "distribution,score,balance,branching,closeness,drop,nothing\n";
            File::create(csv_path).unwrap()
        }
    };

    let csv = samples.iter().fold(
        csv,
        |csv, &(d, ref f)| if f.is_nothing {
            csv + &format!("{},{},,,,,{}\n", d, f.score(), f.nothing_punishment())
        } else {
            csv +
                &format!(
                    "{},{},{},{},{},{},{}\n",
                    d,
                    f.score(),
                    f.balance,
                    f.branching,
                    f.closeness,
                    f.drop,
                    f.nothing_punishment()
                )
        },
    );

    csv_file.write_all(csv.as_bytes()).unwrap();
}

fn run_learning(matches: &ArgMatches) {
    fn generate_sample(grammar: &abnf::Grammar, distribution: &Distribution) -> ol::LSystem {
        let seed = random_seed();
        let chromosome = generate_chromosome(&mut XorShiftRng::from_seed(seed), CHROMOSOME_LEN);
        let mut genotype = WeightedGenotype::new(
            chromosome,
            distribution,
            grammar.symbol_index("stack").unwrap(),
        );
        generate_system(grammar, &mut genotype)
    }

    let (grammar, distribution) = get_sample_setup("grammar/lsys2.abnf");

    let grammar = Arc::new(grammar);
    let distribution = match matches.value_of("distribution") {
        Some(filename) => {
            println!("Using distribution from {}", filename);

            let file = match File::open(filename) {
                Ok(file) => file,
                Err(err) => {
                    println!("Failed opening distribution file \"{}\": {}", filename, err);
                    return;
                }
            };

            match bincode::deserialize_from(&mut BufReader::new(file), bincode::Infinite) {
                Ok(dist) => dist,
                Err(err) => {
                    println!(
                        "Failed deserializing distribution file \"{}\": {}",
                        filename,
                        err
                    );
                    return;
                }
            }
        }
        None => distribution,
    };

    let mut distribution = Arc::new(distribution);

    let bin_path = Path::new(matches.value_of("bin").unwrap());
    println!("Saving distribution to \"{}\".", bin_path.to_str().unwrap());

    let csv_path = Path::new(matches.value_of("csv").unwrap());
    println!("Saving distribution to \"{}\".", csv_path.to_str().unwrap());

    let stats_csv_path = matches.value_of("stats-csv").unwrap().to_string();
    println!("Saving learning stats to \"{}\".", stats_csv_path);

    let settings = Arc::new(lsys::Settings {
        width: 0.05,
        angle: PI / 8.0,
        iterations: 5,
        ..lsys::Settings::new()
    });

    let num_workers = matches.value_of("workers").map_or(
        num_cpus::get() + 1,
        |w| w.parse().unwrap(),
    );

    let dist_dump_path = Path::new("dist");
    match fs::create_dir_all(dist_dump_path) {
        Ok(_) => {}
        Err(err) => {
            println!(
                "Failed creating distribution dump directory \"{}\": {}",
                dist_dump_path.to_str().unwrap(),
                err
            );
            return;
        }
    }

    let mut dist_files = fs::read_dir(dist_dump_path).unwrap().peekable();
    if dist_files.peek().is_some() {
        println!(
            "There are files in \"{}\", remove these?",
            dist_dump_path.to_str().unwrap()
        );
        print!("[y/C]: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        input.pop().unwrap(); // remove '\n'.

        if input != "y" && input != "Y" {
            println!("Cancelled.");
            return;
        } else {
            println!("Removing files in \"{}\"", dist_dump_path.to_str().unwrap());
            for file in dist_files {
                fs::remove_file(file.unwrap().path()).unwrap();
            }
        }
    }

    let dimensions = distribution.dimensions();
    let error_threshold = matches
        .value_of("error-threshold")
        .unwrap()
        .parse()
        .unwrap();
    let min_samples = matches.value_of("min-samples").unwrap().parse().unwrap();
    let fitness_scale: f32 = matches.value_of("fitness-scale").unwrap().parse().unwrap();
    let cooldown: f32 = matches.value_of("cooldown-rate").unwrap().parse().unwrap();
    let max_moves = matches.value_of("max-moves").unwrap().parse::<usize>().unwrap() * dimensions;
    let dump_distributions = !matches.is_present("no-dump");

    println!("Distribution has {} dimensions.", dimensions);
    println!("Using error error threshold {}.", error_threshold);
    println!("Using minimum samples {}.", min_samples);
    println!("Using fitness scale {}.", fitness_scale);
    println!("Running with max {} total moves.", max_moves);

    let start_time = Instant::now();
    let pool = CpuPool::new(num_workers);
    let mut rng = rand::thread_rng();

    let measure_distribution = |distribution: Arc<Distribution>| -> (f32, usize) {
        let mut error = f32::MAX;
        let mut step_mean = 0.0;
        let mut step_scores = Vec::new();

        // Generate samples until error is with accepted threshold.
        while error > error_threshold {
            let tasks: Vec<_> = (0..min_samples)
                .map(|_| {
                    let grammar = grammar.clone();
                    let settings = settings.clone();
                    let distribution = distribution.clone();
                    pool.spawn_fn(move || -> FutureResult<f32, ()> {
                        let lsystem = generate_sample(&grammar, &distribution);

                        let (fit, _) = fitness::evaluate(&lsystem, &settings);
                        let score = fit.score();
                        future::ok(score)
                    })
                })
                .collect();

            let result = match future::join_all(tasks).wait() {
                Ok(result) => result,
                Err(()) => {
                    panic!("Failed joining tasks: Unknown reason.");
                }
            };
            let batch_scores: Vec<_> = result;

            step_scores.extend(batch_scores);

            let score_sum: f32 = step_scores.iter().sum();
            let size = step_scores.len();
            let mean = score_sum / size as f32;
            let unbiased_sample_variance =
                step_scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / (size - 1) as f32;
            let sample_standard_deviation = unbiased_sample_variance.sqrt();

            error = sample_standard_deviation / size as f32;
            step_mean = mean;
        }

        (step_mean, step_scores.len())
    };

    let mut scores = Vec::new();

    println!("Measuring initial distribution");

    let (mut current_score, mut num_samples) = measure_distribution(distribution.clone());
    scores.push((num_samples, num_samples, current_score));

    println!("Generated {} samples.", num_samples);
    println!("Initial distribution has score {}.", current_score);

    let stats_csv_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&stats_csv_path)
        .expect("Could not create stats file");

    let stats_writer = Arc::new(Mutex::new(
        BufWriter::with_capacity(1024 * 1024, stats_csv_file),
    ));

    let mut save_future = {
        let distribution = distribution.clone();
        let stats_writer = stats_writer.clone();

        pool.spawn_fn(move || -> FutureResult<(), ()> {
            if dump_distributions {
                let first_dist_filename = format!("{}.csv", 0);
                let first_dist_file_path = dist_dump_path.join(first_dist_filename);
                let mut csv_file = File::create(first_dist_file_path).unwrap();
                csv_file
                    .write_all(distribution.to_csv().as_bytes())
                    .unwrap();
            }

            let stats_csv = "iteration,samples,measure samples,score,accepted,temperature,type\n"
                .to_string() +
                &format!("{},{},{},{},,,{}\n", 0, num_samples, num_samples, current_score, "init");
            stats_writer
                .lock()
                .unwrap()
                .write_all(stats_csv.as_bytes())
                .expect("Could not write to stats file");

            future::ok(())
        })
    };

    let mut iteration = 0_usize;
    let mut temperature = 1.0_f32;

    while iteration < max_moves {
        println!("Iteration {}.", iteration);
        println!("Temperature: {}.", temperature);

        if temperature < 0.01 {
            println!("Performing re-annealing");
            temperature = 1.0;
        }

        let mut new_distribution = (*distribution).clone();
        mutate_distribution(&mut new_distribution, &mut rng);

        let new_distribution = Arc::new(new_distribution);
        let old_score = current_score;
        let (new_score, new_num_samples) = measure_distribution(new_distribution.clone());
        num_samples += new_num_samples;

        println!(
            "Generated {} of total {} samples.",
            new_num_samples,
            num_samples
        );
        println!(
            "Neighbour score was {} with a difference of {}.",
            new_score,
            new_score - old_score
        );

        let score_diff = (new_score - old_score) * fitness_scale;
        let accepted = if score_diff > 0.0 {
            true
        } else {
            let probability = if temperature > 0.0 {
                E.powf(score_diff / ((1.0 - old_score) * temperature))
            } else {
                0.0
            };

            println!("Probability of being selected is {}.", probability);

            let random = Range::new(0.0, 1.0).ind_sample(&mut rng);

            if random < probability {
                true
            } else {
                false
            }
        };

        if accepted {
            println!("Neighbour was selected.");
            distribution = new_distribution.clone();
            current_score = new_score;
        }

        save_future.wait().unwrap();
        save_future = {
            let distribution = new_distribution.clone();
            let stats_writer = stats_writer.clone();

            pool.spawn_fn(move || -> FutureResult<(), ()> {
                if dump_distributions {
                    let filename = format!("{}.csv", iteration);
                    let file_path = dist_dump_path.join(filename);
                    let mut csv_file = File::create(file_path).unwrap();
                    csv_file
                        .write_all(distribution.to_csv().as_bytes())
                        .unwrap();
                }

                let iteration_type = if !accepted {
                    "stay"
                } else if score_diff < 0.0 {
                    "explore"
                } else {
                    "improve"
                };

                let stats_csv = &format!(
                    "{},{},{},{},{},{},{}\n",
                    iteration + 1,
                    num_samples,
                    new_num_samples,
                    new_score,
                    accepted,
                    temperature,
                    iteration_type
                );
                stats_writer
                    .lock()
                    .unwrap()
                    .write_all(stats_csv.as_bytes())
                    .expect("Could not write to stats file");

                future::ok(())
            })
        };

        iteration += 1;
        temperature /= 1.0 + cooldown * temperature;
    }

    println!("Finished search.");

    save_future.wait().unwrap();

    let end_time = Instant::now();
    let duration = end_time - start_time;
    let seconds = duration.as_secs();
    let hours = seconds / (60 * 60);
    let minutes = seconds % (60 * 60) / 60;
    println!(
        "Duration: {}:{}:{}.{}",
        hours,
        minutes,
        seconds,
        duration.subsec_nanos()
    );

    stats_writer.lock().unwrap().flush().expect(
        "Could not save stats",
    );

    match Arc::try_unwrap(distribution) {
        Ok(distribution) => {
            let dist_file = File::create(bin_path).unwrap();
            bincode::serialize_into(
                &mut BufWriter::new(dist_file),
                &distribution,
                bincode::Infinite,
            ).expect("Failed writing distribution bin file");

            let mut dist_file = File::create(csv_path).unwrap();
            dist_file
                .write_all(distribution.to_csv().as_bytes())
                .expect("Failed writing distribution csv file");
        }
        Err(_) => {
            println!("Error: Could not save distribution: Failed unwrapping distribution Arc.");
        }
    }
}

fn run_distribution_csv_to_bin(matches: &ArgMatches) {
    let input_path = Path::new(matches.value_of("INPUT").unwrap());
    let output_path = match matches.value_of("output") {
        Some(path) => PathBuf::from(path),
        None => input_path.with_extension("bin"),
    };
    let version = matches.value_of("version");

    let mut input_file = File::open(input_path).expect("Could not open input file");
    let mut csv = String::new();
    input_file.read_to_string(&mut csv).expect(
        "Could not read input file",
    );

    let grammar_path = matches.value_of("grammar").unwrap();
    let grammar = abnf::parse_file(grammar_path).expect("Could not parse grammar file");

    let mut distribution = if let Some(version) = version {
        let version: u32 = version.parse().expect(
            "'version' argument must be an integer",
        );
        if version == 1 {
            println!("Loading distribution using v1 structure");
            Distribution::from_csv_v1(&csv, &grammar)
        } else {
            println!("Unknown version {}, assuming latest version", version);
            Distribution::from_csv(&csv)
        }
    } else {
        Distribution::from_csv(&csv)
    };

    let string_index = grammar.symbol_index("string").expect(
        "Grammar does not contain 'string' symbol",
    );
    distribution.set_default_weights(string_index, 1, &[1.0, 0.0]);
    // Remove the default weights from the depth weights.
    distribution.depths[DEPTHS - 1][string_index].pop();

    let output_file = File::create(&output_path).unwrap();
    bincode::serialize_into(
        &mut BufWriter::new(output_file),
        &distribution,
        bincode::Infinite,
    ).expect("Could not write output file");

    println!("Wrote \"{}\"", output_path.to_str().unwrap());
}

fn dividers_from_weights(weights: &[f32]) -> Vec<f32> {
    (0..weights.len() - 1)
        .map(|i| {
            let width: f32 = weights.iter().take(i + 1).sum();
            let parent_width: f32 = weights.iter().take(i + 2).sum();
            if parent_width == 0.0 {
                0.0
            } else {
                width / parent_width
            }
        })
        .collect()
}

#[allow(dead_code)]
fn weights_from_dividers(dividers: &[f32]) -> Vec<f32> {
    if dividers.is_empty() {
        return vec![1.0];
    }

    let mut weights = Vec::with_capacity(dividers.len() + 1);
    let mut remaining = 1.0;

    for divider in dividers.iter().rev() {
        let parent = remaining;
        remaining = divider * parent;
        weights.push(parent - remaining);
    }

    weights.push(remaining);

    weights.into_iter().rev().collect()
}

fn run_sample_weight_space(matches: &ArgMatches) {
    let dimensions: usize = match matches.value_of("DIMENSIONS").unwrap().parse() {
        Ok(d) => d,
        Err(err) => {
            println!("Invalid argument to DIMENSIONS: {}", err);
            return;
        }
    };

    let num_samples: usize = match matches.value_of("num-samples").unwrap().parse() {
        Ok(d) => d,
        Err(err) => {
            println!("Invalid argument to num-samples: {}", err);
            return;
        }
    };

    println!(
        "Sampling {} samples in {}-dimensinal weight space.",
        num_samples,
        dimensions
    );

    let output_path = Path::new(matches.value_of("output").unwrap());
    println!("Saving samples to \"{}\".", output_path.to_str().unwrap());

    let dividers = matches.is_present("dividers");
    if dividers {
        println!("Sampling dividers instead.");
    }

    let workers = num_cpus::get() + 1;
    let pool = CpuPool::new(workers);

    fn generate_weight(d: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut remaining = 1.0;
        let mut components: Vec<f32> = (0..d - 1)
            .map(|_| {
                let w = Range::new(0.0, remaining).ind_sample(&mut rng);
                remaining -= w;
                w
            })
            .collect();
        components.push(remaining);
        rng.shuffle(&mut components);
        components
    };

    let tasks: Vec<_> = (0..num_samples)
        .map(|_| if dividers {
            pool.spawn_fn(move || -> FutureResult<Vec<f32>, ()> {
                future::ok(dividers_from_weights(&generate_weight(dimensions)))
            })
        } else {
            pool.spawn_fn(move || -> FutureResult<Vec<f32>, ()> {
                future::ok(generate_weight(dimensions))
            })
        })
        .collect();

    let samples = match future::join_all(tasks).wait() {
        Ok(result) => result,
        Err(()) => {
            panic!("Failed joining tasks: Unknown reason.");
        }
    };

    let mut csv = String::new();
    for (i, sample) in samples.iter().enumerate() {
        csv += &format!("{}", i);
        for component in sample {
            csv += &format!(",{}", component);
        }
        csv += "\n";
    }

    let mut csv_file = File::create(output_path).unwrap();
    csv_file.write_all(csv.as_bytes()).expect(
        "Failed writing sample csv file",
    );
}

fn run_ge(matches: &ArgMatches) {
    use rsgenetic::pheno::{Fitness, Phenotype};
    use rsgenetic::sim::seq::Simulator;
    use rsgenetic::sim::{Simulation, Builder};
    use rsgenetic::sim::select::MaximizeSelector;

    const POPULATION_SIZE: usize = 128;
    const SELECTION_SIZE: usize = 16;
    const MUTATION_PROB: f32 = 0.2;
    const MAX_ITERATIONS: u64 = 200;

    let (grammar, distribution) = get_sample_setup(matches.value_of("grammar").unwrap());
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

    println!("{}", distribution);

    let settings = lsys::Settings {
        width: 0.05,
        angle: PI / 8.0,
        iterations: 5,
        ..lsys::Settings::default()
    };

    println!("Population size: {}", POPULATION_SIZE);
    println!("Selection size: {}", SELECTION_SIZE);
    println!("Mutation factor: {}", MUTATION_PROB);
    println!("Max iterations: {}", MAX_ITERATIONS);

    let mut population = (0..POPULATION_SIZE)
        .map(|_| {
            let seed = random_seed();
            let chromosome = generate_chromosome(&mut XorShiftRng::from_seed(seed), CHROMOSOME_LEN);
            LsysPhenotype {
                grammar: &grammar,
                settings: &settings,
                genotype: RefCell::new(WeightedGenotype::new(
                    chromosome,
                    &distribution,
                    grammar.symbol_index("stack").unwrap(),
                )),
                fitness: Cell::new(None),
            }
        })
        .collect();

    let mut simulator = Simulator::builder(&mut population)
        .set_selector(Box::new(MaximizeSelector::new(SELECTION_SIZE)))
        .set_max_iters(MAX_ITERATIONS)
        .build();

    simulator.run();
    let best = simulator.get().unwrap();

    let lsystem = generate_system(&grammar, &mut *best.genotype.borrow_mut());
    println!("{}", lsystem);
    println!("Fitness: {}", best.fitness().0);

    let model_dir = Path::new("model");
    fs::create_dir_all(model_dir).unwrap();

    let filename = format!("{}.yaml", Local::now().to_rfc3339());
    let path = model_dir.join(filename);

    let file = File::create(&path).unwrap();
    serde_yaml::to_writer(&mut BufWriter::new(file), &lsystem).unwrap();

    println!("Saved to {}", path.to_str().unwrap());

    #[derive(PartialEq, PartialOrd, Clone, Copy)]
    struct LsysFitness(f32);

    impl Eq for LsysFitness {}

    impl Ord for LsysFitness {
        fn cmp(&self, other: &LsysFitness) -> Ordering {
            self.0.partial_cmp(&other.0).expect(
                "Fitness is NaN and can't be ordered",
            )
        }
    }

    impl Fitness for LsysFitness {
        fn zero() -> LsysFitness {
            LsysFitness(0.0)
        }

        fn abs_diff(&self, other: &LsysFitness) -> LsysFitness {
            LsysFitness(self.0 - other.0)
        }
    }

    #[derive(Clone)]
    struct LsysPhenotype<'a, G> {
        grammar: &'a Grammar,
        settings: &'a lsys::Settings,
        genotype: RefCell<WeightedGenotype<'a, G>>,
        fitness: Cell<Option<LsysFitness>>,
    }

    impl<'a, G: Gene + Clone> Phenotype<LsysFitness> for LsysPhenotype<'a, G> {
        fn fitness(&self) -> LsysFitness {
            if let Some(ref fitness) = self.fitness.get() {
                *fitness
            } else {
                let lsystem = generate_system(self.grammar, &mut *self.genotype.borrow_mut());
                let fitness = fitness::evaluate(&lsystem, self.settings);
                let fitness = LsysFitness(fitness.0.score());

                self.fitness.set(Some(fitness));

                fitness
            }
        }

        fn crossover(&self, other: &Self) -> Self {
            let genotype_self = self.genotype.borrow();
            let crossover_point = Range::new(0, genotype_self.genotype.chromosome.len())
                .ind_sample(&mut rand::thread_rng());
            let iter_self = genotype_self.genotype.chromosome.iter().take(
                crossover_point,
            );

            let genotype_other = other.genotype.borrow();
            let iter_other = genotype_other.genotype.chromosome.iter().skip(
                crossover_point,
            );
            let chromosome = iter_self.chain(iter_other).cloned().collect();

            LsysPhenotype {
                grammar: self.grammar,
                settings: self.settings,
                genotype: RefCell::new(WeightedGenotype::new(
                    chromosome,
                    genotype_self.distribution,
                    genotype_self.stack_rule_index,
                )),
                fitness: Cell::new(None),
            }
        }

        fn mutate(&self) -> Self {
            let mut rng = rand::thread_rng();

            if Range::new(0.0, 1.0).ind_sample(&mut rng) > MUTATION_PROB {
                self.clone()
            } else {
                let genotype = self.genotype.borrow();
                let mut chromosome = genotype.genotype.chromosome.clone();
                let mutation_index =
                    Range::new(0, chromosome.len()).ind_sample(&mut rand::thread_rng());
                chromosome[mutation_index] = Range::new(G::min_value(), G::max_value())
                    .ind_sample(&mut rng);

                LsysPhenotype {
                    grammar: self.grammar,
                    settings: self.settings,
                    genotype: RefCell::new(WeightedGenotype::new(
                        chromosome,
                        genotype.distribution,
                        genotype.stack_rule_index,
                    )),
                    fitness: Cell::new(None),
                }
            }
        }
    }
}

fn run_benchmark(_: &ArgMatches) {
    use cpuprofiler::PROFILER;

    fn generate_sample(grammar: &abnf::Grammar, distribution: &Distribution) -> ol::LSystem {
        let seed = random_seed();
        let chromosome = generate_chromosome(&mut XorShiftRng::from_seed(seed), CHROMOSOME_LEN);
        let mut genotype = WeightedGenotype::new(
            chromosome,
            distribution,
            grammar.symbol_index("stack").unwrap(),
        );
        generate_system(grammar, &mut genotype)
    }

    let (grammar, distribution) = get_sample_setup("grammar/lsys2.abnf");
    let settings = Arc::new(lsys::Settings {
        width: 0.05,
        angle: PI / 8.0,
        iterations: 5,
        ..lsys::Settings::new()
    });

    PROFILER.lock().unwrap().start("./bench.profile").unwrap();

    for _ in 0..200 {
        let lsystem = generate_sample(&grammar, &distribution);
        fitness::evaluate(&lsystem, &settings);
    }

    PROFILER.lock().unwrap().stop().unwrap();
}

#[cfg(test)]
mod test {
    use super::*;
    use abnf::{Item, Content, Grammar, List, Repeat, Symbol};

    #[test]
    fn test_infer_selections_value() {
        let item = Item::new(Content::Value("value".to_string()));

        let grammar = Grammar::from_rules(vec![
            (
                Symbol::from("symbol"),
                List::Sequence(vec![item.clone()])
            ),
        ]);

        assert_eq!(
            infer_item_selections(&item, 0, "value", &grammar),
            Ok((Vec::new(), 5))
        );
    }

    #[test]
    fn test_infer_selections_repeat_limits() {
        let item = Item::repeated(
            Content::Value("value".to_string()),
            Repeat::with_limits(2, 4),
        );

        let grammar = Grammar::from_rules(vec![
            (
                Symbol::from("symbol"),
                List::Sequence(vec![item.clone()])
            ),
        ]);

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

        let grammar = Grammar::from_rules(vec![(Symbol::from("symbol"), list.clone())]);

        assert_eq!(
            infer_list_selections(&list, 0, "2", &grammar),
            Ok((vec![1], 1))
        );
    }

    #[test]
    fn test_infer_selections_match() {
        let item = Item::new(Content::Value("value".to_string()));

        let grammar = Grammar::from_rules(vec![
            (
                Symbol::from("symbol"),
                List::Sequence(vec![item.clone()])
            ),
        ]);

        assert_eq!(
            infer_selections("value", &grammar, "symbol"),
            Ok(Vec::new())
        );
    }

    #[test]
    fn test_infer_selections_match_alternatives() {
        let list = List::Alternatives(vec![
            Item::new(Content::Value("1".to_string())),
            Item::new(Content::Value("2".to_string())),
            Item::new(Content::Value("3".to_string())),
        ]);

        let grammar = Grammar::from_rules(vec![(Symbol::from("symbol"), list.clone())]);

        assert_eq!(infer_selections("2", &grammar, "symbol"), Ok(vec![1]));
    }

    #[test]
    fn test_infer_selections_mismatch() {
        let item = Item::new(Content::Value("value".to_string()));

        let grammar = Grammar::from_rules(vec![
            (
                Symbol::from("symbol"),
                List::Sequence(vec![item.clone()])
            ),
        ]);

        assert_eq!(
            infer_selections("notvalue", &grammar, "symbol"),
            Err("Expanded string does not match grammar".to_string())
        );
    }

    #[test]
    fn test_infer_selections_missing() {
        let item = Item::new(Content::Value("value".to_string()));

        let grammar = Grammar::from_rules(vec![
            (
                Symbol::from("symbol"),
                List::Sequence(vec![item.clone()])
            ),
        ]);

        assert_eq!(
            infer_selections("valueextra", &grammar, "symbol"),
            Err(
                "Expanded string does not fully match grammar. \
                 The first 5 characters matched"
                    .to_string(),
            )
        );
    }

    #[test]
    fn test_lsystem_ge_expansion() {
        let grammar = abnf::parse_file("grammar/lsys.abnf").expect("Could not parse ABNF file");
        let chromosome = vec![
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
        let mut genotype = Genotype::new(chromosome);

        assert_eq!(expand_grammar(&grammar, "axiom", &mut genotype), "F[FX]X");
    }

    #[test]
    fn test_lsystem_ge_inference() {
        let grammar = abnf::parse_file("grammar/lsys.abnf").expect("Could not parse ABNF file");
        let chromosome = vec![
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
            Ok(chromosome)
        );
    }

    #[test]
    fn test_dividers_from_weights() {
        assert_slice_approx_eq!(
            &dividers_from_weights(&[1.0]),
            &Vec::<f32>::new(),
            f32::EPSILON
        );
        assert_slice_approx_eq!(&dividers_from_weights(&[0.5, 0.5]), &[0.5], f32::EPSILON);
        assert_slice_approx_eq!(&dividers_from_weights(&[0.75, 0.25]), &[0.75], f32::EPSILON);
        assert_slice_approx_eq!(
            &dividers_from_weights(&[0.5, 0.25, 0.25]),
            &[2.0 / 3.0, 0.75],
            f32::EPSILON
        );
        assert_slice_approx_eq!(
            &dividers_from_weights(&[0.25, 0.25, 0.25, 0.25]),
            &[0.5, 2.0 / 3.0, 0.75],
            f32::EPSILON
        );
        assert_slice_approx_eq!(&dividers_from_weights(&[0.0, 1.0]), &[0.0], f32::EPSILON);
        assert_slice_approx_eq!(&dividers_from_weights(&[1.0, 0.0]), &[1.0], f32::EPSILON);
        assert_slice_approx_eq!(&dividers_from_weights(&[0.0, 0.0]), &[0.0], f32::EPSILON);
        assert_slice_approx_eq!(&dividers_from_weights(&[1.0, 1.0]), &[0.5], f32::EPSILON);
    }

    #[test]
    fn test_dividers_from_weights_zero() {
        assert_slice_approx_eq!(
            &dividers_from_weights(&[0.0, 1.0, 0.0]),
            &[0.0, 1.0],
            f32::EPSILON
        );
        assert_slice_approx_eq!(
            &dividers_from_weights(&[0.0, 0.5, 0.5]),
            &[0.0, 0.5],
            f32::EPSILON
        );
        assert_slice_approx_eq!(
            &dividers_from_weights(&[0.5, 0.5, 0.0]),
            &[0.5, 1.0],
            f32::EPSILON
        );
        assert_slice_approx_eq!(
            &dividers_from_weights(&[0.0, 0.0, 1.0]),
            &[0.0, 0.0],
            f32::EPSILON
        );
        assert_slice_approx_eq!(
            &dividers_from_weights(&[1.0, 0.0, 0.0]),
            &[1.0, 1.0],
            f32::EPSILON
        );
        assert_slice_approx_eq!(
            &dividers_from_weights(&[0.5, 0.0, 0.5]),
            &[1.0, 0.5],
            f32::EPSILON
        );
    }

    #[test]
    fn test_weights_from_dividers_none() {
        assert_slice_approx_eq!(&weights_from_dividers(&[]), &[1.0], f32::EPSILON);
    }

    #[test]
    fn test_weights_from_dividers_middle() {
        assert_slice_approx_eq!(&weights_from_dividers(&[0.5]), &[0.5, 0.5], f32::EPSILON);
    }

    #[test]
    fn test_weights_from_dividers_quarter() {
        assert_slice_approx_eq!(&weights_from_dividers(&[0.75]), &[0.75, 0.25], f32::EPSILON);
    }

    #[test]
    fn test_weights_from_dividers_multiple() {
        assert_slice_approx_eq!(
            &weights_from_dividers(&[2.0 / 3.0, 0.75]),
            &[0.5, 0.25, 0.25],
            f32::EPSILON
        );
        assert_slice_approx_eq!(
            &weights_from_dividers(&[0.5, 2.0 / 3.0, 0.75]),
            &[0.25, 0.25, 0.25, 0.25],
            f32::EPSILON
        );
    }

    #[test]
    fn test_weights_from_dividers_full_single() {
        assert_slice_approx_eq!(&weights_from_dividers(&[0.0]), &[0.0, 1.0], f32::EPSILON);
        assert_slice_approx_eq!(&weights_from_dividers(&[1.0]), &[1.0, 0.0], f32::EPSILON);
    }

    #[test]
    fn test_weights_from_dividers_zero_sides() {
        assert_slice_approx_eq!(
            &weights_from_dividers(&[0.0, 1.0]),
            &[0.0, 1.0, 0.0],
            f32::EPSILON * 2.0
        );
    }

    #[test]
    fn test_weights_from_dividers_zero_right() {
        assert_slice_approx_eq!(
            &weights_from_dividers(&[1.0, 1.0]),
            &[1.0, 0.0, 0.0],
            f32::EPSILON * 2.0
        );
    }

    #[test]
    fn test_weights_from_dividers_zero_left() {
        assert_slice_approx_eq!(
            &weights_from_dividers(&[0.0, 0.0]),
            &[0.0, 0.0, 1.0],
            f32::EPSILON
        );
    }

    #[test]
    fn test_weights_from_dividers_zero_between() {
        assert_slice_approx_eq!(
            &weights_from_dividers(&[0.0, 0.5]),
            &[0.0, 0.5, 0.5],
            f32::EPSILON
        );
        assert_slice_approx_eq!(
            &weights_from_dividers(&[0.5, 1.0]),
            &[0.5, 0.5, 0.0],
            f32::EPSILON
        );
        assert_slice_approx_eq!(
            &weights_from_dividers(&[1.0, 0.5]),
            &[0.5, 0.0, 0.5],
            f32::EPSILON
        );
    }

    #[test]
    fn test_weight_divider_consistency() {
        let weight_set = [
            vec![1.0],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![0.9, 0.1],
            vec![0.9999, 0.0001],
            vec![0.0001, 0.9999],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.5, 0.5],
            vec![0.5, 0.0, 0.5],
            vec![0.5, 0.5, 0.0],
            vec![0.8, 0.0, 0.2],
            vec![0.2, 0.0, 0.8],
            vec![0.5, 0.0, 0.0, 0.5],
        ];

        for weights in weight_set.iter() {
            assert_slice_approx_eq!(
                &weights,
                &weights_from_dividers(&dividers_from_weights(&weights)),
                f32::EPSILON
            );
        }
    }

    #[test]
    #[ignore]
    fn test_weight_divider_consistency_rand() {
        const NUM: usize = 10000;
        const LEN: usize = 20;
        let mut rng = rand::thread_rng();

        for length in 2..LEN {
            for _ in 0..NUM {
                let weights: Vec<f32> = (0..length)
                    .map(|_| Range::new(0.0, 1.0).ind_sample(&mut rng))
                    .collect();
                let sum: f32 = weights.iter().sum();
                if sum > 0.0 {
                    let weights: Vec<f32> = weights.iter().map(|w| w / sum).collect();
                    assert_slice_approx_eq!(
                        &weights,
                        &weights_from_dividers(&dividers_from_weights(&weights)),
                        0.000001
                    );
                }
            }
        }
    }

    #[ignore]
    #[test]
    fn test_weight_divider_robustness() {
        let divider_set = [
            vec![0.5],
            vec![1.0],
            vec![0.0],
            vec![0.5, 0.5],
            vec![2.0 / 3.0, 0.75],
            vec![0.5, 0.0],
            vec![1.0, 0.5],
        ];

        for dividers in divider_set.iter() {
            assert_slice_approx_eq!(
                &dividers,
                &dividers_from_weights(&weights_from_dividers(&dividers)),
                f32::EPSILON
            );
        }
    }

    #[test]
    #[ignore]
    fn test_weight_divider_robustness_rand() {
        const NUM: usize = 10000;
        const LEN: usize = 20;
        let mut rng = rand::thread_rng();

        for length in 2..LEN {
            for _ in 0..NUM {
                let dividers: Vec<f32> = (0..length)
                    .map(|_| Range::new(0.0, 1.0).ind_sample(&mut rng))
                    .collect();
                assert_slice_approx_eq!(
                    &dividers,
                    &dividers_from_weights(&weights_from_dividers(&dividers)),
                    0.000001
                );
            }
        }
    }

    #[ignore]
    #[test]
    fn test_weight_divider_effect() {
        let weight_set = [
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![0.9999, 0.0001],
            vec![0.0001, 0.9999],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.5, 0.5],
            vec![0.5, 0.0, 0.5],
            vec![0.5, 0.5, 0.0],
            vec![0.5, 0.0, 0.0, 0.5],
        ];

        for weights in weight_set.iter() {
            let dividers = dividers_from_weights(&weights);
            for i in 0..dividers.len() {
                for sign in [1.0_f32, -1.0].iter() {
                    let mut dividers = dividers.clone();
                    let changed = dividers[i] + sign * 0.1;
                    if changed != dividers[i] {
                        dividers[i] = changed;
                        assert_slice_approx_ne!(
                            &weights,
                            &weights_from_dividers(&dividers),
                            f32::EPSILON
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn test_mutate_distribution_valid() {
        let mut distribution = Distribution::new();
        let mut rng = rand::thread_rng();
        let num_runs = 100000;

        distribution.set_weights(0, 0, 0, &[0.5, 0.5, 0.5, 0.5, 0.5]);

        for _ in 0..num_runs {
            mutate_distribution(&mut distribution, &mut rng);

            for rules in &distribution.depths {
                for choices in rules {
                    for options in choices {
                        for weight in options {
                            assert!(!(weight.is_nan() || weight.is_infinite()));
                        }
                    }
                }
            }
        }
    }
}
