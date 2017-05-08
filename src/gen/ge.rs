use std::f32::consts::{PI, E};
use std::f32;
use std::{cmp, fmt, iter};
use std::collections::HashMap;
use std::io::{self, BufWriter, BufReader, Write, Read};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::ops::Add;

use rand::{self, Rng, XorShiftRng, SeedableRng};
use rand::distributions::{IndependentSample, Range};
use na::{self, UnitQuaternion, Point3};
use kiss3d::camera::ArcBall;
use kiss3d::scene::SceneNode;
use num::{self, Unsigned, NumCast};
use glfw::{Key, WindowEvent, Action};
use serde_yaml;
use time;
use num_cpus;
use futures::Future;
use futures::future::{self, FutureResult};
use futures_cpupool::CpuPool;
use bincode;
use crossbeam;
use clap::{App, SubCommand, Arg, ArgMatches};
use csv;

use abnf;
use abnf::expand::{SelectionStrategy, expand_grammar};
use lsys::{self, ol};
use lsys3d;
use lsystems;
use yobun::read_dir_all;
use setup_window;
use gen::fitness::{self, Fitness};

const DEPTHS: usize = 4;

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
            .arg(Arg::with_name("learning-rate")
                .short("l")
                .long("learning-rate")
                .takes_value(true)
                .default_value("1.3")
                .help("Learning rate of algorithm. Should be above 1.")
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
    } else if let Some(matches) = matches.subcommand_matches("stats") {
        run_stats(matches);
    } else if let Some(matches) = matches.subcommand_matches("learning") {
        run_learning(matches);
    } else if let Some(matches) = matches.subcommand_matches("dist-csv-to-bin") {
        run_distribution_csv_to_bin(matches);
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
                        let (fit, _) = fitness::evaluate(&system, settings);
                        Sample {
                            seed: seed,
                            score: fit.score(),
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

fn get_sample_setup() -> (abnf::Ruleset, Distribution) {
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
    fn generate_sample(grammar: &abnf::Ruleset,
                       distribution: &Distribution)
                       -> ([u32; 4], ol::LSystem) {
        let seed = random_seed();
        let genes = generate_genome(&mut XorShiftRng::from_seed(seed), GENOME_LENGTH);
        let system = generate_system(grammar, &mut WeightedGenotype::new(genes, distribution));
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

    let settings = Arc::new(lsys::Settings {
                                width: 0.05,
                                angle: PI / 8.0,
                                iterations: 5,
                                ..lsys::Settings::new()
                            });

    let batch_size = usize::from_str_radix(matches.value_of("batch-size").unwrap(), 10).unwrap();
    println!("Using batch size {}.", batch_size);

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

                while work.load(Ordering::Relaxed) {
                    for _ in 0..SEQUENCE_SIZE {
                        let (seed, lsystem) = generate_sample(&grammar, &distribution);
                        if accept_sample(&lsystem, &settings) {
                            accepted_samples.push(seed);
                        }
                    }

                    batch_num_samples += SEQUENCE_SIZE;

                    if accepted_samples.len() >= batch_size {
                        dump_samples(&accepted_samples, batch_num_samples, batch);

                        num_samples.fetch_add(batch_num_samples, Ordering::Relaxed);

                        accepted_samples.clear();
                        batch += 1;
                        batch_num_samples = 0;
                    }
                }

                dump_samples(&accepted_samples, batch_num_samples, batch);
                num_samples.fetch_add(batch_num_samples, Ordering::Relaxed);
                num_good_samples.fetch_add(batch * batch_size + accepted_samples.len(),
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
    let samples_path = Path::new(matches.value_of("SAMPLES").unwrap());
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

    // Remove the default weights from the depth weights.
    distribution.depths[DEPTHS - 1].get_mut("string").unwrap().pop();

    let dist_file = File::create(bin_path).unwrap();
    bincode::serialize_into(&mut BufWriter::new(dist_file),
                            &distribution,
                            bincode::Infinite)
            .unwrap();
}

fn adjust_distribution(distribution: &mut Distribution, stats: &SelectionStats, factor: f32) {
    for (depth, rules) in stats.data.iter().enumerate() {
        for (rule, choices) in rules.iter() {
            for (choice, options) in choices.iter().enumerate() {
                let weights = distribution.get_weights_mut(depth, rule, choice as u32);
                if let Some(weights) = weights {
                    assert_eq!(weights.len(),
                               options.len(),
                               "Stats has different number of weights than distribution");
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

fn mutate_distribution<R>(distribution: &mut Distribution, factor: f32, rng: &mut R)
    where R: Rng
{
    for rules in &mut distribution.depths {
        for choices in rules.values_mut() {
            for options in choices {
                let num_weights = options.len() as f32;
                for weight in options {
                    let change = Range::new(-factor, factor).ind_sample(rng) / num_weights;
                    *weight = na::clamp(*weight + change, 0.0, 1.0);
                }
            }
        }
    }
}

type SelectionChoice = Vec<usize>;
type SelectionRule = Vec<SelectionChoice>;
type SelectionDepth = HashMap<String, SelectionRule>;

#[derive(Clone)]
struct SelectionStats {
    data: Vec<SelectionDepth>,
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

impl iter::Sum for SelectionStats {
    fn sum<I>(mut iter: I) -> Self
        where I: Iterator<Item=Self>
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
        where I: Iterator<Item=&'a Self>
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

        if self.weighted_genotype.distribution.has_weights(depth, rule, choice) {
            self.stats.make_room(depth, rule, choice, num);
            self.stats.add_selection(selection, depth, rule, choice);
        }

        selection
    }

    fn select_repetition(&mut self, min: u32, max: u32, rulechain: &[&str], choice: u32) -> u32 {
        let selection = self.weighted_genotype
            .select_repetition(min, max, rulechain, choice);

        let depth = WeightedGenotype::<'a, G>::find_depth(rulechain);
        let rule = rulechain.last().unwrap();
        let num = (max - min + 1) as usize;

        if self.weighted_genotype.distribution.has_weights(depth, rule, choice) {
            self.stats.make_room(depth, rule, choice, num);
            self.stats.add_selection((selection - min) as usize, depth, rule, choice);
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

    fn has_weights(&self, depth: usize, rule: &str, choice_num: u32) -> bool {
        if depth < self.depths.len() {
            let rules = &self.depths[depth];
            if let Some(choices) = rules.get(rule) {
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

    fn get_weights_mut(&mut self, depth: usize, rule: &str, choice_num: u32) -> Option<&mut [f32]> {
        if depth < self.depths.len() {
            let rules = &mut self.depths[depth];
            if let Some(choices) = rules.get_mut(rule) {
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

    fn set_weight(&mut self,
                  depth: usize,
                  rule: &str,
                  choice: u32,
                  alternative: usize,
                  weight: f32)
    {
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

        let alternatives = &mut choices[choice];
        while alternatives.len() < alternative + 1 {
            alternatives.push(f32::NAN);
        }

        alternatives[alternative] = weight;
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

    fn normalize(&mut self) {
        for rules in &mut self.depths {
            for choices in rules.values_mut() {
                for alternatives in choices {
                    let total: f32 = alternatives.iter().sum();
                    for weight in alternatives {
                        *weight /= total as f32;
                    }
                }
            }
        }
    }

    fn into_normalized(mut self) -> Distribution {
        self.normalize();
        self
    }

    fn to_csv(&self) -> String {
        let mut csv = "depth,rule,choice,alternative,weight\n".to_string();
        for (depth, rules) in self.depths.iter().enumerate() {
            for (rule, choices) in rules {
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
            let (depth, rule, choice, alternative, weight): (usize, String, u32, usize, f32)
                = row.unwrap();
            dist.set_weight(depth, &rule, choice, alternative, weight);
        }

        dist
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

fn run_stats(matches: &ArgMatches) {
    let grammar =
        Arc::new(abnf::parse_file("grammar/lsys2.abnf").expect("Could not parse ABNF file"));

    let distributions: Vec<Arc<Distribution>> = {
        let paths = matches
            .values_of("DISTRIBUTIONS")
            .expect("No distributions specified");

        println!("Distributions: {:?}", paths.clone().collect::<Vec<_>>());

        paths
            .map(|p| {
                     let file = File::open(p).unwrap();
                     let distribution = bincode::deserialize_from(&mut BufReader::new(file),
                                                                  bincode::Infinite)
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

    fn generate_sample(grammar: &abnf::Ruleset,
                       distribution: &Distribution,
                       settings: &lsys::Settings)
                       -> Fitness {
        let seed = random_seed();
        let genes = generate_genome(&mut XorShiftRng::from_seed(seed), GENOME_LENGTH);
        let system = generate_system(grammar, &mut WeightedGenotype::new(genes, distribution));
        let (fit, _) = fitness::evaluate(&system, settings);
        fit
    }

    println!("Generating {} samples...",
             num_samples * distributions.len());
    let start_time = time::now();

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
                                             let sample = generate_sample(&grammar,
                                                                          &distribution,
                                                                          &settings);
                                             future::ok::<(usize, Fitness), ()>((d, sample))
                                         }));
            }
        }

        future::join_all(tasks).wait().unwrap()
    };

    let end_time = time::now();
    let duration = end_time - start_time;
    println!("Duration: {}.{}",
             duration.num_seconds(),
             duration.num_milliseconds());

    let mut csv = String::new();

    let mut csv_file = match OpenOptions::new().append(true).open(csv_path) {
        Ok(file) => file,
        Err(_) => {
            csv += "distribution,score,balance,branching,closeness,drop\n";
            File::create(csv_path).unwrap()
        }
    };

    let csv = samples
        .iter()
        .fold(csv, |csv, &(d, ref f)| if f.is_nothing {
            csv + &format!("{},,,,,\n", d)
        } else {
            csv +
            &format!("{},{},{},{},{},{}\n",
                     d,
                     f.score(),
                     f.balance,
                     f.branching,
                     f.closeness,
                     f.drop)
        });

    csv_file.write_all(csv.as_bytes()).unwrap();
}

fn run_learning(matches: &ArgMatches) {
    fn generate_sample(grammar: &abnf::Ruleset,
                       distribution: &Distribution)
                       -> ol::LSystem {
        let seed = random_seed();
        let genes = generate_genome(&mut XorShiftRng::from_seed(seed), GENOME_LENGTH);
        let mut genotype = WeightedGenotype::new(genes, distribution);
        generate_system(grammar, &mut genotype)
    }

    let learning_rate: f32 = match matches.value_of("learning-rate").unwrap().parse() {
        Ok(value) => value,
        Err(err) => {
            println!("Failed parsing learning-rate argument: {}", err);
            return;
        }
    };
    println!("Using a learning rate of {}", learning_rate);

    let (grammar, distribution) = get_sample_setup();

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
                    println!("Failed deserializing distribution file \"{}\": {}", filename, err);
                    return;
                }
            }
        }
        None => distribution,
    };

    let mut distribution = Arc::new(distribution);

    let bin_path = Path::new(matches.value_of("bin").unwrap());
    println!("Saving distribution to \"{}\".", bin_path.to_str().unwrap());

    let stats_csv_path = Arc::new(matches.value_of("stats-csv").unwrap().to_string());
    println!("Saving distribution to \"{}\".", stats_csv_path);

    let settings = Arc::new(lsys::Settings {
                                width: 0.05,
                                angle: PI / 8.0,
                                iterations: 5,
                                ..lsys::Settings::new()
                            });

    let num_workers = matches.value_of("workers").map_or(num_cpus::get() + 1, |w| w.parse().unwrap());

    let dist_dump_path = Path::new("dist");
    match fs::create_dir_all(dist_dump_path) {
        Ok(_) => {}
        Err(err) => {
            println!("Failed creating distribution dump directory \"{}\": {}",
                     dist_dump_path.to_str().unwrap(),
                     err);
            return;
        }
    }

    let mut dist_files = fs::read_dir(dist_dump_path).unwrap().peekable();
    if dist_files.peek().is_some() {
        println!("There are files in \"{}\", remove these?", dist_dump_path.to_str().unwrap());
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

    let start_time = time::now();
    let pool = CpuPool::new(num_workers);
    let mut rng = rand::thread_rng();

    let measure_distribution = |distribution: Arc<Distribution>| -> (f32, usize) {
        const ERROR_THRESHOLD: f32 = 0.002;
        const MIN_SAMPLES: usize = 64;

        let mut error = f32::MAX;
        let mut step_mean = 0.0;
        let mut step_scores = Vec::new();

        // Generate samples until error is with accepted threshold.
        while error > ERROR_THRESHOLD {
            let tasks: Vec<_> = (0..MIN_SAMPLES)
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
            let unbiased_sample_variance = step_scores
                .iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f32>() / (size - 1) as f32;
            let sample_standard_deviation = unbiased_sample_variance.sqrt();

            error = sample_standard_deviation / size as f32;
            step_mean = mean;
            println!("M: {}, SE: {}", step_mean, error);
        }

        (step_mean, step_scores.len())
    };

    let mut scores = Vec::new();

    println!("Measuring initial distribution");

    let (mut current_score, mut num_samples) = measure_distribution(distribution.clone());
    scores.push((num_samples, num_samples, current_score));

    println!("Generated {} samples.", num_samples);
    println!("Initial distribution has score {}.", current_score);

    let mut distribution_save_future = {
        let distribution = distribution.clone();
        let stats_csv_path = stats_csv_path.clone();

        pool.spawn_fn(move || -> FutureResult<(), ()> {
            let first_dist_filename = format!("{}.csv", 0);
            let first_dist_file_path = dist_dump_path.join(first_dist_filename);
            let mut csv_file = File::create(first_dist_file_path).unwrap();
            csv_file
                .write_all(distribution.to_csv().as_bytes())
                .unwrap();

            let stats_csv = "iteration,samples,measure samples,score,accepted,temperature\n".to_string() +
                &format!("{},{},{},{},,\n", 0, num_samples, num_samples, current_score);
            let mut stats_csv_file = File::create(&*stats_csv_path).unwrap();
            stats_csv_file
                .write_all(stats_csv.as_bytes())
                .unwrap();

            future::ok(())
        })
    };

    let max_iterations = 128_usize;
    let mutation_factor = 0.2;

    for i in 0..max_iterations {
        println!("Iteration {} of {}.", i, max_iterations);

        let mut new_distribution = (*distribution).clone();
        mutate_distribution(&mut new_distribution, mutation_factor, &mut rng);
        new_distribution.normalize();

        let new_distribution = Arc::new(new_distribution);
        let (new_score, new_num_samples) = measure_distribution(new_distribution.clone());
        num_samples += new_num_samples;

        println!("Generated {} of total {} samples.", new_num_samples, num_samples);
        println!("Neighbour score was {} with a difference of {}.", new_score, new_score - current_score);

        let calc_probability = |new, current, temp| -> f32 {
            E.powf((new - current) / temp)
        };

        let temperature = 1.0 - (i as f32 / max_iterations as f32);
        let probability = calc_probability(new_score, current_score, temperature);

        println!("Temperature is {}.", temperature);
        println!("Porbability of being selected is {}.", probability);

        let random = Range::new(0.0, 1.0).ind_sample(&mut rng);

        let accepted = if probability >= random {
            println!("Neighbour was selected.");

            distribution = new_distribution.clone();
            current_score = new_score;
            true
        } else {
            println!("Neighbour was discarded.");
            false
        };

        distribution_save_future.wait().unwrap();
        distribution_save_future = {
            let distribution = new_distribution.clone();
            let stats_csv_path = stats_csv_path.clone();

            pool.spawn_fn(move || -> FutureResult<(), ()> {
                let filename = format!("{}.csv", i);
                let file_path = dist_dump_path.join(filename);
                let mut csv_file = File::create(file_path).unwrap();
                csv_file
                    .write_all(distribution.to_csv().as_bytes())
                    .unwrap();

                let stats_csv = &format!("{},{},{},{},{},{}\n",
                                         i + 1,
                                         num_samples,
                                         new_num_samples,
                                         new_score,
                                         accepted,
                                         temperature);
                let mut stats_csv_file = OpenOptions::new()
                    .append(true)
                    .open(&*stats_csv_path)
                    .unwrap();
                stats_csv_file.write_all(stats_csv.as_bytes()).unwrap();

                future::ok(())
            })
        };
    }

    println!("Finished search.");

    distribution_save_future.wait().unwrap();

    let end_time = time::now();
    let duration = end_time - start_time;
    println!("Duration: {}:{}:{}.{}",
             duration.num_hours(),
             duration.num_minutes() % 60,
             duration.num_seconds() % 60,
             duration.num_milliseconds() % 1000);

    let dist_file = File::create(bin_path).unwrap();
    bincode::serialize_into(&mut BufWriter::new(dist_file),
                            &distribution,
                            bincode::Infinite)
        .unwrap();
}

fn run_distribution_csv_to_bin(matches: &ArgMatches) {
    let input_path = Path::new(matches.value_of("INPUT").unwrap());
    let output_path = match matches.value_of("output") {
        Some(path) => PathBuf::from(path),
        None => input_path.with_extension("bin"),
    };

    let mut input_file = File::open(input_path)
        .expect("Could not open input file");
    let mut csv = String::new();
    input_file.read_to_string(&mut csv).expect("Could not read input file");

    let distribution = Distribution::from_csv(&csv);

    let output_file = File::create(&output_path).unwrap();
    bincode::serialize_into(&mut BufWriter::new(output_file),
                            &distribution,
                            bincode::Infinite)
        .expect("Could not write output file");

    println!("Wrote \"{}\"", output_path.to_str().unwrap());
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
}
