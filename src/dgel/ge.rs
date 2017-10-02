use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use bincode;
use chrono::prelude::*;
use clap::{App, Arg, ArgMatches, SubCommand};
use rand::{self, SeedableRng, XorShiftRng};
use rand::distributions::{IndependentSample, Range};
use serde_yaml;
use rsgenetic::pheno::{Fitness, Phenotype};
use rsgenetic::sim::{Builder, RunResult, SimResult, StepResult};
use rsgenetic::sim::select::{MaximizeSelector, Selector, TournamentSelector};
use futures::future::{self, Future};
use futures_cpupool::CpuPool;
use num_cpus;
use csv;

use lsys::{self, ol};
use yobun::{mean, rand_remove, unbiased_sample_variance, ToSeconds};

use super::fitness;
use dgel::{generate_chromosome, generate_system, get_sample_setup, random_seed, Distribution,
           GenePrimitive, Grammar, WeightedChromosmeStrategy, CHROMOSOME_LEN};

pub const COMMAND_NAME: &'static str = "ge";

pub fn get_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name(COMMAND_NAME)
        .about("Run grammatical evolution")
        .subcommand(
            SubCommand::with_name("run")
                .about("Run GE")
                .arg(
                    Arg::with_name("distribution")
                        .short("d")
                        .long("distribution")
                        .takes_value(true)
                        .help(
                            "Distribution file to use. Otherwise default distribution is used.",
                        ),
                )
                .arg(
                    Arg::with_name("grammar")
                        .short("g")
                        .long("grammar")
                        .takes_value(true)
                        .default_value("grammar/lsys2.abnf")
                        .help("Which ABNF grammar to use"),
                )
                .arg(
                    Arg::with_name("population-size")
                        .short("p")
                        .long("population-size")
                        .takes_value(true)
                        .default_value("800"),
                )
                .arg(
                    Arg::with_name("tournament-size")
                        .short("t")
                        .long("tournament-size")
                        .takes_value(true)
                        .default_value("2"),
                )
                .arg(
                    Arg::with_name("duplication-rate")
                        .long("duplication-rate")
                        .takes_value(true)
                        .default_value("0"),
                )
                .arg(
                    Arg::with_name("mutation-rate")
                        .short("m")
                        .long("mutation-rate")
                        .takes_value(true)
                        .default_value("1.0"),
                )
                .arg(
                    Arg::with_name("crossover-rate")
                        .short("c")
                        .long("crossover-rate")
                        .takes_value(true)
                        .default_value("0.5"),
                )
                .arg(
                    Arg::with_name("max-generations")
                        .long("max-generations")
                        .takes_value(true)
                        .default_value("200"),
                )
                .arg(
                    Arg::with_name("parallel")
                        .long("parallel")
                        .takes_value(true)
                        .default_value("0"),
                )
                .arg(Arg::with_name("prune").long("prune"))
                .arg(Arg::with_name("no-print").long("no-print"))
                .arg(Arg::with_name("no-dump").long("no-dump")),
        )
        .subcommand(
            SubCommand::with_name("size-sampling")
                .about("Find the best GE population size and number of generations")
                .arg(
                    Arg::with_name("distribution")
                        .short("d")
                        .long("distribution")
                        .takes_value(true)
                        .help(
                            "Distribution file to use. Otherwise default distribution is used.",
                        ),
                )
                .arg(
                    Arg::with_name("grammar")
                        .short("g")
                        .long("grammar")
                        .takes_value(true)
                        .default_value("grammar/lsys2.abnf")
                        .help("Which ABNF grammar to use"),
                ),
        )
        .subcommand(
            SubCommand::with_name("tournament-sampling")
                .about("Find the best GE tournament size")
                .arg(
                    Arg::with_name("distribution")
                        .short("d")
                        .long("distribution")
                        .takes_value(true)
                        .help(
                            "Distribution file to use. Otherwise default distribution is used.",
                        ),
                )
                .arg(
                    Arg::with_name("grammar")
                        .short("g")
                        .long("grammar")
                        .takes_value(true)
                        .default_value("grammar/lsys2.abnf")
                        .help("Which ABNF grammar to use"),
                ),
        )
        .subcommand(
            SubCommand::with_name("recombination-sampling")
                .about("Find the best GE crossover and mutation rates")
                .arg(
                    Arg::with_name("distribution")
                        .short("d")
                        .long("distribution")
                        .takes_value(true)
                        .help(
                            "Distribution file to use. Otherwise default distribution is used.",
                        ),
                )
                .arg(
                    Arg::with_name("grammar")
                        .short("g")
                        .long("grammar")
                        .takes_value(true)
                        .default_value("grammar/lsys2.abnf")
                        .help("Which ABNF grammar to use"),
                ),
        )
        .subcommand(
            SubCommand::with_name("duplication-sampling")
                .about("Find the best GE duplication rate")
                .arg(
                    Arg::with_name("distribution")
                        .short("d")
                        .long("distribution")
                        .takes_value(true)
                        .help(
                            "Distribution file to use. Otherwise default distribution is used.",
                        ),
                )
                .arg(
                    Arg::with_name("grammar")
                        .short("g")
                        .long("grammar")
                        .takes_value(true)
                        .default_value("grammar/lsys2.abnf")
                        .help("Which ABNF grammar to use"),
                ),
        )
}

pub fn run(matches: &ArgMatches) {
    if let Some(matches) = matches.subcommand_matches("run") {
        run_ge(matches)
    } else if let Some(matches) = matches.subcommand_matches("size-sampling") {
        run_size_sampling(matches)
    } else if let Some(matches) = matches.subcommand_matches("tournament-sampling") {
        run_tournament_sampling(matches)
    } else if let Some(matches) = matches.subcommand_matches("recombination-sampling") {
        run_recombination_sampling(matches)
    } else if let Some(matches) = matches.subcommand_matches("duplication-sampling") {
        run_duplication_sampling(matches)
    } else {
        println!("Unknown command.");
        return;
    }
}

pub fn run_ge(matches: &ArgMatches) {
    let parallel: usize = matches.value_of("parallel").unwrap().parse().unwrap();

    let settings = Settings {
        population_size: matches
            .value_of("population-size")
            .unwrap()
            .parse()
            .unwrap(),
        tournament_size: matches
            .value_of("tournament-size")
            .unwrap()
            .parse()
            .unwrap(),
        duplication_rate: matches.value_of("duplication-rate").unwrap().parse().unwrap(),
        mutation_rate: matches.value_of("mutation-rate").unwrap().parse().unwrap(),
        crossover_rate: matches.value_of("crossover-rate").unwrap().parse().unwrap(),
        max_iterations: matches
            .value_of("max-generations")
            .unwrap()
            .parse()
            .unwrap(),
        prune: matches.is_present("prune"),
        print: !matches.is_present("no-print"),
        dump: !matches.is_present("no-dump"),
    };

    let (grammar, distribution, lsys_settings, stack_rule_index) =
        get_sample_setup(matches.value_of("grammar").unwrap());

    let distribution = match matches.value_of("distribution") {
        Some(filename) => {
            println!("Using distribution from {}", filename);
            let file = File::open(filename).unwrap();
            let d: Distribution =
                bincode::deserialize_from(&mut BufReader::new(file), bincode::Infinite)
                    .expect("Could not deserialize distribution");
            Arc::new(d)
        }
        None => Arc::new(distribution),
    };

    println!("Settings");
    println!("----------");
    println!("{}", settings);
    println!("");

    let start_time = Instant::now();

    if parallel == 0 {
        evolve(
            &grammar,
            &distribution,
            stack_rule_index,
            &lsys_settings,
            &settings,
        );
    } else {
        let pool = CpuPool::new(num_cpus::get() + 1);

        let grammar = Arc::new(grammar);
        let distribution = Arc::new(distribution);
        let lsys_settings = Arc::new(lsys_settings);
        let settings = Arc::new(settings);

        let tasks: Vec<_> = (0..parallel)
            .map(|_| {
                let grammar = Arc::clone(&grammar);
                let distribution = Arc::clone(&distribution);
                let lsys_settings = Arc::clone(&lsys_settings);
                let settings = Arc::clone(&settings);

                pool.spawn_fn(move || {
                    let best = evolve(
                        &grammar,
                        &distribution,
                        stack_rule_index,
                        &lsys_settings,
                        &settings,
                    );
                    future::ok::<LsysFitness, ()>(best)
                })
            })
            .collect();

        let scores: Vec<f32> = future::join_all(tasks)
            .wait()
            .unwrap()
            .iter()
            .map(|f| f.0)
            .collect();

        let scores_file = File::create("ge-scores.csv").unwrap();
        let mut scores_writer = csv::Writer::from_writer(BufWriter::new(scores_file));
        scores_writer.write_record(&["score"]).unwrap();
        for s in &scores {
            scores_writer.write_record(&[format!("{}", s)]).unwrap();
        }

        let sum: f32 = scores.iter().sum();
        let size = scores.len() as f32;
        let ubniased_size = (scores.len() - 1) as f32;
        let mean = sum / size;
        let unbiased_sample_variance =
            scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / ubniased_size;
        let sample_standard_deviation = unbiased_sample_variance.sqrt();
        let standard_error = sample_standard_deviation / size.sqrt();

        println!("x̄: {}", mean);
        println!("s²: {}", unbiased_sample_variance);
        println!("s: {}", sample_standard_deviation);
        println!("SE: {}", standard_error);
        println!(
            "CSV: {},{},{},{}",
            mean,
            unbiased_sample_variance,
            sample_standard_deviation,
            standard_error
        );
    }

    let duration: f32 = start_time.elapsed().to_seconds();
    println!("Duration: {}", duration);
}

pub fn run_size_sampling(matches: &ArgMatches) {
    #[derive(Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Decision {
        Continue,
        End,
    }

    #[derive(Serialize)]
    struct DataPoint {
        generations: usize,
        population: usize,
        decision: Decision,
        duration: f32,
        average: f32,
        variance: f32,
        min: f32,
        max: f32,
    }

    let (grammar, distribution, lsys_settings, stack_rule_index) =
        get_sample_setup(matches.value_of("grammar").unwrap());

    let generations_start = 100_usize;
    let population_start = 100_usize;
    let sample_size = 20_usize;

    let base_settings = Settings {
        max_iterations: generations_start as u64,
        population_size: population_start,
        tournament_size: 5,
        duplication_rate: 0.0,
        crossover_rate: 0.5,
        mutation_rate: 0.1,
        prune: false,
        dump: false,
        print: false,
    };

    println!("Settings");
    println!("----------");
    println!("{}", base_settings);
    println!("");

    let distribution = match matches.value_of("distribution") {
        Some(filename) => {
            println!("Using distribution from {}", filename);
            let file = File::open(filename).unwrap();
            let d: Distribution =
                bincode::deserialize_from(&mut BufReader::new(file), bincode::Infinite)
                    .expect("Could not deserialize distribution");
            Arc::new(d)
        }
        None => Arc::new(distribution),
    };

    let pool = CpuPool::new(num_cpus::get() + 1);
    let grammar = Arc::new(grammar);
    let distribution = Arc::new(distribution);
    let lsys_settings = Arc::new(lsys_settings);

    let data_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("ge-size-sampling.csv")
        .expect("Could not create data file");
    let mut data_writer = csv::Writer::from_writer(BufWriter::with_capacity(1024 * 512, data_file));

    let mut frontier = VecDeque::with_capacity(1);
    frontier.push_back((population_start, generations_start, 0.0));

    let mut results = Vec::new();

    while let Some((population_size, num_generations, previous_score)) = frontier.pop_front() {
        println!("Examining p={}, g={}", population_size, num_generations,);

        let settings = Arc::new(Settings {
            population_size: population_size,
            max_iterations: num_generations as u64,
            ..base_settings
        });

        let tasks: Vec<_> = (0..sample_size)
            .map(|_| {
                let grammar = Arc::clone(&grammar);
                let distribution = Arc::clone(&distribution);
                let lsys_settings = Arc::clone(&lsys_settings);
                let settings = Arc::clone(&settings);

                pool.spawn_fn(move || {
                    let best = evolve(
                        &grammar,
                        &distribution,
                        stack_rule_index,
                        &lsys_settings,
                        &settings,
                    );
                    future::ok::<LsysFitness, ()>(best)
                })
            })
            .collect();

        let start_time = Instant::now();
        let scores: Vec<f32> = future::join_all(tasks)
            .wait()
            .unwrap()
            .iter()
            .map(|f| f.0)
            .collect();
        let duration = start_time.elapsed().to_seconds();

        let mean = mean(&scores);
        let variance = unbiased_sample_variance(&scores, mean);
        let min: f32 = *scores
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max: f32 = *scores
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let score = mean;

        println!(
            "Score for p={}, g={} is {}",
            population_size,
            num_generations,
            score
        );

        if score >= previous_score {
            println!("Found improvement. Exploring!");

            data_writer
                .serialize(DataPoint {
                    generations: num_generations,
                    population: population_size,
                    decision: Decision::Continue,
                    duration: duration,
                    average: mean,
                    max: max,
                    min: min,
                    variance: variance,
                })
                .expect("Could not write to data file");

            // Breath first search
            if population_size > num_generations {
                frontier.push_back((population_size * 2, num_generations, score));
            } else if num_generations > population_size {
                frontier.push_back((population_size, num_generations * 2, score));
            } else {
                frontier.push_back((population_size * 2, num_generations, score));
                frontier.push_back((population_size, num_generations * 2, score));
                frontier.push_back((population_size * 2, num_generations * 2, score));
            }
        } else {
            println!("Dead end. Giving up this path");

            data_writer
                .serialize(DataPoint {
                    generations: num_generations,
                    population: population_size,
                    decision: Decision::End,
                    duration: duration,
                    average: mean,
                    max: max,
                    min: min,
                    variance: variance,
                })
                .expect("Could not write to data file");

            results.push((population_size, num_generations, score));
        }

        // Make sure that data is written, in case the program is aborted.
        data_writer.flush().expect("Could not write data file");
    }

    let (best_population_size, best_num_generations, _) = *results
        .iter()
        .max_by(|&&(_, _, score_a), &&(_, _, score_b)| {
            score_a.partial_cmp(&score_b).unwrap()
        })
        .unwrap();

    println!(
        "Best parameters are population_size={}, num_generations={}",
        best_population_size,
        best_num_generations
    );
}

pub fn run_tournament_sampling(matches: &ArgMatches) {
    #[derive(Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Decision {
        Continue,
        End,
    }

    #[derive(Serialize)]
    struct DataPoint {
        size: usize,
        decision: Decision,
        duration: f32,
        average: f32,
        variance: f32,
        min: f32,
        max: f32,
    }

    let (grammar, distribution, lsys_settings, stack_rule_index) =
        get_sample_setup(matches.value_of("grammar").unwrap());

    let tournament_size_start = 2_usize;
    let sample_size = 20_usize;

    let base_settings = Settings {
        max_iterations: 200,
        population_size: 800,
        tournament_size: tournament_size_start,
        duplication_rate: 0.0,
        crossover_rate: 0.5,
        mutation_rate: 0.1,
        prune: false,
        dump: false,
        print: false,
    };

    println!("Settings");
    println!("----------");
    println!("{}", base_settings);
    println!("");

    let distribution = match matches.value_of("distribution") {
        Some(filename) => {
            println!("Using distribution from {}", filename);
            let file = File::open(filename).unwrap();
            let d: Distribution =
                bincode::deserialize_from(&mut BufReader::new(file), bincode::Infinite)
                    .expect("Could not deserialize distribution");
            Arc::new(d)
        }
        None => Arc::new(distribution),
    };

    let pool = CpuPool::new(num_cpus::get() + 1);
    let grammar = Arc::new(grammar);
    let distribution = Arc::new(distribution);
    let lsys_settings = Arc::new(lsys_settings);

    let data_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("ge-tournament-sampling.csv")
        .expect("Could not create data file");
    let mut data_writer = csv::Writer::from_writer(BufWriter::with_capacity(1024 * 512, data_file));

    let mut next = Some((tournament_size_start, 0.0_f32));

    while let Some((tournament_size, previous_score)) = next {
        println!("Examining t={}", tournament_size,);

        let settings = Arc::new(Settings {
            tournament_size: tournament_size,
            ..base_settings
        });

        let tasks: Vec<_> = (0..sample_size)
            .map(|_| {
                let grammar = Arc::clone(&grammar);
                let distribution = Arc::clone(&distribution);
                let lsys_settings = Arc::clone(&lsys_settings);
                let settings = Arc::clone(&settings);

                pool.spawn_fn(move || {
                    let best = evolve(
                        &grammar,
                        &distribution,
                        stack_rule_index,
                        &lsys_settings,
                        &settings,
                    );
                    future::ok::<LsysFitness, ()>(best)
                })
            })
            .collect();

        let start_time = Instant::now();
        let scores: Vec<f32> = future::join_all(tasks)
            .wait()
            .unwrap()
            .iter()
            .map(|f| f.0)
            .collect();
        let duration = start_time.elapsed().to_seconds();

        let mean = mean(&scores);
        let variance = unbiased_sample_variance(&scores, mean);
        let min: f32 = *scores
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max: f32 = *scores
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let score = mean;

        println!("Score for t={} is {}", tournament_size, score);

        if score >= previous_score {
            println!("Found improvement. Exploring!");

            data_writer
                .serialize(DataPoint {
                    size: tournament_size,
                    decision: Decision::Continue,
                    duration: duration,
                    average: mean,
                    max: max,
                    min: min,
                    variance: variance,
                })
                .expect("Could not write to data file");

            next = Some((tournament_size * 2, score));
        } else {
            println!("Dead end. Giving up.");

            data_writer
                .serialize(DataPoint {
                    size: tournament_size,
                    decision: Decision::End,
                    duration: duration,
                    average: mean,
                    max: max,
                    min: min,
                    variance: variance,
                })
                .expect("Could not write to data file");

            next = None;
        };

        // Make sure that data is written, in case the program is aborted.
        data_writer.flush().expect("Could not write data file");
    }
}

pub fn run_recombination_sampling(matches: &ArgMatches) {
    #[derive(Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Decision {
        Continue,
        End,
    }

    #[derive(Serialize)]
    struct DataPoint {
        crossover_rate: f32,
        mutation_rate: f32,
        decision: Decision,
        duration: f32,
        average: f32,
        variance: f32,
        min: f32,
        max: f32,
    }

    let (grammar, distribution, lsys_settings, stack_rule_index) =
        get_sample_setup(matches.value_of("grammar").unwrap());

    let sample_size = 20_usize;
    let steps = [0.01, 0.1, 0.5, 1.0];

    let base_settings = Settings {
        max_iterations: 200,
        population_size: 800,
        tournament_size: 2,
        duplication_rate: 0.0,
        prune: false,
        dump: false,
        print: false,
        ..Settings::default()
    };

    println!("Settings");
    println!("----------");
    println!("{}", base_settings);
    println!("");

    let distribution = match matches.value_of("distribution") {
        Some(filename) => {
            println!("Using distribution from {}", filename);
            let file = File::open(filename).unwrap();
            let d: Distribution =
                bincode::deserialize_from(&mut BufReader::new(file), bincode::Infinite)
                    .expect("Could not deserialize distribution");
            Arc::new(d)
        }
        None => Arc::new(distribution),
    };

    let pool = CpuPool::new(num_cpus::get() + 1);
    let grammar = Arc::new(grammar);
    let distribution = Arc::new(distribution);
    let lsys_settings = Arc::new(lsys_settings);

    let data_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("ge-recombination-sampling.csv")
        .expect("Could not create data file");
    let mut data_writer = csv::Writer::from_writer(BufWriter::with_capacity(1024 * 512, data_file));

    let mut frontier = VecDeque::with_capacity(1);
    frontier.push_back((0, 0, 0.0));

    let mut results = Vec::new();

    while let Some((crossover_step, mutation_step, previous_score)) = frontier.pop_front() {
        let crossover_rate = steps[crossover_step];
        let mutation_rate = steps[mutation_step];
        println!("Examining c={}, m={}", crossover_rate, mutation_rate);

        let settings = Arc::new(Settings {
            crossover_rate: crossover_rate,
            mutation_rate: mutation_rate,
            ..base_settings
        });

        let tasks: Vec<_> = (0..sample_size)
            .map(|_| {
                let grammar = Arc::clone(&grammar);
                let distribution = Arc::clone(&distribution);
                let lsys_settings = Arc::clone(&lsys_settings);
                let settings = Arc::clone(&settings);

                pool.spawn_fn(move || {
                    let best = evolve(
                        &grammar,
                        &distribution,
                        stack_rule_index,
                        &lsys_settings,
                        &settings,
                    );
                    future::ok::<LsysFitness, ()>(best)
                })
            })
            .collect();

        let start_time = Instant::now();
        let scores: Vec<f32> = future::join_all(tasks)
            .wait()
            .unwrap()
            .iter()
            .map(|f| f.0)
            .collect();
        let duration = start_time.elapsed().to_seconds();

        let mean = mean(&scores);
        let variance = unbiased_sample_variance(&scores, mean);
        let min: f32 = *scores
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max: f32 = *scores
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let score = mean;

        println!("Score for c={}, m={} is {}", crossover_rate, mutation_rate, score);

        if score >= previous_score {
            println!("Found improvement. Exploring!");

            data_writer
                .serialize(DataPoint {
                    crossover_rate: crossover_rate,
                    mutation_rate: mutation_rate,
                    decision: Decision::Continue,
                    duration: duration,
                    average: mean,
                    max: max,
                    min: min,
                    variance: variance,
                })
                .expect("Could not write to data file");

            let next_crossover = crossover_step + 1;
            let next_mutation = mutation_step + 1;
            // Breath first search
            if crossover_step > mutation_step && next_crossover < steps.len() {
                frontier.push_back((next_crossover, mutation_step, score));
            } else if mutation_step > crossover_step && next_mutation < steps.len() {
                frontier.push_back((crossover_step, next_mutation, score));
            } else if next_crossover < steps.len() && next_mutation < steps.len() {
                frontier.push_back((next_crossover, mutation_step, score));
                frontier.push_back((crossover_step, next_mutation, score));
                frontier.push_back((next_crossover, next_mutation, score));
            } else {
                println!("Finished path.");
            }
        } else {
            println!("Dead end. Giving up.");

            data_writer
                .serialize(DataPoint {
                    crossover_rate: crossover_rate,
                    mutation_rate: mutation_rate,
                    decision: Decision::End,
                    duration: duration,
                    average: mean,
                    max: max,
                    min: min,
                    variance: variance,
                })
                .expect("Could not write to data file");

            results.push((crossover_step, mutation_step, score));
        };

        // Make sure that data is written, in case the program is aborted.
        data_writer.flush().expect("Could not write data file");
    }

    let (best_crossover_step, best_mutation_step, _) = results
        .into_iter()
        .max_by(|&(_, _, score_a), &(_, _, score_b)| {
            score_a.partial_cmp(&score_b).unwrap()
        })
        .unwrap();

    println!(
        "Best parameters are crossover_rate={}, mutation_rate={}",
        steps[best_crossover_step],
        steps[best_mutation_step]
    );
}

pub fn run_duplication_sampling(matches: &ArgMatches) {
    #[derive(Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Decision {
        Continue,
        End,
    }

    #[derive(Serialize)]
    struct DataPoint {
        rate: f32,
        decision: Decision,
        duration: f32,
        average: f32,
        variance: f32,
        min: f32,
        max: f32,
    }

    let (grammar, distribution, lsys_settings, stack_rule_index) =
        get_sample_setup(matches.value_of("grammar").unwrap());

    let sample_size = 20_usize;
    let steps = [0.01, 0.1, 0.5, 1.0];

    let base_settings = Settings {
        max_iterations: 200,
        population_size: 800,
        tournament_size: 2,
        crossover_rate: 0.5,
        mutation_rate: 1.0,
        prune: true,
        dump: false,
        print: false,
        ..Settings::default()
    };

    println!("Settings");
    println!("----------");
    println!("{}", base_settings);
    println!("");

    let distribution = match matches.value_of("distribution") {
        Some(filename) => {
            println!("Using distribution from {}", filename);
            let file = File::open(filename).unwrap();
            let d: Distribution =
                bincode::deserialize_from(&mut BufReader::new(file), bincode::Infinite)
                    .expect("Could not deserialize distribution");
            Arc::new(d)
        }
        None => Arc::new(distribution),
    };

    let pool = CpuPool::new(num_cpus::get() + 1);
    let grammar = Arc::new(grammar);
    let distribution = Arc::new(distribution);
    let lsys_settings = Arc::new(lsys_settings);

    let data_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("ge-duplication-sampling.csv")
        .expect("Could not create data file");
    let mut data_writer = csv::Writer::from_writer(BufWriter::with_capacity(1024 * 512, data_file));

    let mut next = Some((0, 0.0_f32));

    while let Some((step, previous_score)) = next {
        let duplication_rate = steps[step];
        println!("Examining d={}", duplication_rate);

        let settings = Arc::new(Settings {
            duplication_rate: duplication_rate,
            ..base_settings
        });

        let tasks: Vec<_> = (0..sample_size)
            .map(|_| {
                let grammar = Arc::clone(&grammar);
                let distribution = Arc::clone(&distribution);
                let lsys_settings = Arc::clone(&lsys_settings);
                let settings = Arc::clone(&settings);

                pool.spawn_fn(move || {
                    let best = evolve(
                        &grammar,
                        &distribution,
                        stack_rule_index,
                        &lsys_settings,
                        &settings,
                    );
                    future::ok::<LsysFitness, ()>(best)
                })
            })
            .collect();

        let start_time = Instant::now();
        let scores: Vec<f32> = future::join_all(tasks)
            .wait()
            .unwrap()
            .iter()
            .map(|f| f.0)
            .collect();
        let duration = start_time.elapsed().to_seconds();

        let mean = mean(&scores);
        let variance = unbiased_sample_variance(&scores, mean);
        let min: f32 = *scores
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max: f32 = *scores
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let score = mean;

        println!("Score for d={} is {}", duplication_rate, score);

        if score >= previous_score {
            println!("Found improvement. Exploring!");

            data_writer
                .serialize(DataPoint {
                    rate: duplication_rate,
                    decision: Decision::Continue,
                    duration: duration,
                    average: mean,
                    max: max,
                    min: min,
                    variance: variance,
                })
                .expect("Could not write to data file");

            next = Some((step + 1, score));
        } else {
            println!("Dead end. Giving up.");

            data_writer
                .serialize(DataPoint {
                    rate: duplication_rate,
                    decision: Decision::End,
                    duration: duration,
                    average: mean,
                    max: max,
                    min: min,
                    variance: variance,
                })
                .expect("Could not write to data file");

            next = None;
        };

        // Make sure that data is written, in case the program is aborted.
        data_writer.flush().expect("Could not write data file");
    }
}

/// Settings for GE simulation
#[derive(Debug)]
struct Settings {
    population_size: usize,
    duplication_rate: f32,
    mutation_rate: f32,
    crossover_rate: f32,
    max_iterations: u64,
    tournament_size: usize,
    prune: bool,
    /// Print output while running
    print: bool,
    /// Dump stats and the resulting model
    dump: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            population_size: 800,
            max_iterations: 200,
            tournament_size: 2,
            duplication_rate: 0.0,
            crossover_rate: 0.5,
            mutation_rate: 1.0,
            prune: false,
            print: false,
            dump: false,
        }
    }
}

impl Display for Settings {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        writeln!(f, "Population size: {}", self.population_size)?;
        writeln!(f, "Max iterations: {}", self.max_iterations)?;
        writeln!(f, "Tournament size: {}", self.tournament_size)?;
        writeln!(f, "Duplication rate: {}", self.duplication_rate)?;
        writeln!(f, "Crossover rate: {}", self.crossover_rate)?;
        writeln!(f, "Mutation rate: {}", self.mutation_rate)?;
        writeln!(f, "Prune: {}", self.prune)?;
        writeln!(f, "Print: {}", self.print)?;
        write!(f, "Dump: {}", self.dump)
    }
}

fn evolve(
    grammar: &Grammar,
    distribution: &Distribution,
    stack_rule_index: usize,
    lsys_settings: &lsys::Settings,
    settings: &Settings,
) -> LsysFitness {
    let base_phenotype = LsysPhenotype::new(
        grammar,
        distribution,
        stack_rule_index,
        lsys_settings,
        Vec::new(),
    );

    let stats_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("ge-stats.csv")
        .expect("Could not create stats file");
    let mut stats_writer = BufWriter::new(stats_file);

    if settings.dump {
        let stats_csv = "iteration,avg,best\n";
        stats_writer
            .write_all(stats_csv.as_bytes())
            .expect("Could not write to stats file");
    }

    let mut dumped_mid_distribution = false;

    let best = {
        let mut rng = XorShiftRng::from_seed(random_seed());

        if settings.print {
            println!(
                "Generating initial population of {} individuals.",
                settings.population_size
            );
        }

        let population: Vec<_> = (0..settings.population_size)
            .map(|_| {
                let chromosome = generate_chromosome(&mut rng, CHROMOSOME_LEN);
                base_phenotype.clone_with_chromosome(chromosome)
            })
            .collect();

        if settings.dump {
            println!("Dumping initial fitness distribution.");
            write_fitness_distribution("ge-distribution-initial.csv", &population);
        }

        if settings.print {
            println!("Building simulator.");
        }

        let mut builder = Simulator::builder(population)
            .set_selector(Box::new(TournamentSelector::new(settings.tournament_size)))
            .set_max_iters(settings.max_iterations);

        if settings.prune {
            builder = builder.chain_operator(move |population| {
                population
                    .into_iter()
                    .map(|x| {
                        x.prune()
                    })
                    .collect()
            });
        }

        if settings.duplication_rate > 0.0 {
            let mut rng = rng.clone();
            let duplication_rate = settings.duplication_rate;
            builder = builder.chain_operator(move |population| {
                population
                    .into_iter()
                    .map(|x| {
                        if Range::new(0.0, 1.0).ind_sample(&mut rng) > duplication_rate {
                            x
                        } else {
                            x.duplicate()
                        }
                    })
                    .collect()
            });
        }

        if settings.crossover_rate > 0.0 {
            let mut rng = rng.clone();
            let crossover_rate = settings.crossover_rate;
            builder = builder.chain_operator(move |mut population| {
                let num_individuals = population.len();
                let num_children = (num_individuals as f32 * crossover_rate) as usize;
                let range = Range::new(0, population.len());

                let mut children: Vec<_> = (0..num_children)
                    .map(|_| {
                        let i = range.ind_sample(&mut rng);
                        let j = range.ind_sample(&mut rng);
                        let a = &population[i];
                        let b = &population[j];
                        a.crossover(b)
                    })
                    .collect();

                let num_survivors = num_individuals - num_children;
                let mut survivors = rand_remove(&mut rng, &mut population, num_survivors);
                children.append(&mut survivors);

                children
            });
        }

        if settings.mutation_rate > 0.0 {
            let mut rng = rng.clone();
            let mutation_rate = settings.mutation_rate;
            builder = builder.chain_operator(move |population| {
                population
                    .into_iter()
                    .map(|x| {
                        if Range::new(0.0, 1.0).ind_sample(&mut rng) > mutation_rate {
                            x
                        } else {
                            x.mutate()
                        }
                    })
                    .collect()
            });
        }

        if settings.dump {
            builder = builder.set_step_callback(
                |iteration: u64, population: &[LsysPhenotype]| {
                    let fitnesses: Vec<_> = population.iter().map(|p| p.fitness().0).collect();

                    let sum: f32 = fitnesses.iter().sum();
                    let average = sum / population.len() as f32;
                    let best = fitnesses
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();

                    if !dumped_mid_distribution && average >= 0.5 {
                        println!("Dumping mid fitness distribution.");
                        write_fitness_distribution("ge-distribution-mid.csv", population);
                        dumped_mid_distribution = true;
                    }

                    let stats_csv = format!("{},{},{}\n", iteration, average, best);
                    stats_writer
                        .write_all(stats_csv.as_bytes())
                        .expect("Could not write to stats file");
                },
            );
        }

        let mut simulator = builder.build();

        if settings.print {
            println!("Evolving...");
        }

        simulator.run();

        if settings.dump {
            println!("Dumping final fitness distribution.");
            write_fitness_distribution("ge-distribution-final.csv", &simulator.population);
        }

        simulator.get().unwrap().clone()
    };

    if settings.print {
        println!("Done.");
        println!("Fitness: {}", best.fitness());
    }

    if settings.dump {
        let model_dir = Path::new("model");
        fs::create_dir_all(model_dir).unwrap();

        let filename = format!("{}.yaml", Local::now().to_rfc3339());
        let path = model_dir.join(filename);

        let file = File::create(&path).unwrap();
        let lsystem = generate_system(
            grammar,
            &mut WeightedChromosmeStrategy::new(&best.chromosome, distribution, stack_rule_index),
        );
        serde_yaml::to_writer(&mut BufWriter::new(file), &lsystem).unwrap();

        println!("Saved to {}", path.to_str().unwrap());
    }

    best.fitness()
}

fn write_fitness_distribution<P: AsRef<Path>>(path: P, population: &[LsysPhenotype]) {
    let file = File::create(path).unwrap();
    let mut writer = csv::Writer::from_writer(BufWriter::new(file));
    writer.write_record(&["fitness"]).unwrap();

    for p in population {
        writer.write_record(&[format!("{}", p.fitness().0)]).unwrap();
    }
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
struct LsysFitness(f32);

impl Eq for LsysFitness {}

impl Ord for LsysFitness {
    fn cmp(&self, other: &LsysFitness) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("Fitness is NaN and can't be ordered")
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

impl Display for LsysFitness {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug)]
struct LsysPhenotype<'a> {
    grammar: &'a Grammar,
    distribution: &'a Distribution,
    stack_rule_index: usize,
    settings: &'a lsys::Settings,
    chromosome: Vec<GenePrimitive>,
    lsystem: RefCell<Option<(Rc<ol::LSystem>, usize)>>,
    fitness: Cell<Option<LsysFitness>>,
}

impl<'a> LsysPhenotype<'a> {
    pub fn new(
        grammar: &'a Grammar,
        distribution: &'a Distribution,
        stack_rule_index: usize,
        settings: &'a lsys::Settings,
        chromosome: Vec<GenePrimitive>,
    ) -> Self {
        LsysPhenotype {
            grammar: grammar,
            distribution: distribution,
            stack_rule_index: stack_rule_index,
            settings: settings,
            chromosome: chromosome,
            lsystem: RefCell::new(None),
            fitness: Cell::new(None),
        }
    }

    pub fn clone_with_chromosome(&self, chromosome: Vec<GenePrimitive>) -> Self {
        LsysPhenotype::new(
            self.grammar,
            self.distribution,
            self.stack_rule_index,
            self.settings,
            chromosome,
        )
    }

    fn generate_lsystem(&self) -> (ol::LSystem, usize) {
        let mut strategy = WeightedChromosmeStrategy::new(
            &self.chromosome,
            self.distribution,
            self.stack_rule_index,
        );
        let lsystem = generate_system(self.grammar, &mut strategy);
        let genes_used = strategy.genotype.genes_used();

        (lsystem, genes_used)
    }

    pub fn lsystem(&self) -> Rc<ol::LSystem> {
        if let Some((ref lsystem, _)) = *self.lsystem.borrow() {
            return Rc::clone(lsystem);
        }

        let (lsystem, genes_used) = self.generate_lsystem();
        let lsystem = Rc::new(lsystem);
        *self.lsystem.borrow_mut() = Some((Rc::clone(&lsystem), genes_used));

        lsystem
    }

    pub fn genes_used(&self) -> usize {
        if let Some((_, genes_used)) = *self.lsystem.borrow() {
            return genes_used;
        }

        let (lsystem, genes_used) = self.generate_lsystem();
        let lsystem = Rc::new(lsystem);
        *self.lsystem.borrow_mut() = Some((lsystem, genes_used));

        genes_used
    }

    /// How many introns (unused genes) there are in the chromosome.
    pub fn introns(&self) -> usize {
        let genes_used = self.genes_used();
        if genes_used < self.chromosome.len() {
            self.chromosome.len() - genes_used
        } else {
            0
        }
    }

    /// Remove introns (unused genes)
    pub fn prune(mut self) -> Self {
        let genes_used = self.genes_used();
        if genes_used < self.chromosome.len() {
            self.chromosome.truncate(genes_used);
        }

        self
    }

    /// Duplicate genes
    pub fn duplicate(mut self) -> Self {
        let mut rng = rand::thread_rng();

        let num_genes = Range::new(1, self.chromosome.len() + 1).ind_sample(&mut rng);
        let start = Range::new(0, self.chromosome.len() - num_genes + 1).ind_sample(&mut rng);
        let end = start + num_genes;

        let mut duplicate = self.chromosome[start..end].to_vec();
        self.chromosome.append(&mut duplicate);

        self
    }
}

impl<'a> Phenotype<LsysFitness> for LsysPhenotype<'a> {
    fn fitness(&self) -> LsysFitness {
        if let Some(ref fitness) = self.fitness.get() {
            *fitness
        } else {
            let fitness = fitness::evaluate(&self.lsystem(), self.settings);
            let fitness = LsysFitness(fitness.0.score());
            self.fitness.set(Some(fitness));

            fitness
        }
    }

    fn crossover(&self, other: &Self) -> Self {
        let crossover_point =
            Range::new(0, self.chromosome.len()).ind_sample(&mut rand::thread_rng());
        let iter_self = self.chromosome.iter().take(crossover_point);

        let iter_other = other.chromosome.iter().skip(crossover_point);
        let chromosome = iter_self.chain(iter_other).cloned().collect();

        self.clone_with_chromosome(chromosome)
    }

    fn mutate(mut self) -> Self {
        let mut rng = rand::thread_rng();

        let mutation_index = Range::new(0, self.chromosome.len()).ind_sample(&mut rand::thread_rng());
        self.chromosome[mutation_index] =
            Range::new(GenePrimitive::min_value(), GenePrimitive::max_value()).ind_sample(&mut rng);

        self
    }
}

type StepCallback<'a, T> = Box<FnMut(u64, &[T]) + 'a>;
type Operator<'a, T> = Box<FnMut(Vec<T>) -> Vec<T> + 'a>;

/// A genetic algorithm implementation, designed after "A Comparison of Selection Schemes used
/// in Genetic Algorithms", Tobias Blickle and Lothar Thiele, 1995. It is a modification of
/// `rsgenetic::sim::seq::Simulator`.
///
/// By default it has no genetic operators (`operators`). To add one, use
/// `SimulatorBuilder::chain_operator`.
struct Simulator<'a, T, F> {
    population: Vec<T>,
    selector: Box<Selector<T, F>>,
    iteration_limit: u64,
    iteration: u64,
    step_callback: Option<StepCallback<'a, T>>,
    operators: Vec<Operator<'a, T>>,
}

impl<'a, T, F> Simulator<'a, T, F>
where
    T: Phenotype<F>,
    F: Fitness,
{
    /// Create builder.
    fn builder(population: Vec<T>) -> SimulatorBuilder<'a, T, F> {
        let sim = Simulator {
            population: population,
            selector: Box::new(MaximizeSelector::new(3)),
            iteration_limit: 100,
            iteration: 0,
            step_callback: None,
            operators: Vec::new(),
        };

        SimulatorBuilder { sim: sim }
    }

    fn step(&mut self) -> StepResult {
        if self.population.is_empty() {
            println!(
                "Tried to run a simulator without a population, or the \
                 population was empty."
            );
            return StepResult::Failure;
        }

        if self.iteration >= self.iteration_limit {
            return StepResult::Done;
        }

        let next_population: Vec<T> = {
            let parents = match self.selector.select(&self.population) {
                Ok(parents) => parents,
                Err(e) => {
                    println!("Error selecting parents: {}", e);
                    return StepResult::Failure;
                }
            };

            let population_size = self.population.len();

            let recombined: Vec<T> = self.operators.iter_mut().fold(
                parents.into_iter().cloned().collect(),
                |population, operator| operator(population),
            );
            assert_eq!(recombined.len(), population_size);

            recombined
        };

        self.population = next_population;
        self.iteration += 1;

        if let Some(ref mut callback) = self.step_callback {
            callback(self.iteration, &self.population);
        }

        StepResult::Success // Not done yet, but successful
    }

    fn run(&mut self) -> RunResult {
        // Loop until Failure or Done.
        loop {
            match self.step() {
                StepResult::Success => {}
                StepResult::Failure => return RunResult::Failure,
                StepResult::Done => return RunResult::Done,
            }
        }
    }

    fn get(&self) -> SimResult<T> {
        Ok(self.population.iter().max_by_key(|x| x.fitness()).unwrap())
    }
}

impl<'a, T, F> fmt::Debug for Simulator<'a, T, F>
where
    T: Phenotype<F> + fmt::Debug,
    F: Fitness,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Simulator")
            .field("population", &self.population)
            .field("selector", &self.selector)
            .field("iteration_limit", &self.iteration_limit)
            .field("iteration", &self.iteration)
            .field("step_callback", &self.step_callback.is_some())
            .finish()
    }
}

/// A `Builder` for the `Simulator` type.
struct SimulatorBuilder<'a, T, F>
where
    T: Phenotype<F>,
    F: Fitness,
{
    sim: Simulator<'a, T, F>,
}

impl<'a, T, F> SimulatorBuilder<'a, T, F>
where
    T: Phenotype<F>,
    F: Fitness,
{
    fn set_selector(mut self, sel: Box<Selector<T, F>>) -> Self {
        self.sim.selector = sel;
        self
    }

    fn set_max_iters(mut self, i: u64) -> Self {
        self.sim.iteration_limit = i;
        self
    }

    fn set_step_callback<C>(mut self, callback: C) -> Self
    where
        C: FnMut(u64, &[T]) + 'a,
    {
        self.sim.step_callback = Some(Box::new(callback));
        self
    }

    /// Chain an `operator` in the operator chain used for creating the new population from the
    /// current population. An `operator` gets the current population as the parameter,
    /// and uses it to produce a new population as the return value. The last `operator` in the
    /// chain **must** return a population of size equal to the input population of the first
    /// `operator`.
    fn chain_operator<O>(mut self, operator: O) -> Self
    where
        O: FnMut(Vec<T>) -> Vec<T> + 'a,
    {
        self.sim.operators.push(Box::new(operator));
        self
    }
}

impl<'a, T, F> Builder<Simulator<'a, T, F>> for SimulatorBuilder<'a, T, F>
where
    T: Phenotype<F>,
    F: Fitness,
{
    fn build(self) -> Simulator<'a, T, F> {
        self.sim
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_prune() {
        let (grammar, distribution, lsys_settings, stack_rule_index) =
            get_sample_setup("grammar/lsys2.abnf");
        let phenotype = LsysPhenotype::new(
            &grammar,
            &distribution,
            stack_rule_index,
            &lsys_settings,
            vec![
                // Axiom
                0_u32, // string len 1
                0, // symbol
                0, // variable
                0, // x41
                // Productions
                0, // 1 production
                // Predecessor
                0, // x41
                // Successor
                0, // string len 1
                0, // symbol
                0, // variable
                0, // x41
                // 4 Introns
                0,
                0,
                0,
                0,
            ]
        );

        let chromosome_len = phenotype.chromosome.len();
        let pruned = phenotype.prune();
        assert_eq!(pruned.chromosome.len(), chromosome_len - 4);
    }
}
