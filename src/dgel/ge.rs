use std::cell::Cell;
use std::collections::VecDeque;
use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use bincode;
use chrono::prelude::*;
use clap::{App, Arg, ArgMatches, SubCommand};
use rand::{self, thread_rng, Rng, SeedableRng, XorShiftRng};
use rand::distributions::{IndependentSample, Range};
use serde_yaml;
use rsgenetic::pheno::{Fitness, Phenotype};
use rsgenetic::sim::{Builder, NanoSecond, RunResult, SimResult, Simulation, StepResult};
use rsgenetic::sim::select::{MaximizeSelector, Selector, TournamentSelector};
use futures::future::{self, Future};
use futures_cpupool::CpuPool;
use num_cpus;
use csv;

use lsys;

use super::fitness;
use dgel::{generate_chromosome, generate_system, get_sample_setup, random_seed, Distribution,
           Gene, GenePrimitive, Grammar, WeightedChromosmeStrategy, CHROMOSOME_LEN};

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
                        .default_value("500"),
                )
                .arg(
                    Arg::with_name("selection-size")
                        .short("s")
                        .long("selection-size")
                        .takes_value(true)
                        .default_value("16"),
                )
                .arg(
                    Arg::with_name("tournament-size")
                        .short("t")
                        .long("tournament-size")
                        .takes_value(true)
                        .default_value("50"),
                )
                .arg(
                    Arg::with_name("mutation-rate")
                        .short("m")
                        .long("mutation-rate")
                        .takes_value(true)
                        .default_value("0.1"),
                )
                .arg(
                    Arg::with_name("max-generations")
                        .long("max-generations")
                        .takes_value(true)
                        .default_value("60"),
                )
                .arg(Arg::with_name("no-mutate").long("no-mutate"))
                .arg(Arg::with_name("no-crossover").long("no-crossover"))
                .arg(
                    Arg::with_name("parallel")
                        .long("parallel")
                        .takes_value(true)
                        .default_value("0"),
                )
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
}

pub fn run(matches: &ArgMatches) {
    if let Some(matches) = matches.subcommand_matches("run") {
        run_ge(matches)
    } else if let Some(matches) = matches.subcommand_matches("size-sampling") {
        run_size_sampling(matches)
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
        selection_size: matches.value_of("selection-size").unwrap().parse().unwrap(),
        tournament_size: matches
            .value_of("tournament-size")
            .unwrap()
            .parse()
            .unwrap(),
        mutation_rate: matches.value_of("mutation-rate").unwrap().parse().unwrap(),
        max_iterations: matches
            .value_of("max-generations")
            .unwrap()
            .parse()
            .unwrap(),
        mutate: !matches.is_present("no-mutate"),
        crossover: !matches.is_present("no-crossover"),
        print: !matches.is_present("no-print"),
        dump: !matches.is_present("no-dump"),
    };

    let (grammar, distribution, lsys_settings) =
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

    println!("{:#?}", settings);

    let stack_rule_index = grammar.symbol_index("stack").unwrap();

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
        average: f32,
        max: f32,
    }

    let (grammar, distribution, lsys_settings) =
        get_sample_setup(matches.value_of("grammar").unwrap());

    let generations_start = 100_usize;
    let population_start = 100_usize;
    let sample_size = 20_usize;

    let base_settings = Settings {
        max_iterations: generations_start as u64,
        population_size: population_start,
        selection_size: 16,
        tournament_size: 50,
        mutation_rate: 0.4,
        dump: false,
        print: false,
        ..Settings::default()
    };

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

    let stack_rule_index = grammar.symbol_index("stack").unwrap();
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
    let mut data_writer = csv::Writer::from_writer(
        BufWriter::with_capacity(1024 * 512, data_file)
    );

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

        let scores = future::join_all(tasks).wait().unwrap();

        let best: f32 = scores
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
            .0;

        let avg: f32 = scores.iter().map(|f| f.0).sum::<f32>() / scores.len() as f32;

        let score = avg;

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
                    average: avg,
                    max: best,
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
                    average: avg,
                    max: best,
                })
                .expect("Could not write to data file");

            results.push((population_size, num_generations, score));
        }
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

/// Settings for GE simulation
#[derive(Debug)]
struct Settings {
    population_size: usize,
    mutation_rate: f32,
    max_iterations: u64,
    crossover: bool,
    mutate: bool,
    selection_size: usize,
    tournament_size: usize,
    /// Print output while running
    print: bool,
    /// Dump stats and the resulting model
    dump: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            population_size: 100,
            max_iterations: 100,
            selection_size: 16,
            tournament_size: 50,
            crossover: true,
            mutate: true,
            mutation_rate: 0.1,
            print: false,
            dump: false,
        }
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
        settings.mutation_rate,
        Vec::new(),
    );

    let mut population = (0..settings.population_size)
        .map(|_| {
            let seed = random_seed();
            let chromosome = generate_chromosome(&mut XorShiftRng::from_seed(seed), CHROMOSOME_LEN);
            base_phenotype.clone_with_chromosome(chromosome)
        })
        .collect();

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

    let best = {
        let mut builder = Simulator::builder(&mut population)
            .set_selector(Box::new(TournamentSelector::new(
                settings.selection_size,
                settings.tournament_size,
            )))
            .set_max_iters(settings.max_iterations)
            .crossover(settings.crossover)
            .mutate(settings.mutate);

        if settings.dump {
            builder = builder.set_step_callback(
                |iteration: u64, population: &[LsysPhenotype<GenePrimitive>]| {
                    let fitnesses: Vec<_> = population.iter().map(|p| p.fitness().0).collect();

                    let sum: f32 = fitnesses.iter().sum();
                    let average = sum / population.len() as f32;
                    let best = fitnesses
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();

                    let stats_csv = format!("{},{},{}\n", iteration, average, best);
                    stats_writer
                        .write_all(stats_csv.as_bytes())
                        .expect("Could not write to stats file");
                },
            );
        }

        let mut simulator = builder.build();

        simulator.run();
        simulator.get().unwrap().clone()
    };

    if settings.print {
        println!("Fitness: {}", best.fitness(),);
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
struct LsysPhenotype<'a, G: 'a> {
    grammar: &'a Grammar,
    distribution: &'a Distribution,
    stack_rule_index: usize,
    settings: &'a lsys::Settings,
    chromosome: Vec<G>,
    mutation_probability: f32,
    fitness: Cell<Option<LsysFitness>>,
}

impl<'a, G: Gene> LsysPhenotype<'a, G> {
    fn new(
        grammar: &'a Grammar,
        distribution: &'a Distribution,
        stack_rule_index: usize,
        settings: &'a lsys::Settings,
        mutation_probability: f32,
        chromosome: Vec<G>,
    ) -> Self {
        LsysPhenotype {
            grammar: grammar,
            distribution: distribution,
            stack_rule_index: stack_rule_index,
            settings: settings,
            chromosome: chromosome,
            fitness: Cell::new(None),
            mutation_probability: mutation_probability,
        }
    }

    fn clone_with_chromosome(&self, chromosome: Vec<G>) -> Self {
        LsysPhenotype::new(
            self.grammar,
            self.distribution,
            self.stack_rule_index,
            self.settings,
            self.mutation_probability,
            chromosome,
        )
    }
}

impl<'a, G: Gene + Clone> Phenotype<LsysFitness> for LsysPhenotype<'a, G> {
    fn fitness(&self) -> LsysFitness {
        if let Some(ref fitness) = self.fitness.get() {
            *fitness
        } else {
            let lsystem = generate_system(
                self.grammar,
                &mut WeightedChromosmeStrategy::new(
                    &self.chromosome,
                    self.distribution,
                    self.stack_rule_index,
                ),
            );
            let fitness = fitness::evaluate(&lsystem, self.settings);
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

    fn mutate(&self) -> Self {
        let mut rng = rand::thread_rng();

        if Range::new(0.0, 1.0).ind_sample(&mut rng) > self.mutation_probability {
            self.clone()
        } else {
            let mut chromosome = self.chromosome.clone();
            let mutation_index =
                Range::new(0, chromosome.len()).ind_sample(&mut rand::thread_rng());
            chromosome[mutation_index] =
                Range::new(G::min_value(), G::max_value()).ind_sample(&mut rng);

            self.clone_with_chromosome(chromosome)
        }
    }
}

type StepCallback<'a, T> = Box<FnMut(u64, &[T]) + 'a>;

/// A sequential implementation of `::sim::Simulation`.
/// The genetic algorithm is run in a single thread.
struct Simulator<'a, T, F>
where
    T: 'a + Phenotype<F>,
    F: Fitness,
{
    population: &'a mut Vec<T>,
    selector: Box<Selector<T, F>>,
    iteration_limit: u64,
    iteration: u64,
    step_callback: Option<StepCallback<'a, T>>,
    do_crossover: bool,
    mutate: bool,
}

impl<'a, T, F> Simulation<'a, T, F> for Simulator<'a, T, F>
where
    T: Phenotype<F>,
    F: Fitness,
{
    type B = SimulatorBuilder<'a, T, F>;

    /// Create builder.
    fn builder(population: &'a mut Vec<T>) -> SimulatorBuilder<'a, T, F> {
        SimulatorBuilder {
            sim: Simulator {
                population: population,
                selector: Box::new(MaximizeSelector::new(3)),
                iteration_limit: 100,
                iteration: 0,
                step_callback: None,
                do_crossover: true,
                mutate: true,
            },
        }
    }

    fn step(&mut self) -> StepResult {
        self.checked_step()
    }

    fn checked_step(&mut self) -> StepResult {
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

        let mut children: Vec<T> = {
            // Perform selection
            let parents = match self.selector.select(self.population) {
                Ok(parents) => parents,
                Err(e) => {
                    println!("Error selecting parents: {}", e);
                    return StepResult::Failure;
                }
            };

            let num_children = parents.len() / 2;
            let children: Vec<T> = if self.do_crossover && self.mutate {
                parents
                    .iter()
                    .take(num_children)
                    .zip(parents.iter().skip(num_children))
                    .map(|(a, b)| a.crossover(b))
                    .map(|c| c.mutate())
                    .collect()
            } else if self.do_crossover {
                parents
                    .iter()
                    .take(num_children)
                    .zip(parents.iter().skip(num_children))
                    .map(|(a, b)| a.crossover(b))
                    .collect()
            } else if self.mutate {
                parents
                    .into_iter()
                    .take(num_children)
                    .map(|c| c.mutate())
                    .collect()
            } else {
                parents.into_iter().take(num_children).cloned().collect()
            };

            children
        };

        // Kill off parts of the population at random to make room for the children
        self.kill_off(children.len());
        self.population.append(&mut children);

        self.iteration += 1;

        if let Some(ref mut callback) = self.step_callback {
            callback(self.iteration, self.population);
        }

        StepResult::Success // Not done yet, but successful
    }

    fn run(&mut self) -> RunResult {
        // Loop until Failure or Done.
        loop {
            match self.checked_step() {
                StepResult::Success => {}
                StepResult::Failure => return RunResult::Failure,
                StepResult::Done => return RunResult::Done,
            }
        }
    }

    fn get(&'a self) -> SimResult<'a, T> {
        Ok(self.population.iter().max_by_key(|x| x.fitness()).unwrap())
    }

    fn iterations(&self) -> u64 {
        self.iteration
    }

    fn time(&self) -> Option<NanoSecond> {
        unimplemented!()
    }

    fn population(&self) -> Vec<T> {
        unimplemented!()
    }
}

impl<'a, T, F> Simulator<'a, T, F>
where
    T: Phenotype<F>,
    F: Fitness,
{
    /// Kill off phenotypes using stochastic universal sampling.
    fn kill_off(&mut self, count: usize) {
        let ratio = self.population.len() / count;
        let mut i = thread_rng().gen_range::<usize>(0, self.population.len());
        for _ in 0..count {
            self.population.swap_remove(i);
            i = (i + ratio) % self.population.len();
        }
    }
}

impl<'a, T, F> fmt::Debug for Simulator<'a, T, F>
where
    T: Phenotype<F> + fmt::Debug,
    F: Fitness,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Simulator")
            .field("population", self.population)
            .field("selector", &self.selector)
            .field("iteration_limit", &self.iteration_limit)
            .field("iteration", &self.iteration)
            .field("step_callback", &self.step_callback.is_some())
            .field("do_crossover", &self.do_crossover)
            .field("mutate", &self.mutate)
            .finish()
    }
}

/// A `Builder` for the `Simulator` type.
struct SimulatorBuilder<'a, T, F>
where
    T: 'a + Phenotype<F>,
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

    fn crossover(mut self, crossover: bool) -> Self {
        self.sim.do_crossover = crossover;
        self
    }

    fn mutate(mut self, mutate: bool) -> Self {
        self.sim.mutate = mutate;
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
