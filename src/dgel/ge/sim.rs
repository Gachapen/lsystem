// Copyright 2015-2017 The RsGenetic Developers
// Copyright 2017 Magnus Bjerke Vik
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use rayon::prelude::*;
use rsgenetic::pheno::{Fitness, Phenotype};
use rsgenetic::sim::{Builder, NanoSecond, RunResult, SimResult, Simulation, StepResult};
use rsgenetic::sim::select::{MaximizeSelector, Selector};
use std::fmt::{self, Debug, Formatter};

type StepCallback<'a, T> = Box<FnMut(u64, &[T]) + 'a>;
type Operator<'a, T> = Box<FnMut(Vec<T>) -> Vec<T> + 'a>;

/// A genetic algorithm implementation, designed after "A Comparison of Selection Schemes used
/// in Genetic Algorithms", Tobias Blickle and Lothar Thiele, 1995. It is a modification of
/// `rsgenetic::sim::seq::Simulator`.
///
/// By default it has no genetic operators (`operators`). To add one, use
/// `SimulatorBuilder::chain_operator`.
pub struct Simulator<'a, T, F> {
    population: Vec<T>,
    selector: Box<Selector<T, F>>,
    iteration_limit: u64,
    iteration: u64,
    step_callback: Option<StepCallback<'a, T>>,
    operators: Vec<Operator<'a, T>>,
}

impl<'a, T, F> Simulation<'a, T, F> for Simulator<'a, T, F>
where
    T: Phenotype<F> + Clone + Sync,
    F: Fitness + Send,
{
    type B = SimulatorBuilder<'a, T, F>;

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
        Ok(
            self.population
                .par_iter()
                .max_by_key(|x| x.fitness())
                .unwrap(),
        )
    }

    fn time(&self) -> Option<NanoSecond> {
        None
    }

    fn iterations(&self) -> u64 {
        self.iteration
    }

    fn population(self) -> Vec<T> {
        self.population
    }
}

impl<'a, T, F> Debug for Simulator<'a, T, F>
where
    T: Phenotype<F> + Debug,
    F: Fitness,
{
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
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
pub struct SimulatorBuilder<'a, T, F>
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
    pub fn set_selector(mut self, sel: Box<Selector<T, F>>) -> Self {
        self.sim.selector = sel;
        self
    }

    pub fn set_max_iters(mut self, i: u64) -> Self {
        self.sim.iteration_limit = i;
        self
    }

    pub fn set_step_callback<C>(mut self, callback: C) -> Self
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
    pub fn chain_operator<O>(mut self, operator: O) -> Self
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
