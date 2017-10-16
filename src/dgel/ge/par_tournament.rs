// file: tournament.rs
//
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

use rsgenetic::sim::select::Selector;
use rsgenetic::pheno::{Fitness, Phenotype};
use rand::{self, Rng};
use rayon::prelude::*;

/// Runs several tournaments, and selects best performing phenotypes from each tournament.
#[derive(Clone, Debug)]
pub struct ParTournamentSelector {
    participants: usize,
}

impl ParTournamentSelector {
    /// Create and return a tournament selector.
    ///
    /// Such a selector runs N tournaments (N = population size),
    /// each with `participants` participants.
    /// From each tournament, the best phenotype is selected, yielding N parents.
    ///
    /// * `participants`: must be larger than zero and less than the population size.
    pub fn new(participants: usize) -> ParTournamentSelector {
        ParTournamentSelector {
            participants: participants,
        }
    }
}

impl<T, F> Selector<T, F> for ParTournamentSelector
where
    T: Phenotype<F> + Sync,
    F: Fitness,
{
    fn select<'a>(&self, population: &'a [T]) -> Result<Vec<&'a T>, String> {
        if self.participants == 0 || self.participants >= population.len() {
            return Err(format!(
                "Invalid parameter `participants`: {}. Should be larger than \
                 zero and less than the population size.",
                self.participants
            ));
        }

        let selected: Vec<_> = (0..population.len())
            .into_par_iter()
            .map(|_| {
                let mut tournament: Vec<&T> = Vec::with_capacity(self.participants);
                let mut rng = rand::thread_rng();
                for _ in 0..self.participants {
                    let index = rng.gen_range::<usize>(0, population.len());
                    tournament.push(&population[index]);
                }
                *tournament
                    .iter()
                    .max_by(|x, y| x.fitness().cmp(&y.fitness()))
                    .unwrap()
            })
            .collect();

        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dgel::ge::test::Test;

    #[test]
    fn test_participants_zero() {
        let selector = ParTournamentSelector::new(0);
        let population: Vec<Test> = (0..100).map(|i| Test { f: i }).collect();
        assert!(selector.select(&population).is_err());
    }

    #[test]
    fn test_participants_too_large() {
        let selector = ParTournamentSelector::new(100);
        let population: Vec<Test> = (0..100).map(|i| Test { f: i }).collect();
        assert!(selector.select(&population).is_err());
    }

    #[test]
    fn test_result_size() {
        let selector = ParTournamentSelector::new(5);
        let population: Vec<Test> = (0..100).map(|i| Test { f: i }).collect();
        assert_eq!(100, selector.select(&population).unwrap().len());
    }
}
