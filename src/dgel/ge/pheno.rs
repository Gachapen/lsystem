use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::sync::{Arc, Mutex};
use rand::{self, Rng};
use rand::distributions::{IndependentSample, Range};
use rsgenetic::pheno::{Fitness, Phenotype};

use lsys::{self, ol};

use fitness;
use dgel::{generate_chromosome, generate_system, Distribution, GenePrimitive, Grammar,
           WeightedChromosmeStrategy, CHROMOSOME_LEN};

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct LsysFitness(f32);

impl LsysFitness {
    pub fn as_f32(&self) -> f32 {
        self.0
    }
}

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

#[derive(Debug)]
pub struct LsysPhenotype<'a> {
    grammar: &'a Grammar,
    distribution: &'a Distribution,
    stack_rule_index: usize,
    settings: &'a lsys::Settings,
    chromosome: Vec<GenePrimitive>,
    lsystem: Mutex<Option<(Arc<ol::LSystem>, usize)>>,
    fitness: Mutex<Option<LsysFitness>>,
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
            lsystem: Mutex::new(None),
            fitness: Mutex::new(None),
        }
    }

    pub fn new_random<R: Rng>(
        grammar: &'a Grammar,
        distribution: &'a Distribution,
        stack_rule_index: usize,
        settings: &'a lsys::Settings,
        rng: &mut R,
    ) -> Self {
        LsysPhenotype {
            grammar: grammar,
            distribution: distribution,
            stack_rule_index: stack_rule_index,
            settings: settings,
            chromosome: generate_chromosome(rng, CHROMOSOME_LEN),
            lsystem: Mutex::new(None),
            fitness: Mutex::new(None),
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

    pub fn lsystem(&self) -> Arc<ol::LSystem> {
        let mut lsystem_lock = self.lsystem.lock().unwrap();
        if let Some((ref lsystem, _)) = *lsystem_lock {
            return Arc::clone(lsystem);
        }

        let (lsystem, genes_used) = self.generate_lsystem();
        let lsystem = Arc::new(lsystem);
        *lsystem_lock = Some((Arc::clone(&lsystem), genes_used));

        lsystem
    }

    pub fn genes_used(&self) -> usize {
        let mut lsystem_lock = self.lsystem.lock().unwrap();
        if let Some((_, genes_used)) = *lsystem_lock {
            return genes_used;
        }

        let (lsystem, genes_used) = self.generate_lsystem();
        let lsystem = Arc::new(lsystem);
        *lsystem_lock = Some((lsystem, genes_used));

        genes_used
    }

    /// How many introns (unused genes) there are in the chromosome.
    #[allow(dead_code)]
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
            *self.fitness.lock().unwrap() = None;
            *self.lsystem.lock().unwrap() = None;
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
        *self.fitness.lock().unwrap() = None;
        *self.lsystem.lock().unwrap() = None;

        self
    }

    pub fn chromosome(&self) -> &Vec<GenePrimitive> {
        &self.chromosome
    }
}

impl<'a> Phenotype<LsysFitness> for LsysPhenotype<'a> {
    fn fitness(&self) -> LsysFitness {
        let mut fitness_lock = self.fitness.lock().unwrap();
        if let Some(fitness) = *fitness_lock {
            return fitness;
        }

        let fitness = fitness::evaluate(&self.lsystem(), self.settings);
        let fitness = LsysFitness(fitness.0.score());
        *fitness_lock = Some(fitness);

        fitness
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

        let index_range = Range::new(0, self.chromosome.len());
        let mutation_index = index_range.ind_sample(&mut rng);

        let gene_range = Range::new(GenePrimitive::min_value(), GenePrimitive::max_value());
        let new_gene = gene_range.ind_sample(&mut rng);

        self.chromosome[mutation_index] = new_gene;
        *self.fitness.lock().unwrap() = None;
        *self.lsystem.lock().unwrap() = None;

        self
    }
}

impl<'a> Clone for LsysPhenotype<'a> {
    fn clone(&self) -> LsysPhenotype<'a> {
        LsysPhenotype {
            chromosome: self.chromosome.clone(),
            lsystem: Mutex::new(self.lsystem.lock().unwrap().clone()),
            fitness: Mutex::new(*self.fitness.lock().unwrap()),
            ..*self
        }
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
                0, // string len 1
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
            ],
        );

        let chromosome_len = phenotype.chromosome.len();
        let pruned = phenotype.prune();
        assert_eq!(pruned.chromosome.len(), chromosome_len - 4);
    }
}
