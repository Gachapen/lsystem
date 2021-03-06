use std::f32;
use std::f32::consts::{FRAC_PI_4, PI};
use std::fmt;

use na::{self, Point2, Point3, Translation3, Unit, UnitQuaternion, Vector2};
use ncu;
use kiss3d::scene::SceneNode;
use yobun::partial_clamp;

use lsys::{self, ol};
use lsys::{Skeleton, SkeletonBuilder};
use lsys3d;
use yobun::*;

pub fn is_nothing(lsystem: &ol::LSystem) -> bool {
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
    while let Some(predicate) = visit_stack.pop() {
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

#[derive(Debug)]
pub struct Properties {
    pub reach: f32,
    pub drop: f32,
    pub spread: f32,
    pub center: Point3<f32>,
    pub center_spread: f32,
    pub num_points: usize,
    pub complexity: f32,
}

const SKELETON_LIMIT: usize = 10_000;
const INSTRUCTION_LIMIT: usize = SKELETON_LIMIT * 50;
const RULE_LIMIT: usize = 1000;

fn within_limits(lsystem: &ol::LSystem) -> bool {
    if lsystem.axiom.len() > RULE_LIMIT {
        return false;
    }

    for rule in lsystem.productions.iter() {
        if rule.len() > RULE_LIMIT {
            return false;
        }
    }

    true
}

pub fn is_crap(lsystem: &ol::LSystem, settings: &lsys::Settings) -> bool {
    if is_nothing(lsystem) || !within_limits(lsystem) {
        return true;
    }

    let instruction_iter = lsystem.instructions_iter(settings.iterations, &settings.command_map);
    let skeleton = instruction_iter.build_skeleton_with_limits(
        settings,
        Some(SKELETON_LIMIT),
        Some(INSTRUCTION_LIMIT),
    );
    if let Some(skeleton) = skeleton {
        if skeleton.points.len() <= 1 {
            return true;
        }

        return false;
    } else {
        return true;
    }
}

const BALANCE_WEIGHT: f32 = 1.0;
const BRANCHING_WEIGHT: f32 = 1.0;
const CLOSENESS_WEIGHT: f32 = 1.0;
const DROP_WEIGHT: f32 = 1.0;
const FOLIAGE_WEIGHT: f32 = 1.5;
const CURVATURE_WEIGHT: f32 = 0.4;
const LENGTH_WEIGHT: f32 = 1.0;

#[derive(Debug)]
pub struct Fitness {
    pub balance: f32,
    pub branching: f32,
    pub closeness: f32,
    pub drop: f32,
    pub foliage: f32,
    pub curvature: f32,
    pub length: f32,
    pub is_nothing: bool,
}

impl Fitness {
    pub fn nothing() -> Fitness {
        Fitness {
            balance: 0.0,
            branching: 0.0,
            closeness: 0.0,
            drop: 0.0,
            foliage: 0.0,
            curvature: 0.0,
            length: 0.0,
            is_nothing: true,
        }
    }

    /// Amount of reward in range [0, 1], where 1 is the best.
    pub fn reward(&self) -> f32 {
        let branching_reward =
            partial_max(self.branching, 0.0).expect("Brancing is NaN") * BRANCHING_WEIGHT;
        let balance_reward =
            partial_max(self.balance, 0.0).expect("Balance is NaN") * BALANCE_WEIGHT;
        let foliage_reward = self.foliage * FOLIAGE_WEIGHT;
        let curvature_reward = self.curvature * CURVATURE_WEIGHT;
        let length_reward = self.length * LENGTH_WEIGHT;

        const TOTAL_WEIGHT: f32 =
            BRANCHING_WEIGHT + BALANCE_WEIGHT + FOLIAGE_WEIGHT + CURVATURE_WEIGHT + LENGTH_WEIGHT;
        if TOTAL_WEIGHT == 0.0 {
            0.0
        } else {
            const NORMALIZER: f32 = 1.0 / TOTAL_WEIGHT;
            (balance_reward + branching_reward + foliage_reward + curvature_reward + length_reward)
                * NORMALIZER
        }
    }

    /// Punisment of being nothing as either 0 or 1, where 1 is worst.
    pub fn nothing_punishment(&self) -> f32 {
        if self.is_nothing {
            1.0
        } else {
            0.0
        }
    }

    /// Amount of punishment in range [0, 1], where 1 is the worst.
    pub fn punishment(&self) -> f32 {
        let branching_punishment =
            partial_max(-self.branching, 0.0).expect("Branching is NaN") * BRANCHING_WEIGHT;
        let balance_punishment =
            partial_max(-self.balance, 0.0).expect("Balance is NaN") * BALANCE_WEIGHT;
        let drop_punishment = self.drop * DROP_WEIGHT;
        let closeness_punishment = self.closeness * CLOSENESS_WEIGHT;

        const TOTAL_WEIGHT: f32 =
            BRANCHING_WEIGHT + BALANCE_WEIGHT + DROP_WEIGHT + CLOSENESS_WEIGHT;
        if TOTAL_WEIGHT == 0.0 {
            0.0
        } else {
            const NORMALIZER: f32 = 1.0 / TOTAL_WEIGHT;
            (balance_punishment + drop_punishment + branching_punishment + closeness_punishment)
                * NORMALIZER + self.nothing_punishment()
        }
    }

    /// Combined reward and punisment in range [0, 1], where 1 is the best.
    pub fn score(&self) -> f32 {
        (self.reward() - self.punishment() + 1.0) * 0.5
    }
}

impl fmt::Display for Fitness {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.3}", self.score())?;

        if self.is_nothing {
            write!(f, " (nothing)")
        } else {
            write!(
                f,
                " (bl: {:.3}, br: {:.3}, fl: {:.3}, cu: {:.3}, le: {:.3}, cl: {:.3}, dr: {:.3})",
                self.balance,
                self.branching,
                self.foliage,
                self.curvature,
                self.length,
                self.closeness,
                self.drop
            )
        }
    }
}

pub fn evaluate(lsystem: &ol::LSystem, settings: &lsys::Settings) -> (Fitness, Option<Properties>) {
    if is_nothing(lsystem) || !within_limits(lsystem) {
        return (Fitness::nothing(), None);
    }

    let instruction_iter = lsystem.instructions_iter(settings.iterations, &settings.command_map);
    let skeleton = instruction_iter.build_skeleton_with_limits(
        settings,
        Some(SKELETON_LIMIT),
        Some(INSTRUCTION_LIMIT),
    );

    if let Some(skeleton) = skeleton {
        if skeleton.points.len() <= 1 {
            return (Fitness::nothing(), None);
        }

        let reach = skeleton
            .points
            .iter()
            .max_by(|a, b| {
                a.y.partial_cmp(&b.y).expect("Points can not be compared")
            })
            .expect("Can't evaluate skeleton with no points")
            .y;

        let (drop_fitness, drop) = evaluate_drop(&skeleton);
        let balance = evaluate_balance(&skeleton);
        let closeness = evaluate_closeness(&skeleton);
        let (branching, complexity) = evaluate_branching(&skeleton);
        let foliage = evaluate_foliage(&skeleton);
        let curvature = evaluate_curvature(&skeleton);
        let length = evaluate_length(&skeleton);

        let fit = Fitness {
            balance: balance.fitness,
            branching: branching,
            drop: drop_fitness,
            closeness: closeness,
            foliage: foliage,
            curvature: curvature,
            length: length,
            is_nothing: false,
        };

        let prop = Properties {
            reach: reach,
            drop: drop,
            spread: balance.spread,
            center: balance.center,
            center_spread: balance.center_spread,
            num_points: skeleton.points.len(),
            complexity: complexity,
        };

        (fit, Some(prop))
    } else {
        (Fitness::nothing(), None)
    }
}

fn evaluate_drop(skeleton: &Skeleton) -> (f32, f32) {
    let drop = skeleton
        .points
        .iter()
        .min_by(|a, b| {
            a.y.partial_cmp(&b.y).expect("Points can not be compared")
        })
        .expect("Can't evaluate skeleton with no points")
        .y;

    let clamped_drop = partial_clamp(drop, -1.0, 0.0).expect("Drop is NaN");
    let interpolated_drop = (clamped_drop * PI / 2.0).sin();
    (-interpolated_drop, interpolated_drop)
}

struct Balance {
    center: Point3<f32>,
    fitness: f32,
    spread: f32,
    center_spread: f32,
}

/// Evaluate the balance of the plant.
/// A plant with the center of gravity in the origin has a score of 1.
/// A plant with the center of gravity at the furthest plant point away from the origin has a score
/// of -1. The rest is linear.
fn evaluate_balance(skeleton: &Skeleton) -> Balance {
    let floor_points: Vec<_> = skeleton
        .points
        .iter()
        .map(|p| Point2::new(p.x, p.z))
        .collect();

    let spread = floor_points
        .iter()
        .map(|p| na::norm(&Vector2::new(p.x, p.y)))
        .max_by(|a, b| {
            a.partial_cmp(b).expect("Lengths can not be compared")
        })
        .expect("Can't evaluate skeleton with no points");

    let center = ncu::center(&skeleton.points);
    let floor_center = Vector2::new(center.x, center.z);
    let center_distance = na::norm(&Vector2::new(floor_center.x, floor_center.y));

    // TODO: Give good score to plants with center of gravity close to root.
    if center_distance > 0.0 {
        let center_direction = Unit::new_unchecked(floor_center / center_distance);
        let center_spread = floor_points
            .iter()
            .map(|p| Vector2::new(p.x, p.y))
            .map(|p| project_onto(&p, &center_direction))
            .max_by(|a, b| {
                a.partial_cmp(b).expect("Projections can not be compared")
            })
            .expect("Can't evaluate skeleton with no points");

        Balance {
            fitness: (0.5 - (center_distance / center_spread)) * 2.0,
            spread: spread,
            center: center,
            center_spread: center_spread,
        }
    } else {
        // Distance of 0 will result in NaN, so handle as special case.
        Balance {
            fitness: 0.0,
            spread: 0.0,
            center: center,
            center_spread: 0.0,
        }
    }
}

fn evaluate_closeness(skeleton: &Skeleton) -> f32 {
    skeleton
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i >= skeleton.children_map.len() {
                return 0.0;
            }

            let edges = skeleton.children_map[i].iter().map(|e| skeleton.points[*e]);

            let segments: Vec<_> = edges.map(|e| (e - p).normalize()).collect();
            let closeness = segments
                .iter()
                .enumerate()
                .map(|(a_i, a_s)| {
                    let mut closest = -1.0;
                    for (b_i, b_s) in segments.iter().enumerate() {
                        if b_i != a_i {
                            let dot = na::dot(a_s, b_s);
                            closest =
                                partial_max(dot, closest).expect("Closeness can not be compared");
                        }
                    }

                    const THRESHOLD: f32 = 0.9;
                    if closest < THRESHOLD {
                        0.0
                    } else {
                        (closest - THRESHOLD) * (1.0 / (1.0 - THRESHOLD))
                    }
                })
                .max_by(|a, b| {
                    a.partial_cmp(b).expect("Closeness can not be compared")
                });

            if let Some(closeness) = closeness {
                closeness
            } else {
                0.0
            }
        })
        .max_by(|a, b| {
            a.partial_cmp(b).expect("Closeness can not be compared")
        })
        .expect("Can't evaluate skeleton with no points")
}

fn branching_complexity(skeleton: &Skeleton) -> f32 {
    // First point is root, which we don't want to measure, so skip 1.
    let branching_counts = skeleton
        .points
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(i, _)| {
            if i >= skeleton.children_map.len() {
                return None;
            }

            if skeleton.children_map[i].is_empty() {
                return None;
            }

            Some(skeleton.children_map[i].len())
        })
        .collect::<Vec<_>>();

    let total_branching_count = branching_counts.iter().fold(0, |total, b| total + b);

    if !branching_counts.is_empty() {
        total_branching_count as f32 / branching_counts.len() as f32
    } else {
        0.0
    }
}

fn branching_fitness(complexity: f32) -> f32 {
    if complexity < 1.0 {
        // No branches.
        -1.0
    } else if complexity < 1.2 {
        interpolate_cos(-1.0, 0.0, (complexity - 1.0) / (1.2 - 1.0))
    } else if complexity < 2.0 {
        interpolate_cos(0.0, 1.0, (complexity - 1.2) / (2.0 - 1.2))
    } else if complexity < 3.0 {
        1.0
    } else if complexity < 7.0 {
        let t = (complexity - 3.0) / (7.0 - 3.0);
        interpolate_cos(1.0, -1.0, t)
    } else {
        // 7 or more branches.
        -1.0
    }
}

fn evaluate_branching(skeleton: &Skeleton) -> (f32, f32) {
    let complexity = branching_complexity(skeleton);
    (branching_fitness(complexity), complexity)
}

fn evaluate_foliage(skeleton: &Skeleton) -> f32 {
    let leaves = lsys3d::place_leaves(&skeleton);
    let num_leaves = leaves.len();

    // 0 leaves = 0 score, asymptotic towards 1.
    let steepness = 0.1;
    let x = num_leaves as f32 * steepness;
    x / (1.0 + x)
}

fn evaluate_curvature(skeleton: &Skeleton) -> f32 {
    let min_angles: Vec<f32> = skeleton
        .children_map
        .iter()
        .enumerate()
        .filter_map(|(parent, children)| {
            if children.len() == 0 {
                return None;
            }

            let grand_parent = skeleton.parent_map[parent];
            if grand_parent == parent {
                return None;
            }

            let parent_pos = skeleton.points[parent];

            let parent_direction =
                Unit::new_normalize(parent_pos - skeleton.points[grand_parent]).unwrap();

            let min_angle: f32 = children
                .iter()
                .map(|child| {
                    let direction =
                        Unit::new_normalize(skeleton.points[*child] - parent_pos).unwrap();
                    na::angle(&direction, &parent_direction)
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            Some(min_angle)
        })
        .collect();

    let avg_min_angle: f32 = min_angles.iter().sum::<f32>() / min_angles.len() as f32;

    const ANGLE_MIN: f32 = 0.0;
    const ANGLE_OPTIMUM: f32 = 0.24711092; // Angle found in a nice looking plant (~14 deg)
    const ANGLE_MAX: f32 = FRAC_PI_4;

    if avg_min_angle >= ANGLE_MIN && avg_min_angle < ANGLE_OPTIMUM {
        const ANGLE_RANGE: f32 = ANGLE_OPTIMUM - ANGLE_MIN;
        interpolate_cos(0.0, 1.0, (avg_min_angle - ANGLE_MIN) / ANGLE_RANGE)
    } else if avg_min_angle >= ANGLE_OPTIMUM && avg_min_angle < ANGLE_MAX {
        const ANGLE_RANGE: f32 = ANGLE_MAX - ANGLE_OPTIMUM;
        interpolate_cos(1.0, 0.0, (avg_min_angle - ANGLE_OPTIMUM) / ANGLE_RANGE)
    } else {
        ANGLE_MIN
    }
}

fn evaluate_length(skeleton: &Skeleton) -> f32 {
    let mut visit_stack = vec![(skeleton.root(), 0)];
    let mut longest = 0;
    while let Some((index, length)) = visit_stack.pop() {
        if length > longest {
            longest = length;
        }

        for child in &skeleton.children_map[index] {
            visit_stack.push((*child, length + 1));
        }
    }

    // 0 length = 0 score, asymptotic towards 1.
    let steepness = 0.5;
    let x = longest as f32 * steepness;
    x / (1.0 + x)
}

#[allow(dead_code)]
pub fn add_properties_rendering(node: &mut SceneNode, properties: &Properties) {
    const LINE_LEN: f32 = 1.0;
    const LINE_WIDTH: f32 = 0.02;

    let mut center = SceneNode::new_empty();
    center.add_cube(LINE_WIDTH, LINE_LEN, LINE_WIDTH);
    center.add_cube(LINE_LEN, LINE_WIDTH, LINE_WIDTH);
    center.add_cube(LINE_WIDTH, LINE_WIDTH, LINE_LEN);
    center.set_local_translation(Translation3::new(
        properties.center.x,
        properties.center.y,
        properties.center.z,
    ));
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

    let mut center_imbalance = balance.add_cube(
        properties.center_spread / 2.0,
        LINE_WIDTH * 1.1,
        LINE_WIDTH * 1.1,
    );
    center_imbalance.set_color(0.1, 0.8, 0.1);
    center_imbalance
        .set_local_translation(Translation3::new(properties.center_spread / 4.0, 0.0, 0.0));

    let mut center_spread = balance.add_cube(properties.center_spread, LINE_WIDTH, LINE_WIDTH);
    center_spread
        .set_local_translation(Translation3::new(properties.center_spread / 2.0, 0.0, 0.0));

    node.add_child(balance);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_branching_fitness() {
        assert_eq!(branching_fitness(0.0), -1.0);
        assert_eq!(branching_fitness(0.5), -1.0);
        assert_eq!(branching_fitness(1.0), -1.0);
        assert_eq!(branching_fitness(1.1), -0.5);
        assert_eq!(branching_fitness(1.2), 0.0);
        assert_eq!(branching_fitness(1.6), 0.5);
        assert_eq!(branching_fitness(2.0), 1.0);
        assert_eq!(branching_fitness(2.5), 1.0);
        assert_eq!(branching_fitness(3.0), 1.0);
        assert_eq!(branching_fitness(5.0), 0.0);
        assert_eq!(branching_fitness(7.0), -1.0);
        assert_eq!(branching_fitness(10.0), -1.0);
    }
}
