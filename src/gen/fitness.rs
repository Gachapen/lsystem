use std::f32;
use std::f32::consts::{PI, FRAC_PI_2};
use std::fmt;

use na::{self, Unit, UnitQuaternion, Point2, Point3, Vector2, Vector3, Translation3, Rotation3};
use ncu;
use kiss3d::scene::SceneNode;
use yobun::partial_clamp;

use lsys::{self, ol, Command};
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

pub fn build_skeleton(
    instructions: ol::InstructionsIter,
    settings: &lsys::Settings,
    size_limit: usize,
    instruction_limit: usize,
) -> Option<Skeleton> {
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
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        segment_length
                    }
                };

                if !filling {
                    let direction = rotation * Vector3::new(0.0, 0.0, -1.0);
                    position += direction * segment_length;

                    let index = skeleton.points.len();
                    skeleton.points.push(position);
                    skeleton.edges.push(Vec::new());

                    skeleton.edges[parent].push(index);
                    parent = index;
                }
            }
            Command::YawRight => {
                let angle = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation *= Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * -angle);
            }
            Command::YawLeft => {
                let angle = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation *= Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * angle);
            }
            Command::UTurn => {
                let angle = PI;
                rotation *= Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * angle);
            }
            Command::PitchUp => {
                let angle = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation *= Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * angle);
            }
            Command::PitchDown => {
                let angle = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation *= Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * -angle);
            }
            Command::RollRight => {
                let angle = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation *= Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * -angle);
            }
            Command::RollLeft => {
                let angle = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation *= Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * angle);
            }
            Command::Shrink | Command::Grow | Command::Width | Command::NextColor |
            Command::Noop => {}
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
        };
    }

    Some(skeleton)
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
    let skeleton = build_skeleton(
        instruction_iter,
        settings,
        SKELETON_LIMIT,
        INSTRUCTION_LIMIT,
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

#[derive(Debug)]
pub struct Fitness {
    pub balance: f32,
    pub branching: f32,
    pub closeness: f32,
    pub drop: f32,
    pub is_nothing: bool,
}

impl Fitness {
    pub fn nothing() -> Fitness {
        Fitness {
            balance: 0.0,
            branching: 0.0,
            closeness: 0.0,
            drop: 0.0,
            is_nothing: true,
        }
    }

    pub fn reward(&self) -> f32 {
        let branching_reward = partial_max(self.branching, 0.0).expect("Brancing is NaN");
        let balance_reward = partial_max(self.balance, 0.0).expect("Balance is NaN");
        (balance_reward + branching_reward) / 2.0
    }

    pub fn nothing_punishment(&self) -> f32 {
        if self.is_nothing { 1.0 } else { 0.0 }
    }

    pub fn punishment(&self) -> f32 {
        let branching_punishment = partial_max(-self.branching, 0.0).expect("Branching is NaN");
        let balance_punishment = partial_max(-self.balance, 0.0).expect("Balance is NaN");
        (balance_punishment + self.drop + branching_punishment + self.closeness) / 4.0 +
            self.nothing_punishment()
    }

    pub fn score(&self) -> f32 {
        self.reward() - self.punishment()
    }
}

impl fmt::Display for Fitness {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.score())?;

        if self.is_nothing {
            write!(f, " (nothing)")
        } else {
            write!(
                f,
                " (bl: {}, br: {}, cl: {}, dr: {})",
                self.balance,
                self.branching,
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
    let skeleton = build_skeleton(
        instruction_iter,
        settings,
        SKELETON_LIMIT,
        INSTRUCTION_LIMIT,
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

        let fit = Fitness {
            balance: balance.fitness,
            branching: branching,
            drop: drop_fitness,
            closeness: closeness,
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
    center_imbalance.set_local_translation(Translation3::new(
        properties.center_spread / 4.0,
        0.0,
        0.0,
    ));

    let mut center_spread = balance.add_cube(properties.center_spread, LINE_WIDTH, LINE_WIDTH);
    center_spread.set_local_translation(Translation3::new(
        properties.center_spread / 2.0,
        0.0,
        0.0,
    ));

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
