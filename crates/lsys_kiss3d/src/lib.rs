extern crate kiss3d;
extern crate nalgebra as na;
extern crate ncollide_transformation as nct;
extern crate rand;
extern crate time;

extern crate lsys;

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
use std::borrow::Borrow;
use std::collections::HashMap;

use na::{Isometry3, Point3, Rotation3, Translation3, UnitQuaternion, Vector3};
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;
use kiss3d::camera::Camera;
use rand::{SeedableRng, XorShiftRng};
use rand::distributions::{IndependentSample, Range};

use lsys::{param, Command, Skeleton, SkeletonBuilder};

pub fn build_model<I>(instructions: I, settings: &lsys::Settings) -> SceneNode
where
    I: IntoIterator,
    I::Item: Borrow<lsys::Instruction>,
{
    let mut model = SceneNode::new_empty();

    let segment_length = settings.step;

    let mut position = Point3::new(0.0, 0.0, 0.0);
    let mut width = settings.width;
    let mut color_index = 0;
    let mut states = Vec::<(Point3<f32>, UnitQuaternion<f32>, f32, usize)>::new();

    let mut filling = false;
    let mut surface_points = Vec::new();

    let mut rotation = UnitQuaternion::from_euler_angles(FRAC_PI_2, 0.0, 0.0);

    for instruction in instructions {
        let instruction = instruction.borrow();
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
                    let mut segment = model.add_cube(1.0 * width, 1.0 * width, segment_length);
                    segment.append_translation(&Translation3::from_vector(
                        Vector3::new(0.0, 0.0, -segment_length / 2.0),
                    ));
                    segment.append_transformation(&Isometry3::from_parts(
                        Translation3::new(position.x, position.y, position.z),
                        rotation,
                    ));

                    assert!(
                        color_index < settings.colors.len(),
                        "Color index is outside color palette"
                    );
                    let color = settings.colors[color_index];
                    segment.set_color(color.0, color.1, color.2);

                    let direction = rotation * Vector3::new(0.0, 0.0, -1.0);
                    position = position + (direction * segment_length);
                } else {
                    let direction = rotation * Vector3::new(0.0, 0.0, -1.0);
                    position = position + (direction * segment_length);

                    surface_points.push(position);
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
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * -angle);
            }
            Command::YawLeft => {
                let angle = {
                    if let Some(ref args) = instruction.args {
                        args[0]
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
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * angle);
            }
            Command::PitchDown => {
                let angle = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * -angle);
            }
            Command::RollRight => {
                let angle = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * -angle);
            }
            Command::RollLeft => {
                let angle = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * angle);
            }
            Command::Shrink => {
                let rate = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.shrink_rate
                    }
                };
                width /= rate;
            }
            Command::Grow => {
                let rate = {
                    if let Some(ref args) = instruction.args {
                        args[0]
                    } else {
                        settings.shrink_rate
                    }
                };
                width *= rate;
            }
            Command::Width => {
                width = instruction.args.as_ref().unwrap()[0];
            }
            Command::Push => {
                states.push((position, rotation, width, color_index));
            }
            Command::Pop => {
                if let Some((stored_position, stored_rotation, stored_width, stored_color_index)) =
                    states.pop()
                {
                    position = stored_position;
                    rotation = stored_rotation;
                    width = stored_width;
                    color_index = stored_color_index;
                } else {
                    panic!("Tried to pop empty state stack");
                }
            }
            Command::BeginSurface => {
                filling = true;

                states.push((position, rotation, width, color_index));
                position = Point3::origin();
                rotation = UnitQuaternion::identity();
                width = settings.width;

                surface_points.push(position);
            }
            Command::EndSurface => {
                if let Some((stored_position, stored_rotation, stored_width, stored_color_index)) =
                    states.pop()
                {
                    position = stored_position;
                    rotation = stored_rotation;
                    width = stored_width;
                    color_index = stored_color_index;
                } else {
                    panic!("Tried to pop empty state stack");
                }

                if surface_points.len() >= 3 {
                    surface_points = surface_points
                        .iter()
                        .map(|p| Point3::new(p.x, p.z, 0.0))
                        .collect();

                    let mesh = nct::triangulate(&surface_points);
                    let mut node = model.add_trimesh(mesh, Vector3::new(1.0, 1.0, 1.0));

                    let surface_rot =
                        rotation * UnitQuaternion::from_euler_angles(FRAC_PI_2, 0.0, 0.0);

                    node.enable_backface_culling(false);
                    node.append_transformation(&Isometry3::from_parts(
                        Translation3::new(position.x, position.y, position.z),
                        surface_rot,
                    ));

                    assert!(
                        color_index < settings.colors.len(),
                        "Color index is outside color palette"
                    );
                    let color = settings.colors[color_index];
                    node.set_color(color.0, color.1, color.2);
                }

                surface_points.clear();
                filling = false;
            }
            Command::NextColor => {
                color_index += 1;
            }
            Command::Noop => {}
        };
    }

    model
}

pub struct LeafPlacement {
    pub position: Point3<f32>,
    pub orientation: UnitQuaternion<f32>,
    pub scale: f32,
}

const STARTING_WIDTH: f32 = 1.0;
const WIDTH_INCR: f32 = 1.0;

pub fn place_leaves(skeleton: &Skeleton) -> Vec<LeafPlacement>
{
    const LEAF_WIDTH_LIMIT: f32 = 20.0;

    fn segment_orientation(from: Point3<f32>, to: Point3<f32>) -> UnitQuaternion<f32> {
        let direction = na::normalize(&(to - from));

        let rotation = UnitQuaternion::rotation_between(
            &Vector3::new(0.0, 0.0, -1.0),
            &Vector3::new(direction.x, direction.y, direction.z),
        );

        match rotation {
            Some(rot) => rot,
            None => {
                println!(
                    "WARNING: Could not calulate necessary segment rotation, using hack instead..."
                );
                // Getting `None` seems to happen when directions are pointing away from each other
                // (not confirmed), so we just manually rotate it all the way around.
                UnitQuaternion::from_euler_angles(0.0, PI, 0.0)
            }
        }
    }

    let mut rng = XorShiftRng::from_seed([2170436650, 448509823, 3575179593, 3066426285]);
    let angle_range = Range::new(-PI, PI);
    let mut placements = Vec::new();

    let mut add_leaf =
        |placements: &mut Vec<LeafPlacement>, at: Point3<f32>, orientation: UnitQuaternion<f32>, width: f32| {
            let width = (width * 5.0).log10() / 3.0;

            let pitch = FRAC_PI_4 + angle_range.ind_sample(&mut rng) * 0.05;
            let pitch_rotation = UnitQuaternion::from_euler_angles(pitch, 0.0, 0.0);

            let yaw = angle_range.ind_sample(&mut rng);
            let yaw_rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, yaw);

            let rotation = orientation * yaw_rotation * pitch_rotation;

            placements.push(LeafPlacement {
                position: at,
                orientation: rotation,
                scale: width,
            });
        };

    let leaves = skeleton.find_leaves();

    let mut branchings = HashMap::<usize, (usize, f32)>::new();
    let mut visit_stack: Vec<_> = leaves
        .into_iter()
        .map(|index| (index, STARTING_WIDTH))
        .collect();

    while let Some((index, mut width)) = visit_stack.pop() {
        let mut parent = skeleton.parent_map[index];

        let from = skeleton.points[parent];
        let to = skeleton.points[index];
        let orientation = segment_orientation(from, to);

        if width <= LEAF_WIDTH_LIMIT {
            add_leaf(&mut placements, to, orientation, width);
        }

        while parent != 0 && skeleton.children_map[parent].len() == 1 {
            let grandparent = skeleton.parent_map[parent];

            let from = skeleton.points[grandparent];
            let to = skeleton.points[parent];
            let orientation = segment_orientation(from, to);
            width += WIDTH_INCR;

            if width <= LEAF_WIDTH_LIMIT {
                add_leaf(&mut placements, to, orientation, width);
            }

            parent = grandparent;
        }

        if parent != 0 {
            let (ref mut remaining_branches, ref mut largest_width) = *branchings
                .entry(parent)
                .or_insert_with(|| (skeleton.children_map[parent].len(), width));
            assert!(*remaining_branches > 0);

            *remaining_branches -= 1;
            *largest_width = largest_width.max(width);

            if *remaining_branches == 0 {
                visit_stack.push((parent, *largest_width + WIDTH_INCR));
            }
        }
    }

    placements
}

pub fn build_heuristic_model<I>(instructions: I, settings: &lsys::Settings) -> SceneNode
where
    I: IntoIterator,
    I::IntoIter: Iterator + SkeletonBuilder,
    I::Item: Borrow<lsys::Instruction>,
{
    fn segment_orientation(from: Point3<f32>, to: Point3<f32>) -> UnitQuaternion<f32> {
        let direction = na::normalize(&(to - from));

        let rotation = UnitQuaternion::rotation_between(
            &Vector3::new(0.0, 0.0, -1.0),
            &Vector3::new(direction.x, direction.y, direction.z),
        );

        match rotation {
            Some(rot) => rot,
            None => {
                println!(
                    "WARNING: Could not calulate necessary segment rotation, using hack instead..."
                );
                // Getting `None` seems to happen when directions are pointing away from each other
                // (not confirmed), so we just manually rotate it all the way around.
                UnitQuaternion::from_euler_angles(0.0, PI, 0.0)
            }
        }
    }

    fn add_segment(
        model: &mut SceneNode,
        at: Point3<f32>,
        orientation: UnitQuaternion<f32>,
        length: f32,
        width: f32,
    ) {
        // let width = (1.0 + width * 0.25).log10() / 10.0;
        let width = 0.002 + width * 0.002;
        let mut segment = model.add_cube(1.0 * width, 1.0 * width, length);
        segment.append_translation(&Translation3::from_vector(
            Vector3::new(0.0, 0.0, -length / 2.0),
        ));

        segment.append_transformation(&Isometry3::from_parts(
            Translation3::new(at.x, at.y, at.z),
            orientation,
        ));

        segment.set_color(0.757, 0.604, 0.420);
    }

    let skeleton = match instructions.into_iter().build_skeleton(settings) {
        None => return SceneNode::new_empty(),
        Some(skeleton) => skeleton,
    };

    let leaf_placements = place_leaves(&skeleton);

    // This points are the result of the "['{+f-f-f+|+f-f}]" grammar
    let leaf_points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.07653669, -0.1847759, 0.0),
        Point3::new(0.07653669, -0.3847759, 0.0),
        Point3::new(0.0, -0.5695518, 0.0),
        Point3::new(-0.07653669, -0.3847759, 0.0),
        Point3::new(-0.07653669, -0.1847759, 0.0),
    ];
    let leaf_mesh = nct::triangulate(&leaf_points);

    let add_leaf =
        |model: &mut SceneNode, at: Point3<f32>, orientation: UnitQuaternion<f32>, width: f32| {
            let scale = Vector3::new(width, width, width);
            let mut node = model.add_trimesh(leaf_mesh.clone(), scale);
            node.enable_backface_culling(false);
            node.append_transformation(&Isometry3::from_parts(
                Translation3::new(at.x, at.y, at.z),
                orientation,
            ));
            node.set_color(0.3, 1.0, 0.2);
        };

    let leaf_branches = skeleton.find_leaves();

    let mut model = SceneNode::new_empty();
    let mut branchings = HashMap::<usize, (usize, f32)>::new();
    let mut visit_stack: Vec<_> = leaf_branches
        .into_iter()
        .map(|index| (index, STARTING_WIDTH))
        .collect();

    while let Some((index, mut width)) = visit_stack.pop() {
        let mut parent = skeleton.parent_map[index];

        let from = skeleton.points[parent];
        let to = skeleton.points[index];
        let length = na::distance(&from, &to);
        let orientation = segment_orientation(from, to);

        add_segment(&mut model, from, orientation, length, width);

        while parent != 0 && skeleton.children_map[parent].len() == 1 {
            let grandparent = skeleton.parent_map[parent];

            let from = skeleton.points[grandparent];
            let to = skeleton.points[parent];
            let length = na::distance(&from, &to);
            let orientation = segment_orientation(from, to);
            width += WIDTH_INCR;

            add_segment(&mut model, from, orientation, length, width);

            parent = grandparent;
        }

        if parent != 0 {
            let (ref mut remaining_branches, ref mut largest_width) = *branchings
                .entry(parent)
                .or_insert_with(|| (skeleton.children_map[parent].len(), width));
            assert!(*remaining_branches > 0);

            *remaining_branches -= 1;
            *largest_width = largest_width.max(width);

            if *remaining_branches == 0 {
                visit_stack.push((parent, *largest_width + WIDTH_INCR));
            }
        }
    }

    for placement in leaf_placements {
        add_leaf(&mut model, placement.position, placement.orientation, placement.scale);
    }

    model
}

#[allow(dead_code)]
pub fn run_static<T>(
    window: &mut Window,
    camera: &mut Camera,
    (system, settings): (T, lsys::Settings),
) where
    T: lsys::Rewriter,
{
    let instructions = system.instructions(settings.iterations, &settings.command_map);

    let mut model = build_model(instructions.iter().cloned(), &settings);
    window.scene_mut().add_child(model.clone());

    while window.render_with_camera(camera) {
        model.append_rotation(&UnitQuaternion::from_euler_angles(0.0f32, 0.004, 0.0));
    }
}

#[allow(dead_code)]
pub fn run_animated(
    window: &mut Window,
    camera: &mut Camera,
    (system, settings): (param::LSystem, lsys::Settings),
) {
    let mut model = SceneNode::new_empty();

    let mut word = system.axiom.clone();
    let mut time = time::precise_time_s();

    while window.render_with_camera(camera) {
        let prev_time = time;
        time = time::precise_time_s();
        let dt = time - prev_time;

        word = param::step(&word, &system.productions, dt as f32 * 0.3);
        let instructions = param::map_word_to_instructions(&word, &settings.command_map);

        model.unlink();
        model = build_model(instructions.iter().cloned(), &settings);
        window.scene_mut().add_child(model.clone());
    }
}
