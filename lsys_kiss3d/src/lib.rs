extern crate kiss3d;
extern crate nalgebra as na;
extern crate ncollide_transformation as nct;
extern crate num_traits;
extern crate time;

extern crate lsys;

use std::{f32};

use na::{Vector3, Point3, Rotation3, Translate, BaseFloat, Origin};
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;
use kiss3d::camera::Camera;
use num_traits::identities::One;

use lsys::Command;
use lsys::param;

pub fn build_model(instructions: &Vec<lsys::Instruction>, settings: &lsys::Settings) -> SceneNode {
    let mut model = SceneNode::new_empty();

    let segment_length = 0.2;

    let mut position = Point3::new(0.0, 0.0, 0.0);
    let mut rotation = Rotation3::new(Vector3::new(0.0, 0.0, 0.0));
    let mut width = settings.width;
    let mut color_index = 0;
    let mut states = Vec::<(Point3<f32>, Rotation3<f32>, f32, usize)>::new();

    let mut filling = false;
    let mut surface_points = Vec::new();

    rotation = rotation * Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * f32::frac_pi_2());

    for instruction in instructions {
        let command = instruction.command;
        match command {
            Command::Forward => {
                let segment_length = {
                    if !instruction.args.is_empty() {
                       instruction.args[0]
                    } else {
                        segment_length
                    }
                };

                if !filling {
                    let mut segment = model.add_cube(1.0 * width, 1.0 * width, segment_length);
                    segment.append_translation(&Vector3::new(0.0, 0.0, -segment_length / 2.0));
                    segment.append_transformation(
                        &na::Isometry3 {
                            translation: position.to_vector(),
                            rotation: rotation,
                        }
                    );

                    let color = settings.colors[color_index];
                    segment.set_color(color.0, color.1, color.2);

                    let direction = na::rotate(&rotation, &Vector3::new(0.0, 0.0, -1.0));
                    position = (direction * segment_length).translate(&position);
                } else {
                    let direction = na::rotate(&rotation, &Vector3::new(0.0, 0.0, -1.0));
                    position = (direction * segment_length).translate(&position);

                    surface_points.push(position);
                }
            },
            Command::YawRight => {
                let angle = {
                    if !instruction.args.is_empty() {
                       instruction.args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * -angle);
            },
            Command::YawLeft => {
                let angle = {
                    if !instruction.args.is_empty() {
                       instruction.args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * angle);
            },
            Command::UTurn => {
                let angle = f32::pi();
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * -angle);
            },
            Command::PitchUp => {
                let angle = {
                    if !instruction.args.is_empty() {
                       instruction.args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * angle);
            },
            Command::PitchDown => {
                let angle = {
                    if !instruction.args.is_empty() {
                       instruction.args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * -angle);
            }
            Command::RollRight => {
                let angle = {
                    if !instruction.args.is_empty() {
                       instruction.args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * -angle);
            },
            Command::RollLeft => {
                let angle = {
                    if !instruction.args.is_empty() {
                       instruction.args[0]
                    } else {
                        settings.angle
                    }
                };
                rotation = rotation * Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * angle);
            },
            Command::Shrink => {
                let rate = {
                    if !instruction.args.is_empty() {
                       instruction.args[0]
                    } else {
                        settings.shrink_rate
                    }
                };
                width = width / rate;
            },
            Command::Grow => {
                let rate = {
                    if !instruction.args.is_empty() {
                       instruction.args[0]
                    } else {
                        settings.shrink_rate
                    }
                };
                width = width * rate;
            },
            Command::Width => {
                width = instruction.args[0];
            },
            Command::Push => {
                states.push((position, rotation, width, color_index));
            },
            Command::Pop => {
                if let Some((stored_position, stored_rotation, stored_width, stored_color_index)) = states.pop() {
                    position = stored_position;
                    rotation = stored_rotation;
                    width = stored_width;
                    color_index = stored_color_index;
                } else {
                    panic!("Tried to pop empty state stack");
                }
            },
            Command::BeginSurface => {
                filling = true;

                states.push((position, rotation, width, color_index));
                position = Point3::origin();
                rotation = Rotation3::new(Vector3::new(0.0, 0.0, 0.0));
                width = settings.width;

                surface_points.push(position);
            },
            Command::EndSurface => {
                surface_points = surface_points.iter().map(|p| Point3::new(p.x, p.z, 0.0)).collect();

                let mesh = nct::triangulate(&surface_points);
                let mut node = model.add_trimesh(mesh, Vector3::one());

                if let Some((stored_position, stored_rotation, stored_width, stored_color_index)) = states.pop() {
                    position = stored_position;
                    rotation = stored_rotation;
                    width = stored_width;
                    color_index = stored_color_index;
                } else {
                    panic!("Tried to pop empty state stack");
                }

                let surface_rot = rotation * Rotation3::new(Vector3::new(1.0, 0.0, 0.0) * f32::frac_pi_2());

                node.enable_backface_culling(false);
                node.append_transformation(
                    &na::Isometry3 {
                        translation: position.to_vector(),
                        rotation: surface_rot,
                    }
                );

                let color = settings.colors[color_index];
                node.set_color(color.0, color.1, color.2);

                surface_points.clear();
                filling = false;
            },
            Command::NextColor => {
                color_index += 1;
            },
            Command::Noop => {},
        };
    }

    model
}

#[allow(dead_code)]
pub fn run_static<T>(window: &mut Window, camera: &mut Camera, (system, settings): (T, lsys::Settings))
    where T: lsys::Rewriter
{
    let instructions = system.instructions(settings.iterations);

    let mut model = build_model(&instructions, &settings);
    window.scene_mut().add_child(model.clone());

    while window.render_with_camera(camera) {
        model.append_rotation(&Vector3::new(0.0f32, 0.004, 0.0));
    }
}

#[allow(dead_code)]
pub fn run_animated(window: &mut Window, camera: &mut Camera, (system, settings): (param::LSystem, lsys::Settings))
{
    let mut model = SceneNode::new_empty();

    let mut word = system.axiom.clone();
    let mut time = time::precise_time_s();

    while window.render_with_camera(camera) {
        let prev_time = time;
        time = time::precise_time_s();
        let dt = time - prev_time;

        word = param::step(&word, &system.productions, dt as f32 * 0.3);
        let instructions = param::map_word_to_instructions(&word, &system.command_map);

        model.unlink();
        model = build_model(&instructions, &settings);
        window.scene_mut().add_child(model.clone());
    }
}
