extern crate kiss3d;
extern crate nalgebra as na;
extern crate time;

#[macro_use]
extern crate lsys;
extern crate lsys_kiss3d as l3d;

use std::f32;

use na::{Vector3, Point3};
use kiss3d::window::Window;
use kiss3d::light::Light;
use kiss3d::camera::Camera;
use kiss3d::camera::ArcBall;
use kiss3d::scene::SceneNode;

use lsys::param;
mod lsystems;

fn main() {
    let mut window = Window::new("lsystem");
    window.set_light(Light::Absolute(Point3::new(15.0, 40.0, 15.0)));
    window.set_background_color(135.0/255.0, 206.0/255.0, 250.0/255.0);
    window.set_framerate_limit(Some(60));

    let mut camera = {
        let eye = Point3::new(0.0, 0.0, 20.0);
        let at = na::origin();
        ArcBall::new(eye, at)
    };

    run_static(&mut window, &mut camera);
    //run_animated(&mut window, &mut camera);
}

#[allow(dead_code)]
fn run_static(window: &mut Window, camera: &mut Camera) {
    let (system, settings) = lsystems::make_bush();

    let instructions = system.instructions(settings.iterations);

    let mut model = l3d::build_model(&instructions, &settings);
    window.scene_mut().add_child(model.clone());

    while window.render_with_camera(camera) {
        model.append_rotation(&Vector3::new(0.0f32, 0.004, 0.0));
    }
}

#[allow(dead_code)]
fn run_animated(window: &mut Window, camera: &mut Camera) {
    let (system, settings) = lsystems::make_anim_tree();

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
        model = l3d::build_model(&instructions, &settings);
        window.scene_mut().add_child(model.clone());
    }
}
