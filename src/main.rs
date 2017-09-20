extern crate bincode;
extern crate chrono;
extern crate clap;
extern crate cpuprofiler;
extern crate crossbeam;
extern crate csv;
extern crate futures;
extern crate futures_cpupool;
extern crate glfw;
extern crate kiss3d;
extern crate nalgebra as na;
extern crate ncollide_utils as ncu;
extern crate num;
extern crate num_cpus;
extern crate rand;
extern crate rsgenetic;
#[macro_use]
extern crate serde_derive;
extern crate serde_yaml;

extern crate abnf;
#[macro_use]
extern crate lsys;
extern crate lsys_kiss3d as lsys3d;

#[cfg(not(test))]
extern crate yobun;

#[cfg(test)]
#[macro_use]
extern crate yobun;

use na::Point3;
use kiss3d::window::Window;
use kiss3d::light::Light;
use kiss3d::camera::ArcBall;
use clap::{App, SubCommand};

mod lsystems;
mod glp;
mod dgel;
mod fitness;

fn main() {
    let matches = App::new("lsystem")
        .version("0.0.1")
        .author("Magnus Bjerke Vik <mbvett@gmail.com>")
        .about("Various L-system generation and visualization experiments")
        .subcommand(
            SubCommand::with_name("static").about("Run visualization of static plant"),
        )
        .subcommand(
            SubCommand::with_name("animated").about("Run animated visualization of plant growth"),
        )
        .subcommand(
            SubCommand::with_name("generated").about("Run random generation of plant"),
        )
        .subcommand(dgel::get_subcommand())
        .get_matches();

    if matches.subcommand_matches("static").is_some() {
        let (mut window, mut camera) = setup_window();
        let (system, settings) = lsystems::make_bush();
        println!("{}", system);
        println!("Fitness: {}", fitness::evaluate(&system, &settings).0);
        lsys3d::run_static(&mut window, &mut camera, (system, settings));
    } else if matches.subcommand_matches("animated").is_some() {
        let (mut window, mut camera) = setup_window();
        lsys3d::run_animated(&mut window, &mut camera, lsystems::make_anim_tree());
    } else if matches.subcommand_matches("generated").is_some() {
        let (mut window, mut camera) = setup_window();
        glp::run_generated(&mut window, &mut camera);
    } else if let Some(matches) = matches.subcommand_matches(dgel::COMMAND_NAME) {
        dgel::run_dgel(matches);
    } else {
        println!("A subcommand must be specified. See help by passing -h.");
    }
}

fn setup_window() -> (Window, ArcBall) {
    let mut window = Window::new("lsystem");
    window.set_light(Light::Absolute(Point3::new(15.0, 40.0, 15.0)));
    window.set_background_color(135.0 / 255.0, 206.0 / 255.0, 250.0 / 255.0);
    window.set_framerate_limit(Some(60));

    let camera = {
        let eye = Point3::new(0.0, 0.0, 20.0);
        let at = na::origin();
        ArcBall::new(eye, at)
    };

    (window, camera)
}
