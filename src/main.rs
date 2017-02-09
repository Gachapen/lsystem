extern crate kiss3d;
extern crate nalgebra as na;

#[macro_use]
extern crate lsys;
extern crate lsys_kiss3d as l3d;

use na::{Point3};
use kiss3d::window::Window;
use kiss3d::light::Light;
use kiss3d::camera::ArcBall;

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

    //l3d::run_static(&mut window, &mut camera, lsystems::make_bush());
    l3d::run_animated(&mut window, &mut camera, lsystems::make_anim_tree());
}
