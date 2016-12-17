extern crate kiss3d;
extern crate nalgebra as na;
extern crate ncollide_transformation as nct;
extern crate time;
extern crate num_traits;
extern crate rand;
extern crate strsim;
extern crate glfw;

#[macro_use]
extern crate lsys;

use std::rc::Rc;
use std::{f32, u32, cmp, mem, fs};

use na::{Vector3, Point3, Rotation3, Translate, BaseFloat, Origin};
use num_traits::identities::{One};
use kiss3d::window::Window;
use kiss3d::light::Light;
use kiss3d::camera::Camera;
use kiss3d::camera::ArcBall;
use kiss3d::scene::SceneNode;
use glfw::WindowEvent;
use glfw::Key;
use glfw::Action;
use rand::distributions::{IndependentSample, Range};

use lsys::Command;
use lsys::ol;
use lsys::il;
use lsys::param;
use lsys::param::Param;
use lsys::param::WordFromString;
use lsys::param::Param::{I,F};

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

    //run_static(&mut window, &mut camera);
    //run_animated(&mut window, &mut camera);
    run_experiment(&mut window, &mut camera);
}

fn run_static(window: &mut Window, camera: &mut Camera) {
    let (system, settings) = make_bush();

    let instructions = system.instructions(settings.iterations);

    let mut model = build_model(&instructions, &settings);
    window.scene_mut().add_child(model.clone());

    while window.render_with_camera(camera) {
        model.append_rotation(&Vector3::new(0.0f32, 0.004, 0.0));
    }
}

fn run_animated(window: &mut Window, camera: &mut Camera) {
    let (system, settings) = make_anim_tree();

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

fn run_experiment(window: &mut Window, camera: &mut Camera) {
    #[derive(Copy, Clone, Debug)]
    struct Point {
        medoid: usize,
        index: usize,
    };

    #[derive(Copy, Clone, Debug)]
    struct Medoid {
        index: usize,
    };

    struct Cluster {
        medoid: String,
        members: Vec<String>,
    };

    fn calculate_cost(costs: &Vec<Vec<f32>>, medoids: &Vec<Medoid>, points: &Vec<Point>) -> f32 {
        let mut sum = 0.0;
        for point in points {
            sum += costs[medoids[point.medoid].index][point.index];
        }
        sum
    }

    fn organize_clusters(rewrites: &Vec<String>, medoids: &Vec<Medoid>, points: &Vec<Point>) -> Vec<Cluster> {
        let mut clusters = vec![];

        for medoid in medoids {
            clusters.push(Cluster {
                medoid: rewrites[medoid.index].clone(),
                members: vec![],
            });
        }

        for point in points {
            clusters[point.medoid].members.push(rewrites[point.index].clone());
        }

        clusters
    }

    fn print_clusters(rewrites: &Vec<String>, medoids: &Vec<Medoid>, points: &Vec<Point>) {
        let clusters = organize_clusters(&rewrites, &medoids, &points);

        println!("Created {} clusters.", medoids.len());
        for (i, cluster) in clusters.iter().enumerate() {
            println!("Cluster {}: {}", i, cluster.medoid);
            for member in &cluster.members {
                println!("\t{}", member);
            }
        }
    }

    fn generate_rewrite() -> String {
        let mut rng = rand::thread_rng();
        let alphabet = vec!['X', 'F', 'X', 'F', '+', '-'];

        let mut generated = vec![];
        generated.push('F');
        for _ in 0..9 {
            let letter = rand::sample(&mut rng, alphabet.iter(), 1);
            generated.push(*letter[0]);
        }

        let num_branches_range = Range::new(1, 3);
        let num_branches = num_branches_range.ind_sample(&mut rng);
        let num_letters_range = Range::new(1, 4);

        for _ in 0..num_branches {
            let p = rand::sample(&mut rng, 0..generated.len(), 1)[0];

            let num_letters = num_letters_range.ind_sample(&mut rng);
            let mut branch = vec![];
            branch.push('[');
            for _ in 0..num_letters {
                let letter = rand::sample(&mut rng, alphabet.iter(), 1);
                branch.push(*letter[0]);
            }
            branch.push(']');
            branch.reverse();

            // Slow...
            for c in branch {
                generated.insert(p, c);
            }
        }

        let mut fixed = vec![];
        let mut depth = 0;
        for letter in generated {
            if letter == '[' {
                depth += 1;
                fixed.push(letter);
            } else if letter == ']' {
                if depth > 0 {
                    depth -= 1;
                    fixed.push(letter);
                }
            } else {
                fixed.push(letter);
            }
        }

        while depth > 0 {
            fixed.push(']');
            depth -= 1;
        }

        let mut rewrite = String::new();
        for letter in fixed {
            rewrite.push(letter);
        }

        rewrite
    }

    fn save_as_images(rewrites: &Vec<String>) {
        let mut sys = ol::LSystem::new();
        sys.axiom = "X".to_string();

        let settings = lsys::Settings {
            angle: f32::to_radians(22.5),
            width: 0.03,
            iterations: 5,
            ..lsys::Settings::new()
        };

        let mut window = Window::new_with_size("lsystem", 1000, 1000);
        window.set_light(Light::Absolute(Point3::new(15.0, 40.0, 15.0)));
        window.set_background_color(135.0/255.0, 206.0/255.0, 250.0/255.0);
        window.set_framerate_limit(Some(60));

        let mut camera = {
            let eye = Point3::new(0.0, 0.0, 20.0);
            let at = na::origin();
            ArcBall::new(eye, at)
        };

        let mut model = SceneNode::new_empty();

        fs::remove_dir_all("img");
        fs::create_dir("img");

        for (i, rewrite) in rewrites.iter().enumerate() {
            model.unlink();

            println!("Saving rewrite: {}", rewrite);
            sys.set_rule('X', &rewrite);

            let instructions = sys.instructions(settings.iterations);

            model = build_model(&instructions, &settings);
            model.prepend_to_local_translation(&Vector3::new(0.0, -4.0, 0.0));
            window.scene_mut().add_child(model.clone());
            window.render_with_camera(&mut camera);
            let image = window.snap_image();
            image.save(format!("img/{}_{}.png", i, rewrite));
        }
    }

    fn render_clusters(clusters: &Vec<Cluster>, window: &mut Window, camera: &mut Camera) {
        let mut cluster_index = 0;
        let mut member_index = 0;

        let mut sys = ol::LSystem::new();

        sys.axiom = "X".to_string();

        let settings = lsys::Settings {
            angle: f32::to_radians(25.7),
            width: 0.03,
            iterations: 5,
            ..lsys::Settings::new()
        };

        let mut medoid_model = SceneNode::new_empty();
        let mut member_model = SceneNode::new_empty();
        let mut medoid_changed = true;
        let mut member_changed = true;

        while window.render_with_camera(camera) {
            for mut event in window.events().iter() {
                match event.value {
                    WindowEvent::Key(Key::Right, _, Action::Release, _) => {
                        member_index = (member_index + 1) % clusters[cluster_index].members.len();
                        event.inhibited = true;
                        member_changed = true;
                    },
                    WindowEvent::Key(Key::Left, _, Action::Release, _) => {
                        if member_index > 0 {
                            member_index -= 1;
                        } else if clusters[cluster_index].members.len() > 0 {
                            member_index = clusters[cluster_index].members.len() - 1;
                        }
                        event.inhibited = true;
                        member_changed = true;
                    },
                    WindowEvent::Key(Key::Up, _, Action::Release, _) => {
                        cluster_index = (cluster_index + 1) % clusters.len();
                        member_index = 0;
                        event.inhibited = true;
                        medoid_changed = true;
                        member_changed = true;
                    },
                    WindowEvent::Key(Key::Down, _, Action::Release, _) => {
                        if cluster_index > 0 {
                            cluster_index -= 1;
                        } else {
                            cluster_index = clusters.len() - 1;
                        }
                        member_index = 0;
                        event.inhibited = true;
                        medoid_changed = true;
                        member_changed = true;
                    },
                    _ => {}
                }
            }

            if medoid_changed {
                medoid_model.unlink();

                println!("Showing cluster {} medoid: {}", cluster_index, clusters[cluster_index].medoid);
                sys.set_rule('X', &clusters[cluster_index].medoid);

                let instructions = sys.instructions(settings.iterations);

                medoid_model = build_model(&instructions, &settings);
                medoid_model.prepend_to_local_translation(&Vector3::new(0.0, 5.0, 0.0));
                window.scene_mut().add_child(medoid_model.clone());

                medoid_changed = false;
            }

            if member_changed {
                member_model.unlink();

                if clusters[cluster_index].members.len() > 0 {
                    println!("Showing member {}: {}", member_index, clusters[cluster_index].members[member_index]);
                    sys.set_rule('X', &clusters[cluster_index].members[member_index]);

                    let instructions = sys.instructions(settings.iterations);

                    member_model = build_model(&instructions, &settings);
                    member_model.prepend_to_local_translation(&Vector3::new(0.0, -5.0, 0.0));
                    window.scene_mut().add_child(member_model.clone());
                }

                member_changed = false;
            }
        }
    }

    fn cluster(rewrites: &Vec<String>) -> Vec<Cluster> {
        println!("Calculating costs");
        let mut costs = vec![vec![0.0f32; rewrites.len()]; rewrites.len()];
        for i in 0..rewrites.len() {
            for j in i+1..rewrites.len() {
                println!("Checking {} <-> {}", &rewrites[i], &rewrites[j]);
                let cost = strsim::damerau_levenshtein(&rewrites[i], &rewrites[j]) as f32;
                println!("Cost: {}", cost);
                costs[i][j] = cost;
                costs[j][i] = cost;
            }
        }

        println!("Assigning medoids");

        let mut rng = rand::thread_rng();
        let num_clusters = 4;

        // Assign medoids.
        let mut medoids = vec![];
        let indices = rand::sample(&mut rng, 0..rewrites.len(), num_clusters);
        for i in indices {
            medoids.push(Medoid{ index: i });
        }

        println!("Assigning points");

        // Assign points.
        let mut points = vec![];
        for i in 0..rewrites.len() {
            if let None = medoids.iter().find(|m| m.index == i) {
                points.push(Point{ medoid: 0, index: i });
            }
        }

        println!("Rewrites: {:?}", rewrites);
        println!("Points: {:?}", points);
        println!("Medoids: {:?}", medoids);

        println!("Optimizing");

        let mut total_cost = f32::MAX;
        let mut optimize = true;
        while optimize {
            println!("Clustering points");

            // Place points into medoids.
            for point in &mut points {
                let mut best_cost = f32::MAX;
                let mut best_medoid = 0;
                for m in 0..medoids.len() {
                    let ref medoid = medoids[m];
                    let cost = costs[point.index][medoid.index];
                    if cost < best_cost {
                        best_cost = cost;
                        best_medoid = m;
                    }
                }
                point.medoid = best_medoid;
            }

            print_clusters(&rewrites, &medoids, &points);

            println!("Optimizing medoids");
            let prev_cost = total_cost;

            // Optimize medoids.
            for m in 0..medoids.len() {
                for p in 0..points.len() {
                    mem::swap(&mut medoids[m].index, &mut points[p].index);
                    let cost = calculate_cost(&costs, &medoids, &points);
                    if cost >= total_cost {
                        mem::swap(&mut medoids[m].index, &mut points[p].index);
                    } else {
                        total_cost = cost;
                        println!("Cost improved to {}", cost);
                    }
                }
            }

            // Can't improve more.
            if total_cost == 0.0 || total_cost == prev_cost {
                optimize = false;
            }
        }

        println!("Done");

        print_clusters(&rewrites, &medoids, &points);
        organize_clusters(&rewrites, &medoids, &points)
    }

    let num_rewrites = 16;

    println!("Generating rewrites");

    let mut rewrites = vec![];
    for _ in 0..num_rewrites {
        let rewrite = generate_rewrite();
        rewrites.push(rewrite);
    }
    let rewrites = rewrites;

    //let clusters = cluster(&rewrites);
    //render_clusters(&clusters, window, camera);
    save_as_images(&rewrites);
}

fn build_model(instructions: &Vec<lsys::Instruction>, settings: &lsys::Settings) -> SceneNode {
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

//fn init_rules() -> Vec<Vec<Command>> {
//    let mut rules: Vec<Vec<Command>> = vec![vec![]; 17];
//    rules[Command::A as usize] = vec![Command::A];
//    rules[Command::Forward as usize] = vec![Command::Forward];
//    rules[Command::Backward as usize] = vec![Command::Backward];
//    rules[Command::YawRight as usize] = vec![Command::YawRight];
//    rules[Command::YawLeft as usize] = vec![Command::YawLeft];
//    rules[Command::PitchUp as usize] = vec![Command::PitchUp];
//    rules[Command::PitchDown as usize] = vec![Command::PitchDown];
//    rules[Command::RollRight as usize] = vec![Command::RollRight];
//    rules[Command::RollLeft as usize] = vec![Command::RollLeft];
//    rules[Command::Shrink as usize] = vec![Command::Shrink];
//    rules[Command::Grow as usize] = vec![Command::Grow];
//    rules[Command::Push as usize] = vec![Command::Push];
//    rules[Command::Pop as usize] = vec![Command::Pop];
//    rules[Command::A as usize] = vec![Command::A];
//    rules[Command::B as usize] = vec![Command::B];
//    rules[Command::C as usize] = vec![Command::C];
//    rules[Command::D as usize] = vec![Command::D];

//    rules
//}

//fn make_thing1() -> ol::LSystem {
//    let mut rules = init_rules();

//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawRight,
//        Command::YawRight,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward
//    ];

//    ol::LSystem {
//        axiom: vec![Command::Forward],
//        iterations: 4,
//        angle: f32::frac_pi_4(),
//        rules: rules,
//    }
//}

fn make_hilbert() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.rules['A' as usize] = String::from("B-F+CFC+F-D&F^D-F+&&CFC+F+B>>");
    system.rules['B' as usize] = String::from("A&F^CFB^F^D^^-F-D^++F^B++FC^F^A>>");
    system.rules['C' as usize] = String::from("++D^++F^-F+C^F^A&&FA&F^C+F+B^F^D>>");
    system.rules['D' as usize] = String::from("++CFB-F+B++FA&F^A&&FB-F+B++FC>>");

    system.axiom = String::from("A");

    let settings = lsys::Settings {
        angle: f32::to_radians(90.0),
        width: 0.01,
        iterations: 2,
        ..lsys::Settings::new()
    };

    (system, settings)
}

//fn make_koch1() -> ol::LSystem {
//    let mut rules = init_rules();

//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawRight,
//        Command::Forward,
//    ];

//    ol::LSystem {
//        axiom: vec![
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//        ],
//        iterations: 4,
//        angle: f32::frac_pi_2(),
//        rules: rules,
//    }
//}

//fn make_koch2() -> ol::LSystem {
//    let mut rules = init_rules();

//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//    ];

//    ol::LSystem {
//        axiom: vec![
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//        ],
//        iterations: 4,
//        angle: f32::frac_pi_2(),
//        rules: rules,
//    }
//}

//fn make_koch3() -> ol::LSystem {
//    let mut rules = init_rules();

//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//        Command::Forward,
//        Command::YawLeft,
//        Command::YawLeft,
//        Command::Forward,
//        Command::YawLeft,
//        Command::Forward,
//    ];

//    ol::LSystem {
//        axiom: vec![
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//            Command::YawLeft,
//            Command::Forward,
//        ],
//        iterations: 5,
//        angle: f32::frac_pi_2(),
//        rules: rules,
//    }
//}

fn make_plant1() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.command_map['X' as usize] = Command::Noop;

    system.rules['X' as usize] = String::from("F[+X][-X][&X][^X]FX");
    system.rules['F' as usize] = String::from("F!F!");

    system.axiom = String::from("X");

    let settings = lsys::Settings {
        angle: 0.4485496,
        width: 0.03,
        shrink_rate: 1.01,
        iterations: 6,
        ..lsys::Settings::new()
    };

    (system, settings)
}

//fn make_plant2() -> ol::LSystem {
//    let mut rules = init_rules();

//    rules[Command::A as usize] = vec![
//        Command::Forward,
//        Command::Shrink,
//        Command::YawLeft,
//        Command::Push,
//        Command::Push,
//        Command::A,
//        Command::Pop,
//        Command::YawRight,
//        Command::A,
//        Command::Pop,
//        Command::YawRight,
//        Command::Forward,
//        Command::Shrink,
//        Command::Push,
//        Command::YawRight,
//        Command::Forward,
//        Command::Shrink,
//        Command::A,
//        Command::Pop,
//        Command::PitchDown,
//        Command::Push,
//        Command::Push,
//        Command::A,
//        Command::Pop,
//        Command::PitchUp,
//        Command::A,
//        Command::Pop,
//        Command::PitchUp,
//        Command::Forward,
//        Command::Shrink,
//        Command::Push,
//        Command::PitchUp,
//        Command::Forward,
//        Command::Shrink,
//        Command::A,
//        Command::Pop,
//        Command::Push,
//        Command::PitchDown,
//        Command::Forward,
//        Command::Shrink,
//        Command::A,
//        Command::Pop,
//        Command::YawLeft,
//        Command::A,
//    ];
//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::Forward,
//    ];

//    ol::LSystem {
//        axiom: vec![Command::A],
//        iterations: 5,
//        angle: 0.3926991,
//        rules: rules,
//    }
//}

//fn make_wheat() -> ol::LSystem {
//    let mut rules = init_rules();

//    rules[Command::A as usize] = vec![
//        Command::Forward,
//        Command::Push,
//        Command::Forward,
//        Command::Pop,
//        Command::Push,
//        Command::YawLeft,
//        Command::Forward,
//        Command::Pop,
//        Command::Push,
//        Command::YawRight,
//        Command::Forward,
//        Command::Pop,
//        Command::A,
//    ];

//    ol::LSystem {
//        axiom: vec![Command::A],
//        iterations: 10,
//        angle: f32::frac_pi_4(),
//        rules: rules,
//    }
//}

//fn make_plant3() -> ol::LSystem {
//    let mut rules = init_rules();

//    rules[Command::A as usize] = vec![
//        Command::Forward,

//        Command::Push,
//        Command::A,
//        Command::Pop,

//        Command::Push,
//        Command::YawLeft,
//        Command::A,
//        Command::Pop,

//        Command::Forward,
//        Command::Push,
//        Command::YawRight,
//        Command::A,
//        Command::Pop,

//        Command::Push,
//        Command::PitchDown,
//        Command::A,
//        Command::Pop,

//        Command::Push,
//        Command::PitchUp,
//        Command::A,
//        Command::Pop,
//    ];

//    rules[Command::Forward as usize] = vec![
//        Command::Forward,
//        Command::Shrink,
//        Command::Forward,
//    ];

//    ol::LSystem {
//        axiom: vec![Command::A],
//        iterations: 6,
//        angle: f32::frac_pi_4(),
//        rules: rules,
//    }
//}

fn make_gosper_hexa() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    system.command_map['l' as usize] = Command::Forward;
    system.command_map['r' as usize] = Command::Forward;

    system.rules['l' as usize] = String::from("l+r++r-l--ll-r+");
    system.rules['r' as usize] = String::from("-l+rr++r+l--l-r");

    system.axiom = String::from("l");

    let settings = lsys::Settings {
        angle: f32::to_radians(60.0),
        width: 0.02,
        iterations: 4,
        ..lsys::Settings::new()
    };

    (system, settings)
}

fn make_2012xuequiang() -> (ol::LSystem, lsys::Settings) {
    let mut system = ol::LSystem::new();

    // 3D
    //system.set_rule('X', "F&[[^F^Y]&F&Y]^[[&F&Y]^F^Y]-[[+F+Y]-F-Y]+F[+FX+Y]-X");
    //system.set_rule('Y', "F[&Y]F[^Y]F[+Y]F[-Y]+Y");

    // 2D
    system.set_rule('X', "F-[[+F+Y]-F-Y]+F[+FX+Y]-X");
    system.set_rule('Y', "F[+Y]F[-Y]+Y");
    system.set_rule('F', "FF");

    system.axiom = String::from("X");

    let settings = lsys::Settings {
        angle: f32::to_radians(23.5),
        width: 0.02,
        iterations: 5,
        ..lsys::Settings::new()
    };

    (system, settings)
}

fn make_hogeweg_b() -> (il::LSystem, lsys::Settings) {
    let mut sys = il::LSystem::new();

    sys.axiom = "F1F1F1".to_string();
    sys.ignore_from_context("+-F");
    sys.productions = vec![
        il::Production::with_context('0', '0', '0', "1"),
        il::Production::with_context('0', '0', '1', "1[-F1F1]"),
        il::Production::with_context('0', '1', '0', "1"),
        il::Production::with_context('0', '1', '1', "1"),
        il::Production::with_context('1', '0', '0', "0"),
        il::Production::with_context('1', '0', '1', "1F1"),
        il::Production::with_context('1', '1', '0', "1"),
        il::Production::with_context('1', '1', '1', "0"),
        il::Production::without_context('+', "-"),
        il::Production::without_context('-', "+"),
    ];

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.02,
        iterations: 30,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_hogeweg_a() -> (il::LSystem, lsys::Settings) {
    let mut sys = il::LSystem::new();

    sys.axiom = "F1F1F1".to_string();
    sys.ignore_from_context("+-F");
    sys.productions = vec![
        il::Production::with_context('0', '0', '0', "0"),
        il::Production::with_context('0', '0', '1', "1[+F1F1]"),
        il::Production::with_context('0', '1', '0', "1"),
        il::Production::with_context('0', '1', '1', "1"),
        il::Production::with_context('1', '0', '0', "0"),
        il::Production::with_context('1', '0', '1', "1F1"),
        il::Production::with_context('1', '1', '0', "0"),
        il::Production::with_context('1', '1', '1', "0"),
        il::Production::without_context('+', "-"),
        il::Production::without_context('-', "+"),
    ];

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.02,
        iterations: 30,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_hogeweg_c() -> (il::LSystem, lsys::Settings) {
    let mut sys = il::LSystem::new();

    sys.axiom = "F1F1F1".to_string();
    sys.ignore_from_context("+-F");
    sys.productions = vec![
        il::Production::with_context('0', '0', '0', "0"),
        il::Production::with_context('0', '0', '1', "1"),
        il::Production::with_context('0', '1', '0', "0"),
        il::Production::with_context('0', '1', '1', "1[+F1F1]"),
        il::Production::with_context('1', '0', '0', "0"),
        il::Production::with_context('1', '0', '1', "1F1"),
        il::Production::with_context('1', '1', '0', "0"),
        il::Production::with_context('1', '1', '1', "0"),
        il::Production::without_context('+', "-"),
        il::Production::without_context('-', "+"),
    ];

    let settings = lsys::Settings {
        angle: f32::to_radians(22.75),
        width: 0.02,
        iterations: 26,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_hogeweg_d() -> (il::LSystem, lsys::Settings) {
    let mut sys = il::LSystem::new();

    sys.axiom = "F0F1F1".to_string();
    sys.ignore_from_context("+-F");
    sys.productions = vec![
        il::Production::with_context('0', '0', '0', "1"),
        il::Production::with_context('0', '0', '1', "0"),
        il::Production::with_context('0', '1', '0', "0"),
        il::Production::with_context('0', '1', '1', "1F1"),
        il::Production::with_context('1', '0', '0', "1"),
        il::Production::with_context('1', '0', '1', "1[+F1F1]"),
        il::Production::with_context('1', '1', '0', "1"),
        il::Production::with_context('1', '1', '1', "0"),
        il::Production::without_context('+', "-"),
        il::Production::without_context('-', "+"),
    ];

    let settings = lsys::Settings {
        angle: f32::to_radians(22.75),
        width: 0.02,
        iterations: 24,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_hogeweg_e() -> (il::LSystem, lsys::Settings) {
    let mut sys = il::LSystem::new();

    sys.axiom = "F1F1F1".to_string();
    sys.ignore_from_context("+-F");
    sys.productions = vec![
        il::Production::with_context('0', '0', '0', "0"),
        il::Production::with_context('0', '0', '1', "1[-F1F1]"),
        il::Production::with_context('0', '1', '0', "1"),
        il::Production::with_context('0', '1', '1', "1"),
        il::Production::with_context('1', '0', '0', "0"),
        il::Production::with_context('1', '0', '1', "1F1"),
        il::Production::with_context('1', '1', '0', "1"),
        il::Production::with_context('1', '1', '1', "0"),
        il::Production::without_context('+', "-"),
        il::Production::without_context('-', "+"),
    ];

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.02,
        iterations: 30,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_bush() -> (ol::LSystem, lsys::Settings) {
    let mut sys = ol::LSystem::new();

    sys.axiom = "A".to_string();
    sys.set_rule('A', "[&FL!A]>>>>>[&FL!A]>>>>>>>[&FL!A]");
    sys.set_rule('F', "S>>>>>F");
    sys.set_rule('S', "FL");
    sys.set_rule('L', "['^^{-f+f+f-|-f+f+f}]");
    sys.map_command('f', Command::Forward);

    let settings = lsys::Settings {
        angle: f32::to_radians(22.5),
        width: 0.1,
        shrink_rate: 1.5,
        iterations: 7,
        colors: vec![
            (193.0/255.0, 154.0/255.0, 107.0/255.0),
            (0.3, 1.0, 0.2),
        ],
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_flower() -> (ol::LSystem, lsys::Settings) {
    let mut sys = ol::LSystem::new();

    sys.axiom = "P".to_string();
    sys.set_rule('P', "I+[P+K]-->>[--L]I[++L]-[PK]++PK"); // Plant
    sys.set_rule('I', "FS[>>&&L][>>^^L]FS"); // Internode`
    sys.set_rule('S', "SFS"); // Segment
    sys.set_rule('L', "['{+F-FF-F+|+F-FF-F}]"); // Leaf
    sys.set_rule('K', "[&&&E''>W>>>>W>>>>W>>>>W>>>>W]"); // Flower
    sys.set_rule('E', "FF"); // Pedicel
    sys.set_rule('W', "['^F][&&&&{-F+F|-F+F}]"); // Wedge

    let settings = lsys::Settings {
        angle: f32::to_radians(18.0),
        width: 0.015,
        iterations: 5,
        colors: vec![
            (193.0/255.0, 154.0/255.0, 107.0/255.0),
            (0.3, 1.0, 0.2),
            (1.5, 1.5, 1.4),
            (1.5, 1.5, 0.5),
        ],
        ..lsys::Settings::new()
    };

    (sys, settings)
}

fn make_antenna() -> (param::LSystem, lsys::Settings) {
    let mut sys = param::LSystem::new();

    sys.axiom = param::Word::from_str("A");

    let r = 1.456;
    sys.productions = vec![
        param::Production::new(
            'A',
            vec![
                param::ProductionLetter::with_params('F', params_f![1.0]),
                param::ProductionLetter::new('['),
                param::ProductionLetter::new('+'),
                param::ProductionLetter::new('A'),
                param::ProductionLetter::new(']'),
                param::ProductionLetter::new('['),
                param::ProductionLetter::new('-'),
                param::ProductionLetter::new('A'),
                param::ProductionLetter::new(']'),
            ]
        ),
        param::Production::new(
            'F',
            vec![
                param::ProductionLetter::with_transform('F', move |p,_| params_f![p[0].f() * r]),
            ]
        ),
    ];

    let settings = lsys::Settings {
        angle: f32::to_radians(85.0),
        width: 0.05,
        iterations: 10,
        ..lsys::Settings::new()
    };

    (sys, settings)
}

//fn make_tree() -> (param::LSystem, lsys::Settings) {
//    let mut sys = param::LSystem::new();

//    sys.axiom = vec![
//        param::Letter::with_params('#', params_f![1.0/10.0]),
//        param::Letter::with_params('F', params_f![5.0]),
//        param::Letter::with_params('>', params_f![f32::to_radians(45.0)]),
//        param::Letter::new('A'),
//    ];

//    let d1 = f32::to_radians(94.74);
//    let d2 = f32::to_radians(132.63);
//    let a = f32::to_radians(18.95);
//    let lr = 1.309;
//    let vr = 1.732 / 10.0;

//    sys.productions = vec![
//        param::Production::new(
//            'A',
//            vec![
//                param::ProductionLetter::with_transform('#', move |_| params_f![vr]),
//                param::ProductionLetter::with_params('F', params_f![2.0]),
//                param::ProductionLetter::new('['),
//                param::ProductionLetter::with_transform('&', move |_| params_f![a]),
//                param::ProductionLetter::with_params('F', params_f![2.0]),
//                param::ProductionLetter::new('A'),
//                param::ProductionLetter::new(']'),
//                param::ProductionLetter::with_transform('>', move |_| params_f![d1]),
//                param::ProductionLetter::new('['),
//                param::ProductionLetter::with_transform('&', move |_| params_f![a]),
//                param::ProductionLetter::with_params('F', params_f![2.0]),
//                param::ProductionLetter::new('A'),
//                param::ProductionLetter::new(']'),
//                param::ProductionLetter::with_transform('>', move |_| params_f![d2]),
//                param::ProductionLetter::new('['),
//                param::ProductionLetter::with_transform('&', move |_| params_f![a]),
//                param::ProductionLetter::with_params('F', params_f![2.0]),
//                param::ProductionLetter::new('A'),
//                param::ProductionLetter::new(']'),
//            ]
//        ),
//        param::Production::new(
//            'F',
//            vec![
//                param::ProductionLetter::with_transform('F', move |p| params_f![p[0].f() * lr]),
//            ]
//        ),
//        param::Production::new(
//            '#',
//            vec![
//                param::ProductionLetter::with_transform('#', move |p| params_f![p[0].f() * vr]),
//            ]
//        ),
//    ];

//    let settings = lsys::Settings {
//        width: 0.05,
//        iterations: 7,
//        ..lsys::Settings::new()
//    };

//    (sys, settings)
//}

fn make_anim_tree() -> (param::LSystem, lsys::Settings) {
    let mut sys = param::LSystem::new();

    sys.axiom = param::Word::from_str("#(0.01)F(0.0)>(0.593412)A(0.0)");
    sys.command_map['f' as usize] = Command::Forward;

    let d1 = f32::to_radians(94.74);
    let d2 = f32::to_radians(132.63);
    let a = f32::to_radians(18.95);
    let lr = 1.309;
    let vr = 1.732 / 10.0;
    let ls = 0.1;

    sys.productions = vec![
        param::Production::with_condition(
            'A',
            |p| p[0].f() < 1.0,
            vec![
                param::ProductionLetter::with_transform('A', |p,dt| params_f![p[0].f() + dt]),
            ]
        ),
        param::Production::with_condition(
            'A',
            |p| p[0].f() >= 1.0,
            vec![
                param::ProductionLetter::new('['),
                param::ProductionLetter::new('L'),
                param::ProductionLetter::with_params('>', params_f![f32::to_radians(90.0)]),
                param::ProductionLetter::new('L'),
                param::ProductionLetter::with_params('>', params_f![f32::to_radians(90.0)]),
                param::ProductionLetter::new('L'),
                param::ProductionLetter::with_params('>', params_f![f32::to_radians(90.0)]),
                param::ProductionLetter::new('L'),
                param::ProductionLetter::new(']'),
                param::ProductionLetter::with_transform('#', move |_,_| params_f![vr]),
                param::ProductionLetter::with_params('F', params_f![0.0]),
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_transform('&', move |_,_| params_f![a]),
                param::ProductionLetter::with_params('F', params_f![0.0]),
                param::ProductionLetter::with_params('A', params_f![0.0]),
                param::ProductionLetter::new(']'),
                param::ProductionLetter::with_transform('>', move |_,_| params_f![d1]),
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_transform('&', move |_,_| params_f![a]),
                param::ProductionLetter::with_params('F', params_f![0.0]),
                param::ProductionLetter::with_params('A', params_f![0.0]),
                param::ProductionLetter::new(']'),
                param::ProductionLetter::with_transform('>', move |_,_| params_f![d2]),
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_transform('&', move |_,_| params_f![a]),
                param::ProductionLetter::with_params('F', params_f![0.0]),
                param::ProductionLetter::with_params('A', params_f![0.0]),
                param::ProductionLetter::new(']'),
            ]
        ),
        param::Production::new(
            'L',
            vec![
                param::ProductionLetter::new('['),
                param::ProductionLetter::with_params('&', params_f![f32::to_radians(45.0)]),
                param::ProductionLetter::new('\''),
                param::ProductionLetter::new('{'),
                param::ProductionLetter::with_params('+', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.01]),
                param::ProductionLetter::with_params('-', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.05]),
                param::ProductionLetter::with_params('-', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.01]),
                param::ProductionLetter::with_params('+', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::new('|'),
                param::ProductionLetter::with_params('+', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.01]),
                param::ProductionLetter::with_params('-', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.05]),
                param::ProductionLetter::with_params('-', params_f![f32::to_radians(60.0)]),
                param::ProductionLetter::with_params('f', params_f![0.01]),
                param::ProductionLetter::new('}'),
                param::ProductionLetter::new(']'),
            ]
        ),
        param::Production::new(
            'f',
            vec![
                param::ProductionLetter::with_transform('f', move |p,dt| params_f![p[0].f() + (dt * ls) / p[0].f()]),
            ]
        ),
        param::Production::new(
            'F',
            vec![
                param::ProductionLetter::with_transform('F', move |p,dt| params_f![p[0].f() + dt * lr]),
            ]
        ),
        param::Production::new(
            '#',
            vec![
                param::ProductionLetter::with_transform('#', move |p,dt| params_f![p[0].f() + dt * vr]),
            ]
        ),
    ];

    let settings = lsys::Settings {
        width: 0.05,
        iterations: 7,
        colors: vec![
            (193.0/255.0, 154.0/255.0, 107.0/255.0),
            (0.3, 1.0, 0.2),
        ],
        ..lsys::Settings::new()
    };

    (sys, settings)
}
