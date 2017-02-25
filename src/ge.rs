use abnf;

pub fn run_ge() {
    let lsys_abnf = abnf::parse_file("lsys.abnf").expect("Could not parse ABNF file");
    println!("{:#?}", lsys_abnf);
}
