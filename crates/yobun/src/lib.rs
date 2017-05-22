extern crate alga;
extern crate nalgebra as na;
#[macro_use]
extern crate nom;

use std::f32::consts::{PI, E};
use std::path::Path;
use std::{fs, io, str};
use std::time::Duration;

use na::Unit;
use alga::general::Real;
use alga::linear::FiniteDimVectorSpace;

/// Calculate the normal distribution probability of a value `x` with standard deviation `sd`
/// and mean `mean`.
///
/// [See Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution) for details.
#[inline]
pub fn normal(x: f32, mean: f32, sd: f32) -> f32 {
    E.powf(-(x - mean).abs().sqrt() / (2.0 * sd.sqrt())) / ((2.0 * PI).sqrt() * sd)
}

/// Calculate the gaussian of value `x` with constants `a`, `b`, `c`.
///
/// [See Wikipedia](https://en.wikipedia.org/wiki/Gaussian_function) for details.
#[inline]
pub fn gaussian(x: f32, a: f32, b: f32, c: f32) -> f32 {
    a * E.powf(-((x - b).powi(2) / (2.0 * c.powi(2))))
}

/// Get the minimum and maximum of two values `a` and `b` as a tuple `(min, max)`.
///
/// More convenient than using both `a.min(b)` and `a.max(b)` if you want both the minimum
/// and maximum of two values.
///
/// # Example
/// ```
/// use yobun::min_max;
///
/// let a = 1.0f32;
/// let b = -1.0f32;
/// assert_eq!(min_max(a, b), (b, a));
/// assert_eq!(min_max(b, a), (a.min(b), a.max(b)));
/// ```
#[inline]
pub fn min_max<T: PartialOrd>(a: T, b: T) -> (T, T) {
    if a < b { (a, b) } else { (b, a) }
}

/// Projects vector `a` onto direction vector `b`.
///
/// This will yield how far vector `a` reaches in diretion `b`.
///
/// # Example
/// ```
/// # extern crate nalgebra;
/// # extern crate yobun;
/// use nalgebra::{Vector2, Unit};
/// use yobun::project_onto;
///
/// # fn main() {
/// let a = Vector2::new(10.0, 5.0);
/// let b = Unit::new_unchecked(Vector2::new(1.0, 0.0));
/// assert_eq!(project_onto(&a, &b), 10.0);
/// # }
/// ```
#[inline]
pub fn project_onto<V: FiniteDimVectorSpace>(a: &V, b: &Unit<V>) -> V::Field {
    na::dot(a, &**b)
}

/// Linearly interpolate between `a` and `b` at time `t` where `t` is in range [0.0, 1.0]
#[inline]
pub fn interpolate_linear<N>(a: N, b: N, t: N) -> N
    where N: Real
{
    a * (na::one::<N>() - t) + b * t
}

/// Interpolate between `a` and `b` at time `t` where `t` is in range [0.0, 1.0] using cosine.
///
/// This will create a smoother interpolation that eases in and out. See
/// [interpolation methods by Paul Bourke](http://paulbourke.net/miscellaneous/interpolation/)
#[inline]
pub fn interpolate_cos(a: f32, b: f32, t: f32) -> f32 {
    interpolate_linear(a, b, (1.0 - (t * PI).cos()) * 0.5)
}

pub struct ReadDirAll {
    visit_stack: Vec<fs::ReadDir>,
}

impl Iterator for ReadDirAll {
    type Item = io::Result<fs::DirEntry>;

    fn next(&mut self) -> Option<io::Result<fs::DirEntry>> {
        let mut iter = match self.visit_stack.pop() {
            Some(iter) => iter,
            None => return None,
        };

        let mut entry = iter.next();
        while entry.is_none() {
            iter = match self.visit_stack.pop() {
                Some(iter) => iter,
                None => return None,
            };

            entry = iter.next();
        }

        let entry = match entry {
            Some(entry) => entry,
            None => return None,
        };

        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => return Some(Err(err)),
        };

        self.visit_stack.push(iter);

        let path = entry.path();
        if path.is_dir() {
            match fs::read_dir(path) {
                Ok(entries) => self.visit_stack.push(entries),
                Err(err) => return Some(Err(err)),
            }
        }

        Some(Ok(entry))
    }
}

#[inline]
pub fn read_dir_all<P: AsRef<Path>>(path: P) -> io::Result<ReadDirAll> {
    let top_dir = fs::read_dir(path)?;

    Ok(ReadDirAll { visit_stack: vec![top_dir] })
}

#[inline]
pub fn partial_min<T>(a: T, b: T) -> T
    where T: PartialOrd
{
    if a <= b {
        a
    } else {
        b
    }
}

#[inline]
pub fn partial_max<T>(a: T, b: T) -> T
    where T: PartialOrd
{
    if a >= b {
        a
    } else {
        b
    }
}

#[inline]
pub fn partial_clamp<T>(v: T, min: T, max: T) -> T
    where T: PartialOrd
{
    partial_min(partial_max(v, min), max)
}

pub fn parse_duration_hms(string: &str) -> Result<Duration, &str> {
    use nom::IResult::{Done, Error, Incomplete};

    match parse::duration(string.as_bytes()) {
        Done(_, duration) => Ok(duration),
        Error(_) | Incomplete(_) => Err("Failed parsing duration in H:M:(S) format")
    }
}

#[macro_export]
macro_rules! assert_slice_approx_eq {
    ($x:expr, $y:expr) => {
        {
            assert!($x.len() == $y.len(),
                    "assertion failed: `(left !== right)` \
                     (left: `{:?}`, right: `{:?}`)",
                     $x,
                     $y);

            let mut equal = true;
            for (a, b) in $x.iter().zip($y.iter()) {
                if (a - b).abs() >= f32::EPSILON {
                    equal = false;
                    break;
                }
            }

            assert!(equal,
                    "assertion failed: `(left !== right)` \
                     (left: `{:?}`, right: `{:?}`)",
                     $x,
                     $y);
        }
    };
}

mod parse {
    use std::str;
    use std::time::Duration;
    use nom::{digit};

    named!(number<u64>,
        fold_many1!(
            call!(digit),
            0_u64,
            |mut acc: u64, item| {
                acc += str::from_utf8(item).unwrap().parse().unwrap();
                acc
            }
        )
    );

    named!(pub duration<Duration>,
        alt!(
            map!(call!(number), |seconds| Duration::new(seconds, 0)) |
            map!(
                do_parse!(
                    hours: call!(number) >>
                    tag!(&b":"[..]) >>
                    minutes: call!(number) >>
                    seconds: opt!(
                        do_parse!(
                            tag!(&b":"[..]) >>
                            seconds: call!(number) >>
                            (seconds)
                        )
                    ) >>
                    (hours, minutes, seconds)
                ),
                |(hours, minutes, seconds)| {
                    let mut duration = minutes * 60 + hours * 60 * 60;
                    if let Some(seconds) = seconds {
                        duration += seconds;
                    }
                    Duration::new(duration, 0)
                }
            )
        )
    );
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_min_max() {
        let a = 1.0f32;
        let b = -1.0f32;

        assert_eq!(min_max(a, b), (b, a));
        assert_eq!(min_max(b, a), (b, a));
        assert_eq!(min_max(b, a), (a.min(b), a.max(b)));
    }

    #[test]
    fn test_project_onto() {
        use na::Vector2;

        let a = Vector2::new(10.0, 5.0);
        let b = Unit::new_unchecked(Vector2::new(1.0, 0.0));
        assert_eq!(project_onto(&a, &b), 10.0);
    }

    #[test]
    fn test_interpolate_linear() {
        assert_eq!(interpolate_linear(0.0, 1.0, 0.0), 0.0);
        assert_eq!(interpolate_linear(0.0, 1.0, 0.5), 0.5);
        assert_eq!(interpolate_linear(0.0, 1.0, 1.0), 1.0);
    }

    #[test]
    fn test_interpolate_cos() {
        assert_eq!(interpolate_cos(0.0, 1.0, 0.0), 0.0);
        assert_eq!(interpolate_cos(0.0, 1.0, 0.5), 0.5);
        assert_eq!(interpolate_cos(0.0, 1.0, 1.0), 1.0);

        assert_eq!(interpolate_cos(0.0, 1.0, 0.25),
                   0.146446609406726237799577818947575480357582031155762981705);
    }

    #[test]
    fn test_read_dir_all() {
        let entries_it = read_dir_all("testdata/read_dir_all").unwrap();
        let paths = entries_it.map(|e| e.unwrap().path()).collect::<Vec<_>>();
        let path_str = paths
            .iter()
            .map(|p| p.to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(path_str,
                   vec!["testdata/read_dir_all/a",
                        "testdata/read_dir_all/a/c",
                        "testdata/read_dir_all/a/c/d",
                        "testdata/read_dir_all/a/c/e",
                        "testdata/read_dir_all/a/b",
                        "testdata/read_dir_all/a/b/y",
                        "testdata/read_dir_all/a/b/z",
                        "testdata/read_dir_all/a/b/x"])
    }

    #[test]
    fn test_partial_clamp() {
        assert_eq!(partial_clamp(0.0, -1.0, 1.0), 0.0);
        assert_eq!(partial_clamp(-1.0, -1.0, 1.0), -1.0);
        assert_eq!(partial_clamp(-2.0, -1.0, 1.0), -1.0);
        assert_eq!(partial_clamp(1.0, -1.0, 1.0), 1.0);
        assert_eq!(partial_clamp(2.0, -1.0, 1.0), 1.0);
    }
}
