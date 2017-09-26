extern crate alga;
extern crate nalgebra as na;
#[macro_use]
extern crate nom;
extern crate num;
extern crate rand;

use std::f32::consts::{E, PI};
use std::path::Path;
use std::{fs, io, str};
use std::time::Duration;
use std::cmp::Ordering;
use num::Float;
use num::cast::cast;
use rand::distributions::{IndependentSample, Range};
use rand::Rng;

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
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
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
where
    N: Real,
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

    Ok(ReadDirAll {
        visit_stack: vec![top_dir],
    })
}

#[inline]
pub fn partial_min<T>(a: T, b: T) -> Option<T>
where
    T: PartialOrd,
{
    if let Some(ordering) = a.partial_cmp(&b) {
        match ordering {
            Ordering::Equal | Ordering::Less => Some(a),
            Ordering::Greater => Some(b),
        }
    } else {
        None
    }
}

#[inline]
pub fn partial_max<T>(a: T, b: T) -> Option<T>
where
    T: PartialOrd,
{
    if let Some(ordering) = a.partial_cmp(&b) {
        match ordering {
            Ordering::Equal | Ordering::Greater => Some(a),
            Ordering::Less => Some(b),
        }
    } else {
        None
    }
}

#[inline]
pub fn partial_clamp<T>(v: T, min: T, max: T) -> Option<T>
where
    T: PartialOrd,
{
    if let Some(min_clamped) = partial_max(v, min) {
        if let Some(max_clamped) = partial_min(min_clamped, max) {
            Some(max_clamped)
        } else {
            None
        }
    } else {
        None
    }
}

/// Parse a `std::time::Duration` from various formats
///
/// # Format
/// The parser supports various formats.
/// A plain unsigned integer (u64) will be parsed as seconds.
/// Alternatively an `hours:minutes:seconds` format is supported, where `:seconds` is optional.
/// `hours`, `minutes` and `seconds` can each be any u64 integer.
pub fn parse_duration_hms(string: &str) -> Result<Duration, &str> {
    use nom::IResult::{Done, Error, Incomplete};

    match parse::duration(string.as_bytes()) {
        Done(_, duration) => Ok(duration),
        Error(_) | Incomplete(_) => Err("Failed parsing duration in H:M:(S) format"),
    }
}

#[macro_export]
macro_rules! assert_approx_eq {
    ($x:expr, $y:expr, $eps:expr) => {
        {
            let (a, b) = ($x, $y);
            let epsilon = $eps;
            let ordering = (a - b).abs().partial_cmp(&epsilon);
            let equal = if let Some(ordering) = ordering {
                match ordering {
                    cmp::Ordering::Less => true,
                    _ => false,
                }
            } else {
                false
            };

            assert!(equal,
                    "assertion failed: `(left !== right)` (left: `{:?}`, right: `{:?}`)",
                    a,
                    b);
        }
    }
}

pub fn slice_approx_eq<T>(x: &[T], y: &[T], epsilon: T) -> bool
where
    T: Float,
{
    if x.len() != y.len() {
        return false;
    }

    let mut equal = true;
    for (a, b) in x.iter().zip(y.iter()) {
        let ordering = (*a - *b).abs().partial_cmp(&epsilon);
        if let Some(ordering) = ordering {
            match ordering {
                Ordering::Greater | Ordering::Equal => {
                    equal = false;
                    break;
                }
                _ => {}
            }
        } else if !a.is_nan() || !b.is_nan() {
            equal = false;
            break;
        }
    }

    equal
}

pub trait ToSeconds<T> {
    fn to_seconds(&self) -> T;
}

impl<T: Float> ToSeconds<T> for Duration {
    fn to_seconds(&self) -> T {
        let seconds: T = cast(self.as_secs()).unwrap();
        let nanoseconds: T = cast(self.subsec_nanos()).unwrap();
        seconds + nanoseconds / T::from(1_000_000_000.0).unwrap()
    }
}

/// Randomly select `num` elements from a list and return them.
///
/// An element can not be selected twice. If there are less than `num` elements,
// the whole `list` will be returned.
pub fn rand_select<'l, R, T>(rng: &mut R, list: &'l [T], num: usize) -> Vec<&'l T>
where
    R: Rng,
{
    if list.len() <= num {
        return list.iter().collect();
    }

    let mut bucket: Vec<&T> = list.iter().collect();

    (0..num)
        .map(|_| {
            let index = Range::new(0, list.len()).ind_sample(rng);
            bucket.swap_remove(index)
        })
        .collect()
}

/// Randomly remove `num` elements from a list and return them.
///
/// If there are less than `num` elements, all elements are removed.
pub fn rand_remove<R, T>(rng: &mut R, list: &mut Vec<T>, num: usize) -> Vec<T>
where
    R: Rng,
{
    if list.len() <= num {
        let mut result = Vec::new();
        result.append(list);
        return result;
    }

    (0..num)
        .map(|_| {
            let index = Range::new(0, list.len()).ind_sample(rng);
            list.swap_remove(index)
        })
        .collect()
}

#[macro_export]
macro_rules! assert_slice_approx_eq {
    ($x:expr, $y:expr, $eps:expr) => {
        {
            let (x, y) = ($x, $y);
            assert!($crate::slice_approx_eq(x, y, $eps),
                    "assertion failed: `(left != right)` \
                     (left: `{:?}`, right: `{:?}`)",
                     x,
                     y);
        }
    };
}

#[macro_export]
macro_rules! assert_slice_approx_ne {
    ($x:expr, $y:expr, $eps:expr) => {
        {
            let (x, y) = ($x, $y);
            assert!(!$crate::slice_approx_eq(x, y, $eps),
                    "assertion failed: `(left == right)` \
                     (left: `{:?}`, right: `{:?}`)",
                     x,
                     y);
        }
    };
}

mod parse {
    use std::str;
    use std::time::Duration;
    use nom::digit;

    named!(pub num_u64<u64>,
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
        alt_complete!(
            map!(
                do_parse!(
                    hours: call!(num_u64) >>
                    tag!(&b":"[..]) >>
                    minutes: call!(num_u64) >>
                    seconds: opt!(complete!(
                        do_parse!(
                            tag!(&b":"[..]) >>
                            seconds: call!(num_u64) >>
                            (seconds)
                        )
                    )) >>
                    (hours, minutes, seconds)
                ),
                |(hours, minutes, seconds)| {
                    let mut duration = minutes * 60 + hours * 60 * 60;
                    if let Some(seconds) = seconds {
                        duration += seconds;
                    }
                    Duration::new(duration, 0)
                }
            ) |
            map!(call!(num_u64), |seconds| Duration::new(seconds, 0))
        )
    );
}

#[cfg(test)]
mod test {
    use super::*;
    use nom::IResult::Done;
    use std::f32;

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
    fn test_project_nan() {
        use na::Vector2;

        assert!(
            project_onto(
                &Vector2::new(f32::NAN, 1.0),
                &Unit::new_unchecked(Vector2::new(1.0, 0.0))
            ).is_nan()
        );

        assert!(
            project_onto(
                &Vector2::new(f32::INFINITY, 0.0),
                &Unit::new_unchecked(Vector2::new(0.0, 0.0))
            ).is_nan()
        );

        assert!(!project_onto(
            &Vector2::new(0.0, 0.0),
            &Unit::new_unchecked(Vector2::new(0.0, 0.0))
        ).is_nan());
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

        assert_eq!(
            interpolate_cos(0.0, 1.0, 0.25),
            0.146446609406726237799577818947575480357582031155762981705
        );
    }

    #[test]
    fn test_read_dir_all() {
        let entries_it = read_dir_all("testdata/read_dir_all").unwrap();
        let paths = entries_it.map(|e| e.unwrap().path()).collect::<Vec<_>>();
        let path_str = paths
            .iter()
            .map(|p| p.to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            path_str,
            vec![
                "testdata/read_dir_all/a",
                "testdata/read_dir_all/a/c",
                "testdata/read_dir_all/a/c/d",
                "testdata/read_dir_all/a/c/e",
                "testdata/read_dir_all/a/b",
                "testdata/read_dir_all/a/b/y",
                "testdata/read_dir_all/a/b/z",
                "testdata/read_dir_all/a/b/x",
            ]
        )
    }

    #[test]
    fn test_partial_clamp() {
        assert_eq!(partial_clamp(0.0, -1.0, 1.0), Some(0.0));
        assert_eq!(partial_clamp(-1.0, -1.0, 1.0), Some(-1.0));
        assert_eq!(partial_clamp(-2.0, -1.0, 1.0), Some(-1.0));
        assert_eq!(partial_clamp(1.0, -1.0, 1.0), Some(1.0));
        assert_eq!(partial_clamp(2.0, -1.0, 1.0), Some(1.0));
    }

    #[test]
    fn test_parse_num_u64() {
        assert_eq!(parse::num_u64(&b"1"[..]), Done(&b""[..], (1)));
        assert_eq!(parse::num_u64(&b"12345"[..]), Done(&b""[..], (12345)));
        assert_eq!(
            parse::num_u64(&b"18446744073709551615"[..]),
            Done(&b""[..], (18446744073709551615))
        );
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(
            parse::duration(&b"0"[..]),
            Done(&b""[..], (Duration::from_secs(0)))
        );
        assert_eq!(
            parse::duration(&b"01"[..]),
            Done(&b""[..], (Duration::from_secs(1)))
        );
        assert_eq!(
            parse::duration(&b"1"[..]),
            Done(&b""[..], (Duration::from_secs(1)))
        );
        assert_eq!(
            parse::duration(&b"100"[..]),
            Done(&b""[..], (Duration::from_secs(100)))
        );
    }

    #[test]
    fn test_parse_duration_hms_seconds() {
        assert_eq!(
            parse::duration(&b"00:00:00"[..]),
            Done(&b""[..], (Duration::from_secs(0)))
        );
        assert_eq!(
            parse::duration(&b"00:00:01"[..]),
            Done(&b""[..], (Duration::from_secs(1)))
        );
        assert_eq!(
            parse::duration(&b"00:00:1"[..]),
            Done(&b""[..], (Duration::from_secs(1)))
        );
        assert_eq!(
            parse::duration(&b"00:00:100"[..]),
            Done(&b""[..], (Duration::from_secs(100)))
        );
    }

    #[test]
    fn test_parse_duration_hms_minutes() {
        assert_eq!(
            parse::duration(&b"00:00"[..]),
            Done(&b""[..], (Duration::from_secs(0)))
        );
        assert_eq!(
            parse::duration(&b"00:01"[..]),
            Done(&b""[..], (Duration::from_secs(60)))
        );
        assert_eq!(
            parse::duration(&b"00:1"[..]),
            Done(&b""[..], (Duration::from_secs(60)))
        );
        assert_eq!(
            parse::duration(&b"00:100"[..]),
            Done(&b""[..], (Duration::from_secs(100 * 60)))
        );
    }

    #[test]
    fn test_parse_duration_hms_hours() {
        assert_eq!(
            parse::duration(&b"00:00"[..]),
            Done(&b""[..], (Duration::from_secs(0)))
        );
        assert_eq!(
            parse::duration(&b"01:00"[..]),
            Done(&b""[..], (Duration::from_secs(60 * 60)))
        );
        assert_eq!(
            parse::duration(&b"1:00"[..]),
            Done(&b""[..], (Duration::from_secs(60 * 60)))
        );
        assert_eq!(
            parse::duration(&b"100:00"[..]),
            Done(&b""[..], (Duration::from_secs(100 * 60 * 60)))
        );
    }
}
