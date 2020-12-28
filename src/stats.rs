use std::{
    iter::{self},
    ops::{Add, DivAssign, SubAssign},
};

use itertools::izip;
use std::iter::Zip;
use std::slice::Iter;

/// $$S =\sum_i x_i$$
#[inline]
fn sum<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    values.iter().fold(0.0, |x, &y| x + y.into())
}

/// $$S^2 =\sum_i x_i^2$$
#[inline]
fn sum_squared<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    values.iter().fold(0.0, |x, &y| {
        let y = y.into();
        x + y * y
    })
}

pub struct RollingStd<'a, T> {
    sum: f64,
    sum_sq: f64,
    n: f64,
    packed_slices: Zip<Iter<'a, T>, Iter<'a, T>>,
    extra: bool,
}

impl<'a, T> RollingStd<'a, T> {
    pub fn new(sum: f64, sum_sq: f64, n: usize, packed_slices: (&'a [T], &'a [T])) -> Self {
        RollingStd {
            sum,
            sum_sq,
            n: n as f64,
            packed_slices: izip!(packed_slices.0, packed_slices.1),
            extra: true,
        }
    }
}

impl<'a, T: Into<f64> + Copy> Iterator for RollingStd<'a, T> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let s = self.sum;
        let sq = self.sum_sq;
        match self.packed_slices.next() {
            Some((&x_a, &x_b)) => {
                let x_a = x_a.into();
                let x_b = x_b.into();
                self.sum += -x_a + x_b;
                self.sum_sq += -x_a * x_a + x_b * x_b;
                Some((sq / self.n - s * s / (self.n * self.n)).sqrt())
            }
            None => {
                if self.extra {
                    self.extra = false;
                    Some((sq / self.n - s * s / (self.n * self.n)).sqrt())
                } else {
                    None
                }
            }
        }
    }
}
/// compute the moving mean in $$O(n)$$ time.
#[inline]
pub fn rolling_std<T: Into<f64> + Add<f64> + Copy>(
    values: &[T],
    window_size: usize,
) -> RollingStd<T> {
    let oldest = &values[..values.len() - window_size];

    let newest = &values[window_size..];

    let s = sum(&values[..window_size]);
    let ssq = sum_squared(&values[..window_size]);

    RollingStd::new(s, ssq, window_size, (oldest, newest))
}

pub struct RollingMean<'a, T: 'a> {
    sum: f64,
    n: f64,
    packed_slices: Zip<Iter<'a, T>, Iter<'a, T>>,
    extra: bool,
}

impl<'a, T: 'a> RollingMean<'a, T> {
    pub fn new(sum: f64, n: usize, packed_slices: (&'a [T], &'a [T])) -> Self {
        RollingMean {
            sum,
            n: n as f64,
            packed_slices: izip!(packed_slices.0, packed_slices.1),
            extra: true,
        }
    }
}

impl<'a, T: 'a + Into<f64> + Copy> Iterator for RollingMean<'a, T> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        match self.packed_slices.next() {
            Some((&x_a, &x_b)) => {
                let dividend = self.sum;
                self.sum -= x_a.into();
                self.sum += x_b.into();
                Some(dividend / self.n)
            }
            None => {
                if self.extra {
                    self.extra = false;
                    Some(self.sum / self.n)
                } else {
                    None
                }
            }
        }
    }
}

/// compute the moving mean in $$O(n)$$ time, favorable over iterating over windows with complexity $$O(nm),\quadm:window size$$
///$$ \mu_i = \frac{-a_{i-n} + S_{i-n:i-1} +a_i}{n}$$
#[inline]
pub fn rolling_mean<T: Into<f64> + Add<f64> + Copy>(
    values: &[T],
    window_size: usize,
) -> RollingMean<T> {
    let oldest = &values[..values.len() - window_size];
    let newest = &values[window_size..];

    let sum = sum(&values[..window_size]);
    let packed_slices = (oldest, newest);
    RollingMean::new(sum, window_size, packed_slices)
}

/// $$E_\[X\]$$
#[inline]
pub fn mean<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    sum(values) / values.len() as f64
}

/// $$E[X^2]$$
#[inline]
fn e_x2<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    sum_squared(values) / values.len() as f64
}

///$$Var\[X\]$$
// could use an online method
#[inline]
pub fn var<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    let mu = mean(values);
    e_x2(values) - mu * mu
}

///Standard Deviation of X
#[inline]
pub fn std<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    var(values).sqrt()
}

///Moving Average of X
#[inline]
pub fn moving_avg<'a, T: Into<f64> + Add<f64> + Copy>(
    values: &'a [T],
    periods: usize,
) -> impl Iterator<Item = f64> + 'a {
    rolling_mean(values, periods)
}

/// append insertion enum for __fn append()__.
//TODO move into proper scope
pub enum Append {
    Front,
    Back,
}

/// Append a sequence of a constant value into the front or back of a Slice.

pub fn append<T>(mut values: Vec<T>, reps: usize, item: T, position: Append) -> Vec<T>
where
    T: From<f64> + Copy + Sized,
{
    let items = iter::repeat(item).take(reps);

    match position {
        Append::Back => {
            values.extend(items);
            values
        }
        Append::Front => items.chain(values).collect(),
    }
}

///Normalize X into $\frac{X - \mu }{\sigma}$
pub fn normalization<T: Into<f64> + Add<f64> + DivAssign<f64> + SubAssign<f64> + Copy>(
    values: &[T],
) -> Vec<f64> {
    let (mu, sigma) = (mean(values), std(values));
    values
        .iter()
        .map(move |&v| (v.into() - mu) / sigma)
        .collect()
}

///Moving standard deviation from X
pub fn moving_std<'a, T: Into<f64> + Add<f64> + Copy>(
    values: &'a [T],
    periods: usize,
) -> impl Iterator<Item = f64> + 'a {
    rolling_std(values, periods)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    #[test]
    fn jit_mean() {
        let x: [f64; 10] = [0., 2., 4., 3., 5., 6., 7., 3., 6., 5.];
        let a: Vec<_> = x.windows(3).map(|x| mean(x)).collect();
        let b = rolling_mean(&x, 3);
        itertools::assert_equal(a, b);
    }

    #[test]
    fn jit_std() {
        let x = [0., 2., 4., 3., 5., 6., 7., 3., 6., 5.];
        let a: Vec<_> = x.windows(3).map(|x| std(x)).collect();
        let b = rolling_std(&x, 3);
        izip!(a.iter(), b).for_each(|(a, b)| assert!(a - b < 1e-15));
    }

    #[test]
    fn rolling_mean_0() {
        let x = [2., 4., 5., 6., 7., 6., 5.];
        let res = [3., 4.5, 5.5, 6.5, 6.5, 5.5];

        let rm = rolling_mean(&x, 2).collect::<Vec<_>>();
        itertools::assert_equal(rm.iter(), res.iter());
    }
}
