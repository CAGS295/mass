use std::{
    iter::{self, repeat},
    ops::{Add, DivAssign, SubAssign},
};

use itertools::izip;

/// $$S =\sum_i x_i$$
#[inline]
fn sum<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    values.iter().fold(0.0, |x, y| x + (*y).into())
}

/// $$S^2 =\sum_i x_i^2$$
#[inline]
fn sum_squared<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    values.iter().fold(0.0, |x, y| {
        let y = (*y).into();
        x + y * y
    })
}

/// compute the moving mean in $$O(n)$$ time.
#[inline]
pub fn rolling_std<T: Into<f64> + Add<f64> + Copy + Default>(
    values: &[T],
    window_size: usize,
) -> Vec<f64> {
    let n = values.len();

    let zero = &T::default();
    let oldest = {
        let oldest = &values[..n - window_size];
        repeat(zero).take(1).chain(oldest)
    };

    let newest = {
        let newest = &values[window_size..];
        repeat(zero).take(1).chain(newest)
    };

    let s = &mut sum(&values[..window_size]);
    let ssq = &mut sum_squared(&values[..window_size]);

    let packed = izip!(oldest, newest);
    packed
        .map(|(xa, xb)| {
            let xa = (*xa).into();
            let xb = (*xb).into();
            let m = window_size as f64;
            *s += -xa + xb;
            *ssq += -xa * xa + xb * xb;
            (*ssq / m - *s * *s / (m * m)).sqrt()
        })
        .collect::<Vec<f64>>()
}

/// compute the moving mean in $$O(n)$$ time, favorable over iterating over windows with complexity $$O(nm),\quadm:window size$$
#[inline]
pub fn rolling_mean<T: Into<f64> + Add<f64> + Copy + Default>(
    values: &[T],
    window_size: usize,
) -> Vec<f64> {
    let n = values.len();

    let zero = &T::default();
    let oldest = {
        let oldest = &values[..n - window_size];
        repeat(zero).take(1).chain(oldest)
    };

    let newest = {
        let newest = &values[window_size..];
        repeat(zero).take(1).chain(newest)
    };

    let inner_sum = &mut sum(&values[..window_size]);

    let packed = izip!(oldest, newest);
    packed
        .map(|(xa, xb)| {
            *inner_sum -= (*xa).into();
            *inner_sum += (*xb).into();
            *inner_sum / window_size as f64
        })
        .collect::<Vec<f64>>()
}

/// $$E_[X]$$
pub fn mean<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    sum(values) / values.len() as f64
}

/// $$E[X^2]$$
fn e_x2<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    sum_squared(values) / values.len() as f64
}

///$$Var[X]$$
// could use an online method
pub fn var<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    let mu = mean(values);
    e_x2(values) - mu * mu
}

///Standard Deviation of X
pub fn std<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    var(values).sqrt()
}

///Moving Average of X
pub fn moving_avg<T: Into<f64> + Add<f64> + Copy + Default>(
    values: &[T],
    periods: usize,
) -> Vec<f64> {
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
        .map(move |v| ((*v).into() - mu) / sigma)
        .collect()
}

///Moving standard deviation from X
pub fn moving_std<T: Into<f64> + Add<f64> + Copy + Default>(
    values: &[T],
    periods: usize,
) -> Vec<f64> {
    rolling_std(values, periods)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    #[test]
    fn jit_mean() {
        let x = [0., 2., 4., 3., 5., 6., 7., 3., 6., 5.];
        let a: Vec<_> = x.windows(3).map(|x| mean(x)).collect();
        let b = rolling_mean(&x, 3);
        itertools::assert_equal(a, b);
    }

    #[test]
    fn jit_std() {
        let x = [0., 2., 4., 3., 5., 6., 7., 3., 6., 5.];
        let a: Vec<_> = x.windows(3).map(|x| std(x)).collect();
        let b = rolling_std(&x, 3);
        izip!(a.iter(), b.iter()).for_each(|(a, b)| assert!(a - b < 1e-15));
    }

    #[test]
    fn rolling_mean_0() {
        let x = [2., 4., 5., 6., 7., 6., 5.];
        let res = [3., 4.5, 5.5, 6.5, 6.5, 5.5];

        let rm = rolling_mean(&x, 2);
        itertools::assert_equal(rm.iter(), res.iter());
        for i in rm {
            print!("{} ", i);
        }
        println!();
    }
}
