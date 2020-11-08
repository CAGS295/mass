use std::{
    iter,
    ops::{Add, DivAssign, SubAssign},
};

/// $$E_[X]$$
// could use online method
pub fn mean<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    values.iter().fold(0.0, |x, y| x + (*y).into()) / values.len() as f64
}

/// $$E[X^2]$$
// could use online method
// product1? TODO
fn e_x2<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    values
        .iter()
        .fold(0.0, |x, y| (*y).into() * (*y).into() + x)
        / values.len() as f64
}

///$$Var[X]$$
// could use online method
pub fn var<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    let mu = mean(values);
    e_x2(values) - mu * mu
}

///Standard Deviation of X
pub fn std<T: Into<f64> + Add<f64> + Copy>(values: &[T]) -> f64 {
    var(values).sqrt()
}

// use std::fmt::Debug;
///Moving Average of X
// pub fn moving_avg<T: Into<f64> + Add<f64> + Copy + Debug>(
pub fn moving_avg<T: Into<f64> + Add<f64> + Copy>(values: &[T], periods: usize) -> Vec<f64> {
    let windows = values.windows(periods);
    windows.map(|win| mean(win)).collect()
}

/// append insertion enum for __fn append()__.
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
pub fn moving_std<T: Into<f64> + Add<f64> + Copy>(values: &[T], periods: usize) -> Vec<f64> {
    let windows = values.windows(periods);
    windows.map(|win| std(win)).collect()
}
