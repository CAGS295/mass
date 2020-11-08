use crate::MassType;
use serde::Deserialize;
use std::ops::Add;
pub struct TimeSeries<T> {
    pub series: Vec<Record<T>>,
}

impl<T> TimeSeries<T> {
    pub fn new(series: Vec<Record<T>>) -> Self {
        Self { series }
    }
}

#[derive(Deserialize, Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct Record<T>(T);

impl From<f64> for Record<f64> {
    fn from(f: f64) -> Self {
        Record(f)
    }
}

impl From<Record<f64>> for f64 {
    fn from(r: Record<f64>) -> Self {
        r.0
    }
}

impl Add<f64> for Record<f64> {
    type Output = f64;

    fn add(self, rhs: f64) -> Self::Output {
        self.0 + rhs
    }
}

impl MassType for Record<f64> {}
