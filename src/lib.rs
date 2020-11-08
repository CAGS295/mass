use std::fmt::Debug;

use std::{ops, vec};

use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use num_cpus;
pub mod math;
pub mod numpy_bindings;
pub mod stats;

pub mod time_series;
use math::argmin;
use rustfft::FFTplanner;
use stats::{append, mean, moving_avg as ma, moving_std as mstd, std, Append};

pub trait MassType: PartialOrd + From<f64> + Into<f64> + Copy + ops::Add<f64> + Debug {}

fn min_subsequence_distance<T>(start_idx: usize, subsequence: &[T], query: &[T]) -> (usize, f64)
where
    T: MassType,
{
    let distances = mass2(subsequence, query);

    //  find mininimum index of this batch which will be between 0 and batch_size
    let min_idx = argmin(&distances);

    // add this distance to best distances
    let dist = distances[min_idx];

    // println!("{:?}", subsequence);

    // compute the actual index and store it
    // fix index mapping due to window + stepby TODO
    let index = min_idx + start_idx;

    return (index, dist);
}

/// Compute the distance profile for the given query over the given time
/// series. Optionally, the correlation coefficient can be returned.
pub fn mass2<T: Debug>(ts: &[T], query: &[T]) -> Vec<f64>
where
    T: MassType,
{
    let n = ts.len();
    let m = query.len();

    debug_assert!(n >= m);

    // mu and sigma for query
    let mu_q = mean(query);
    let sigma_q = std(query);

    // Rolling mean and std for the time series
    let rolling_mean_ts = {
        let avgs = ma(ts, m);
        append(avgs, m - 1, 1.0, Append::Front)
    };

    let rolling_sigma_ts = {
        let sigmas = mstd(ts, m);
        append(sigmas, m - 1, 0.0, Append::Front)
    };

    //flip query and add padding zeroes until it fits ts' length.
    let y = {
        let reversed: Vec<T> = query.iter().rev().map(|v| *v).collect();
        append(reversed, n - m, (0.0).into(), Append::Back)
    };

    // build in/out buffers for the FFTs
    let mut y: Vec<_> = y.iter().map(|y| Complex::new((*y).into(), 0.0)).collect();
    let mut y_o = vec![Complex::<f64>::zero(); y.len()];

    let mut x: Vec<_> = ts.iter().map(|x| Complex::new((*x).into(), 0.0)).collect();
    let mut x_o = vec![Complex::<f64>::zero(); x.len()];

    // FFTs
    let mut forward = FFTplanner::<f64>::new(false);
    let mut inverse = FFTplanner::<f64>::new(true);

    let fft = forward.plan_fft(n);
    let ifft = inverse.plan_fft(n);
    fft.process(&mut x[..], &mut x_o[..]);
    fft.process(&mut y[..], &mut y_o[..]);
    // mult transforms
    let mut z: Vec<Complex<_>> = x_o
        .iter()
        .zip(y_o.iter())
        .map(|(x, y)| (*x) * (*y))
        .collect();

    // inverse fft, reuse buffer
    let z_o = &mut x_o[..];
    ifft.process(&mut z[..], z_o);
    // is it safe to drop im part??
    let z: Vec<f64> = z_o.iter().map(|z| (*z).re).collect();

    // println!("{:?}", z);
    let dist = math::dist(
        mu_q,
        sigma_q,
        &rolling_mean_ts[..],
        &rolling_sigma_ts[..],
        n,
        m,
        &z[..],
    );
    dist
}

// need to try whether chunks over logical is faster than over physical cores SMT!!
pub fn cpus() -> usize {
    num_cpus::get()
}

///MASS2 batch is a batch version of MASS2 that reduces overall memory usage,
///provides parallelization and enables you to find top K number of matches
///within the time series. The goal of using this implementation is for very
///large time series similarity search. The returned results are not sorted
///by distance. So you will need to find the top match with np.argmin() or
///sort them yourself.
pub fn mass_batch<T: MassType>(
    ts: &[T],
    query: &[T],
    batch_size: usize,
    top_matches: usize,
    jobs: usize,
) -> Vec<(usize, f64)> {
    // asserts
    debug_assert!(batch_size > 0, "batch_size must be greater than 0.");
    debug_assert!(top_matches > 0, "Match at least one.");
    debug_assert!(jobs > 0, "Job count must be at least 1.");

    // silently use at max all available cpus
    let jobs = jobs.min(cpus());

    // split work into chunks
    //TODO schedule jobs
    assert!(jobs == 1);
    // TODO support nth top matches in parallel
    // consider doing full nth top matches with a partition pseudosort per thread to ensure global optima.

    let step_size = batch_size - query.len();
    let chunks = ts.windows(batch_size).step_by(step_size);

    let remainder = ts.len() % (step_size);
    let start = ts.len() - remainder;

    // replace with into_iter() TODO
    let tail = (&ts[start..]).chunks(batch_size);

    let mut dists: Vec<_> = chunks
        .chain(tail)
        .enumerate()
        .map(|(i, subsequence)| {
            // print!("{} {} {:p}\n", i, batch_size, subsequence);
            // print!("{} {}\n", i, batch_size);
            min_subsequence_distance(i * step_size, subsequence, query)
        })
        .collect();

    // let mut dists: Vec<_> = ts;
    // TODO implement partition instead of sort to reduce complexity from $O(Log(n)) \rightarrow O(n)$
    // use itertools partition TODO
    // todo!();
    dists.sort_unstable_by(|x, y| x.1.partial_cmp(&(y.1)).unwrap());
    // print!("{:?}", &dists[..top_matches]);

    dists.iter().take(top_matches).copied().collect()
}

#[cfg(test)]
pub mod tests {

    #[test]
    fn usize_div() {
        assert_eq!(5usize / 2usize, 2);
    }
}
