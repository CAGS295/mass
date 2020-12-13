//!
//!MASS: Mueen's Algorithm for Similarity Search in Rust!

//! > Similarity search for time series subsequences is THE most important subroutine for time series pattern mining. Subsequence similarity search has been scaled to trillions obsetvations under both DTW (Dynamic Time Warping) and Euclidean distances [a]. The algorithms are ultra fast and efficient. The key technique that makes the algorithms useful is the Early Abandoning technique [b,e] known since 1994. However, the algorithms lack few properties that are useful for many time series data mining algorithms.

//! > 1. Early abandoning depends on the dataset. The worst case complexity is still O(nm) where n is the length of the larger time series and m is the length of the short query.
//! > 2. The algorithm can produce the most similar subsequence to the query and cannot produce the Distance Profile to all the subssequences given the query.

//! > MASS is an algorithm to create Distance Profile of a query to a long time series. In this page we share a code for The Fastest Similarity Search Algorithm for Time Series Subsequences under Euclidean Distance. Early abandoning can occasionally beat this algorithm on some datasets for some queries. This algorithm is independent of data and query. The underlying concept of the algorithm is known for a long time to the signal processing community. We have used it for the first time on time series subsequence search under z-normalization. The algorithm was used as a subroutine in our papers [c,d] and the code are given below.

//! > 1. The algorithm has an overall time complexity of O(n log n) which does not depend on datasets and is the lower bound of similarity search over time series subsequences.
//! > 2. The algorithm produces all of the distances from the query to the subsequences of a long time series. In our recent paper, we generalize the usage of the distance profiles calculated using MASS in finding motifs, shapelets and discords. Check out the paper here.

//!##features

//!["jemalloc"] *TODO*
//!["pseudo_distance"] simplifies the distance with the same optimization goal for increased performance.
//!The distance output is no longer a distance but a score with the same optimum.

//!@misc{FastestSimilaritySearch,
//!title={The Fastest Similarity Search Algorithm for Time Series Subsequences under Euclidean Distance},
//!author={ Mueen, Abdullah and Zhu, Yan and Yeh, Michael and Kamgar, Kaveh and Viswanathan, Krishnamurthy and Gupta, Chetan and Keogh, Eamonn},
//!year={2017},
//!month={August},
//!note = {\url{http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html}}
//!}
#[cfg(all(not(target_env = "msvc"), jemallocator))]
use jemallocator::Jemalloc;

#[cfg(all(not(target_env = "msvc"), jemallocator))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use std::fmt::Debug;

use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use std::ops;

use num_cpus;
pub mod math;
pub mod stats;

pub mod time_series;
use math::argmin;
use math::fft_mult;
use stats::{mean, moving_avg as ma, moving_std as mstd, std};

pub trait MassType:
    PartialOrd + From<f64> + Into<f64> + Copy + ops::Add<f64> + Debug + Default + Sync
{
}

/// compute the MASS distance and return the index and value of the minimum found.
fn min_subsequence_distance<T>(start_idx: usize, subsequence: &[T], query: &[T]) -> (usize, f64)
where
    T: MassType,
{
    let distances = mass(subsequence, query);

    //  find mininimum index of this batch which will be between 0 and batch_size
    let min_idx = argmin(&distances);

    // add the minimum distance found to the best distances
    let dist = distances[min_idx];

    // compute the global index
    let index = min_idx + start_idx;

    return (index, dist);
}

/// Compute the distance profile for the given query over the given time
/// series. Optionally, the correlation coefficient can be returned.
pub fn mass<T: Debug + Default>(ts: &[T], query: &[T]) -> Vec<f64>
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
    let rolling_mean_ts = ma(ts, m);

    let rolling_sigma_ts = mstd(ts, m);

    let z = fft_mult(&ts, &query);

    let dist = math::dist(
        mu_q,
        sigma_q,
        rolling_mean_ts,
        rolling_sigma_ts,
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

/// Masss batch finds top K subsequences with the lowest distance profile for a given query.
/// This method implements MASS V3 where chunks are split in powers of two and computed in parallel.
/// Results are partitioned and not sorted, you can sort them afterwards if required.
/// Parallelization is yet to be implemented.
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

    // silently use at max all available cpus;not supported atm
    let jobs = jobs.min(cpus());

    // split work into chunks
    //TODO schedule jobs
    assert!(jobs == 1);
    // TODO support nth top matches in parallel
    // consider doing full nth top matches with a partition pseudosort per thread to ensure global optima.

    let mut dists: Vec<_> = job_index(ts.len(), query.len(), batch_size, jobs)
        .into_iter()
        .par_bridge()
        .map(|(l, h)| min_subsequence_distance(l, &ts[l..=h], query))
        .collect();

    // TODO implement partition instead of sort to reduce complexity from $O(Log(n)) \rightarrow O(n)$
    // use itertools partition TODO
    dists.sort_unstable_by(|x, y| x.1.partial_cmp(&(y.1)).unwrap());

    dists.iter().take(top_matches).copied().collect()
}

/// Generate the index for time series slices of size batch size; Batch size may be rounded to the nearest power of two.
/// Rounding to the nearest power of two may panic! if the new batch size is greater than the time series' length.
#[inline]
fn job_index(
    ts: usize,
    query: usize,
    mut batch_size: usize,
    _jobs: usize,
) -> impl Iterator<Item = (usize, usize)> {
    assert!(
        batch_size > query,
        "batch size must be greater than the query's length"
    );

    if !batch_size.is_power_of_two() {
        batch_size = batch_size.next_power_of_two();
    }

    debug_assert!(batch_size <= ts);
    debug_assert!(batch_size >= query);

    let step_size = batch_size - (query - 1);

    let index = (0..ts - query)
        .step_by(step_size)
        .map(move |i| (i, (ts - 1).min(i + batch_size - 1)));
    index
}

#[cfg(test)]
pub mod tests {

    use super::*;

    #[test]
    fn usize_div() {
        assert_eq!(5usize / 2usize, 2);
    }

    #[test]
    #[ignore = "for manual inspection purposes"]
    fn jobs_range_0() {
        let a = job_index(6, 2, 4, 1);
        for i in a {
            print!("{:?}\n", i);
        }
    }

    #[test]
    fn jobs_range_1() {
        let mut a = job_index(10, 4, 5, 1);
        assert!(a.next().unwrap() == (0, 7));
        assert!(a.next().unwrap() == (5, 9));
        assert!(a.next() == None);
    }

    #[test]
    fn jobs_range_2() {
        let mut a = job_index(6, 2, 4, 1);
        assert!(a.next().unwrap() == (0, 3));
        assert!(a.next().unwrap() == (3, 5));
        assert!(a.next() == None);
    }

    #[test]
    fn jobs_range_3() {
        let mut a = job_index(8, 2, 8, 1);
        assert!(a.next().unwrap() == (0, 7));
        assert!(a.next() == None);
    }
    #[test]
    fn jobs_range_4() {
        let mut a = job_index(6, 3, 4, 1);

        assert!(a.next().unwrap() == (0, 3));
        assert!(a.next().unwrap() == (2, 5));
        assert!(a.next() == None);
    }

    #[test]
    fn integration_1() {
        let a = &[10., 3., 2., 3., 4.5, 6., 0., -1.];
        let b = &[2., 3.];
        let bsize = 4;
        let c = mass_batch(a, b, bsize, 1, 1);
        println!("{:?}", c);
        assert!(c[0].0 == 2);
    }

    #[test]
    fn integration_2() {
        let a = &[0., 10., 20., 30., 50., 10.];
        let b = &[2., 3., 2.];
        let c = mass_batch(a, b, 4, 1, 1);
        assert!(c[0].0 == 3);
    }
}
