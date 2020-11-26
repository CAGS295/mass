use crate::stats::{append, Append};
use crate::MassType;
use itertools::izip;
use rustfft::{num_complex::Complex, num_traits::Zero, FFTplanner};

pub fn argmin<T: PartialOrd + Copy>(values: &[T]) -> usize {
    let mut index = 0;
    let (first, rest) = values.split_first().unwrap();
    let mut current = *first;
    for (i, v) in rest.iter().enumerate() {
        if *v < current {
            current = *v;
            index = i + 1;
        }
    }
    index
}

pub fn dist(
    mu_q: f64,
    sigma_q: f64,
    mu_x: &[f64],
    sigma_x: &[f64],
    x_len: usize,
    y_len: usize,
    z: &[f64],
) -> Vec<f64> {
    let n_y = y_len as f64;
    let start = y_len - 1;
    let end = x_len;
    let z_clipped = &z[start..end];
    let mu_x_clipped = mu_x;
    let sigma_x_clipped = sigma_x;

    debug_assert!(
        z_clipped.len() == mu_x_clipped.len() && z_clipped.len() == sigma_x_clipped.len()
    );

    let vars = izip!(mu_x_clipped, sigma_x_clipped, z_clipped);

    // faster with same opt. goal .5dist^2$
    let f = |(mu_x, s_x, z): (&f64, &f64, &f64)| -> f64 {
        let divisor = z - n_y * mu_x * mu_q;
        let dividend = s_x * sigma_q;
        #[cfg(not(feature = "pseudo_distance"))]
        let res = (2. * (n_y - divisor / dividend)).sqrt();
        #[cfg(feature = "pseudo_distance")]
        let res = n_y - divisor / dividend;
        res
    };

    vars.map(f).collect()
}

pub fn fft_mult<T>(ts: &[T], query: &[T]) -> Vec<f64>
where
    T: MassType,
{
    let n = ts.len();
    let m = query.len();

    let y = {
        let reversed: Vec<T> = query.iter().rev().copied().collect();
        append(reversed, n - m, (0.0).into(), Append::Back)
    };

    debug_assert!(n == y.len());

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
    let mut z: Vec<_> = izip!(x_o, y_o).map(|(x, y)| x * y).collect();

    // inverse fft, reuse buffer
    let z_o = &mut x[..];
    ifft.process(&mut z[..], z_o);
    let k = n as f64;

    let z: Vec<f64> = z_o.iter().map(|z| (*z).re / k).collect();
    z
}

#[cfg(test)]
pub mod test {
    use super::*;

    /// fft does not normalize by 1/n.sqrt() out of the box.
    #[test]
    fn fft() {
        let n = 10;
        let v = 0..n;
        let mut x: Vec<_> = v.map(|x| Complex::new((x as f64).into(), 0.0)).collect();
        let mut x_o = vec![Complex::<f64>::zero(); x.len()];
        // FFTs
        let mut forward = FFTplanner::<f64>::new(false);
        let mut inverse = FFTplanner::<f64>::new(true);

        let fft = forward.plan_fft(n);
        let ifft = inverse.plan_fft(n);
        fft.process(&mut x, &mut x_o);
        ifft.process(&mut x_o, &mut x);
        let x: Vec<f64> = x.iter().map(|x| (*x).re / n as f64).collect();
        assert!(x[0] <= 1e15);
        assert!(x[9] - 9. <= std::f64::EPSILON);
    }

    // test whether the fft multiplication can be computed partially.
    #[test]
    fn fft_segmented() {
        let a = &[0., 10., 20., 30., 50., 60.];
        let b = &[2., 3.];
        let z1 = fft_mult(a, b);
        let l = 4;
        let z2 = fft_mult(&a[..l], b);
        for (a, b) in izip!(&z1[1..l], &z2[1..l]) {
            assert!(*a - *b < 1e-15);
        }
    }
}
