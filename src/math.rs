use itertools::izip;

pub fn argmin<T: PartialOrd + Copy>(values: &[T]) -> usize {
    let mut index = 0;
    let (first, rest) = values.split_first().unwrap();
    let mut current = *first;
    // i starts at 0 despite having taken the first value already
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
    let n_x = x_len as f64;
    let n_y = y_len as f64;
    let start = y_len - 1;
    let end = x_len;
    let z_clipped = &z[start..end];
    let mu_x_clipped = &mu_x[start..end];
    let sigma_x_clipped = &sigma_x[start..end];
    let k = n_y * mu_q;

    let vars = izip!(mu_x_clipped, sigma_x_clipped, z_clipped);
    // div z by 1/n_x to compensate for it missing after ifft

    // let f = |(m, s, z): (&f64, &f64, &f64)| -> f64 {
    //     (2.0 * (n_y - ((*z) / n_x - k * (*m)) / (sigma_q * (*s)))).sqrt()
    // };

    // faster with same opt. goal $0.5dist^2$
    let f =
        |(m, s, z): (&f64, &f64, &f64)| -> f64 { n_y - ((*z) / n_x - k * (*m)) / (sigma_q * (*s)) };

    vars.map(f).collect()
}
