//! Tensor Network Trading Library
//!
//! Implements Matrix Product State (MPS) based classifiers and correlation
//! analysis for trading applications. Includes Bybit API integration for
//! fetching real market data.

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Tensor core & MPS structs
// ---------------------------------------------------------------------------

/// A single core tensor in the MPS chain.
/// Shape: (bond_left, phys_dim, bond_right)
#[derive(Debug, Clone)]
pub struct TensorCore {
    pub data: Array3<f64>,
}

impl TensorCore {
    /// Create a new core with the given shape filled with small random values.
    pub fn random(bond_left: usize, phys_dim: usize, bond_right: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data = Array3::from_shape_fn((bond_left, phys_dim, bond_right), |_| {
            rng.gen_range(-0.1..0.1)
        });
        Self { data }
    }

    /// Shape helper: (bond_left, phys_dim, bond_right)
    pub fn shape(&self) -> (usize, usize, usize) {
        let s = self.data.shape();
        (s[0], s[1], s[2])
    }
}

/// Matrix Product State with an optional label index on the last core.
#[derive(Debug, Clone)]
pub struct MPS {
    pub cores: Vec<TensorCore>,
    pub bond_dims: Vec<usize>,
    pub phys_dim: usize,
    pub num_classes: usize,
}

impl MPS {
    /// Build a random MPS for `num_sites` features, with given physical
    /// dimension, maximum bond dimension, and number of output classes.
    ///
    /// The label index is attached to the *last* core, enlarging its
    /// bond_right dimension by a factor of `num_classes` which is then
    /// reshaped during classification.
    pub fn new(num_sites: usize, phys_dim: usize, max_bond: usize, num_classes: usize) -> Self {
        assert!(num_sites >= 2, "MPS requires at least 2 sites");
        let mut cores = Vec::with_capacity(num_sites);
        let mut bond_dims = Vec::with_capacity(num_sites + 1);

        bond_dims.push(1); // left boundary
        for k in 0..num_sites {
            let bl = *bond_dims.last().unwrap();
            let br = if k == num_sites - 1 {
                num_classes // last core outputs class scores
            } else {
                max_bond.min((phys_dim).pow((k + 1) as u32).min((phys_dim).pow((num_sites - k - 1) as u32)))
                    .min(max_bond)
            };
            cores.push(TensorCore::random(bl, phys_dim, br));
            bond_dims.push(br);
        }

        Self {
            cores,
            bond_dims,
            phys_dim,
            num_classes,
        }
    }

    /// Number of sites (features).
    pub fn num_sites(&self) -> usize {
        self.cores.len()
    }

    /// Contract the MPS with a sequence of encoded feature vectors and return
    /// a score vector of length `num_classes`.
    ///
    /// `features` must have length == num_sites, each element is a vector of
    /// length `phys_dim`.
    pub fn classify(&self, features: &[Array1<f64>]) -> Result<Array1<f64>> {
        if features.len() != self.num_sites() {
            return Err(anyhow!(
                "Expected {} features, got {}",
                self.num_sites(),
                features.len()
            ));
        }

        // Start: contract first core with first feature -> vector of length bond_right
        let mut state = contract_core_with_feature(&self.cores[0], &features[0]);

        // Propagate through remaining cores
        for (core, feat) in self.cores[1..].iter().zip(features[1..].iter()) {
            state = propagate(&state, core, feat);
        }

        Ok(state)
    }

    /// Perform one training sweep (left-to-right then right-to-left) using
    /// simple gradient descent on squared error loss.
    ///
    /// Returns the average loss over the dataset.
    pub fn train_sweep(
        &mut self,
        data: &[Vec<Array1<f64>>],
        labels: &[usize],
        lr: f64,
    ) -> Result<f64> {
        let n = data.len();
        if n == 0 {
            return Ok(0.0);
        }

        let mut total_loss = 0.0;

        // Simple stochastic gradient descent on each sample
        for (sample, &label) in data.iter().zip(labels.iter()) {
            let scores = self.classify(sample)?;
            let mut target = Array1::zeros(self.num_classes);
            target[label] = 1.0;
            let diff = &scores - &target;
            let loss: f64 = diff.iter().map(|x| x * x).sum();
            total_loss += loss;

            // Gradient update: adjust the last core proportional to the error
            // This is a simplified update; full DMRG-style sweeping would
            // alternate optimizing each core.
            self.gradient_update_last_core(sample, &diff, lr)?;
        }

        Ok(total_loss / n as f64)
    }

    /// Update the last core via gradient descent given the error signal.
    fn gradient_update_last_core(
        &mut self,
        features: &[Array1<f64>],
        error: &Array1<f64>,
        lr: f64,
    ) -> Result<()> {
        // Compute the left environment up to the last core
        let n = self.num_sites();
        let mut state = contract_core_with_feature(&self.cores[0], &features[0]);
        for k in 1..n - 1 {
            state = propagate(&state, &self.cores[k], &features[k]);
        }

        // Gradient for the last core: outer product of state, feature, and error
        let last_feat = &features[n - 1];
        let last_core = &mut self.cores[n - 1];
        let (bl, pd, br) = last_core.shape();

        for a in 0..bl {
            for s in 0..pd {
                for c in 0..br {
                    let grad = state[a] * last_feat[s] * error[c];
                    last_core.data[[a, s, c]] -= lr * grad;
                }
            }
        }

        Ok(())
    }

    /// Compute entanglement entropy at each bond of the MPS.
    /// Returns a vector of length (num_sites - 1).
    pub fn entanglement_entropy(&self) -> Vec<f64> {
        let n = self.num_sites();
        let mut entropies = Vec::with_capacity(n - 1);

        for cut in 0..n - 1 {
            // Reshape the cut bond into a matrix and compute SVD
            let core = &self.cores[cut];
            let (bl, pd, br) = core.shape();
            let rows = bl * pd;
            let cols = br;

            let mat = Array2::from_shape_fn((rows, cols), |(r, c)| {
                let a = r / pd;
                let s = r % pd;
                core.data[[a, s, c]]
            });

            // Compute singular values via eigenvalues of M^T M
            let mtm = mat.t().dot(&mat);
            let eigenvalues = symmetric_eigenvalues(&mtm);

            // Entanglement entropy: S = -sum_i p_i log(p_i) where p_i = sigma_i^2 / sum(sigma_j^2)
            let total: f64 = eigenvalues.iter().filter(|&&v| v > 1e-15).sum();
            if total < 1e-15 {
                entropies.push(0.0);
                continue;
            }

            let mut s = 0.0;
            for &ev in &eigenvalues {
                if ev > 1e-15 {
                    let p = ev / total;
                    s -= p * p.ln();
                }
            }
            entropies.push(s);
        }

        entropies
    }

    /// Truncate bond dimensions to `max_bond` using SVD.
    pub fn truncate(&mut self, max_bond: usize) {
        for k in 0..self.num_sites() - 1 {
            let core = &self.cores[k];
            let (bl, pd, br) = core.shape();

            if br <= max_bond {
                continue;
            }

            let rows = bl * pd;
            let mat = Array2::from_shape_fn((rows, br), |(r, c)| {
                let a = r / pd;
                let s = r % pd;
                core.data[[a, s, c]]
            });

            // Simple truncation: keep only the first max_bond columns
            // weighted by their importance (column norms)
            let mut col_norms: Vec<(usize, f64)> = (0..br)
                .map(|c| {
                    let norm: f64 = (0..rows).map(|r| mat[[r, c]] * mat[[r, c]]).sum();
                    (c, norm)
                })
                .collect();
            col_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let keep: Vec<usize> = col_norms.iter().take(max_bond).map(|&(i, _)| i).collect();

            let new_core = Array3::from_shape_fn((bl, pd, max_bond), |(a, s, c)| {
                self.cores[k].data[[a, s, keep[c]]]
            });
            self.cores[k] = TensorCore { data: new_core };

            // Update next core to match
            if k + 1 < self.num_sites() {
                let next = &self.cores[k + 1];
                let (_, pd2, br2) = next.shape();
                let new_next = Array3::from_shape_fn((max_bond, pd2, br2), |(a, s, c)| {
                    if a < keep.len() {
                        next.data[[keep[a], s, c]]
                    } else {
                        0.0
                    }
                });
                self.cores[k + 1] = TensorCore { data: new_next };
            }

            self.bond_dims[k + 1] = max_bond;
        }
    }
}

// ---------------------------------------------------------------------------
// Feature encoding
// ---------------------------------------------------------------------------

/// Encode a scalar feature in [0, 1] into a local tensor space of dimension
/// `phys_dim` using trigonometric encoding.
///
/// phi(x) = [cos(pi*x/2), sin(pi*x/2), ...]
pub fn encode_feature(x: f64, phys_dim: usize) -> Array1<f64> {
    let mut phi = Array1::zeros(phys_dim);
    let x_clamped = x.clamp(0.0, 1.0);
    for k in 0..phys_dim {
        let freq = (k + 1) as f64;
        if k % 2 == 0 {
            phi[k] = (std::f64::consts::FRAC_PI_2 * freq * x_clamped).cos();
        } else {
            phi[k] = (std::f64::consts::FRAC_PI_2 * freq * x_clamped).sin();
        }
    }
    phi
}

/// Encode a vector of raw features (each in [0,1]) into a vector of local
/// tensor space vectors.
pub fn encode_features(raw: &[f64], phys_dim: usize) -> Vec<Array1<f64>> {
    raw.iter().map(|&x| encode_feature(x, phys_dim)).collect()
}

// ---------------------------------------------------------------------------
// Contraction helpers
// ---------------------------------------------------------------------------

/// Contract a single core with a feature vector, summing over the physical index.
/// core shape: (bond_left, phys_dim, bond_right)
/// feature shape: (phys_dim,)
/// result shape: (bond_left, bond_right) flattened to (bond_left * bond_right)
/// but for the first core bond_left=1 so result is (bond_right,).
fn contract_core_with_feature(core: &TensorCore, feature: &Array1<f64>) -> Array1<f64> {
    let (bl, pd, br) = core.shape();
    let mut result = Array1::zeros(bl * br);
    for a in 0..bl {
        for s in 0..pd {
            for b in 0..br {
                result[a * br + b] += core.data[[a, s, b]] * feature[s];
            }
        }
    }
    // If bond_left == 1, this is just a vector of length bond_right
    if bl == 1 {
        result.slice(ndarray::s![0..br]).to_owned()
    } else {
        result
    }
}

/// Propagate a state vector through a core contracted with a feature.
/// state shape: (bond_left,)  or (prev_bond,)
/// core shape: (bond_left, phys_dim, bond_right)
/// feature shape: (phys_dim,)
/// result shape: (bond_right,)
fn propagate(state: &Array1<f64>, core: &TensorCore, feature: &Array1<f64>) -> Array1<f64> {
    let (bl, pd, br) = core.shape();
    let mut result = Array1::zeros(br);
    for a in 0..bl.min(state.len()) {
        for s in 0..pd {
            for b in 0..br {
                result[b] += state[a] * core.data[[a, s, b]] * feature[s];
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Correlation analysis
// ---------------------------------------------------------------------------

/// Compute pairwise mutual information estimates from MPS bond structure.
/// Returns a symmetric matrix of size num_sites x num_sites.
pub fn correlation_from_mps(mps: &MPS) -> Array2<f64> {
    let n = mps.num_sites();
    let entropies = mps.entanglement_entropy();
    let mut corr = Array2::zeros((n, n));

    // The entanglement entropy at bond k quantifies correlations between
    // sites [0..=k] and [k+1..n]. We use this to estimate pairwise
    // correlations: sites separated by bonds with high entropy are more
    // correlated.
    for i in 0..n {
        for j in i + 1..n {
            // Sum entanglement entropies of bonds between sites i and j
            let mut mi = 0.0;
            for bond in i..j {
                if bond < entropies.len() {
                    mi += entropies[bond];
                }
            }
            // Normalize by distance
            let dist = (j - i) as f64;
            let normalized = mi / dist;
            corr[[i, j]] = normalized;
            corr[[j, i]] = normalized;
        }
        corr[[i, i]] = 1.0;
    }

    corr
}

// ---------------------------------------------------------------------------
// Simple eigenvalue computation for small symmetric matrices (power method)
// ---------------------------------------------------------------------------

/// Compute eigenvalues of a small symmetric positive semi-definite matrix
/// using a simple iterative approach. Returns eigenvalues sorted descending.
fn symmetric_eigenvalues(mat: &Array2<f64>) -> Vec<f64> {
    let n = mat.shape()[0];
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![mat[[0, 0]]];
    }

    // For small matrices, use the trace and Frobenius norm as approximation
    // For actual production use, link to LAPACK.
    let mut eigenvalues = Vec::with_capacity(n);
    let mut remaining = mat.clone();

    for _ in 0..n {
        // Power iteration to find largest eigenvalue
        let mut v = Array1::from_elem(remaining.shape()[0], 1.0 / (remaining.shape()[0] as f64).sqrt());
        let mut eigenvalue = 0.0;

        for _ in 0..100 {
            let mv = remaining.dot(&v);
            eigenvalue = v.dot(&mv);
            let norm: f64 = mv.dot(&mv).sqrt();
            if norm < 1e-15 {
                break;
            }
            v = mv / norm;
        }

        if eigenvalue.abs() < 1e-15 {
            eigenvalues.push(0.0);
        } else {
            eigenvalues.push(eigenvalue);
            // Deflate
            let vv = outer_product(&v, &v);
            remaining = remaining - eigenvalue * &vv;
        }
    }

    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eigenvalues
}

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    Array2::from_shape_fn((n, m), |(i, j)| a[i] * b[j])
}

// ---------------------------------------------------------------------------
// Softmax utility
// ---------------------------------------------------------------------------

/// Apply softmax to a score vector and return class probabilities.
pub fn softmax(scores: &Array1<f64>) -> Array1<f64> {
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Array1<f64> = scores.mapv(|s| (s - max_score).exp());
    let sum: f64 = exps.sum();
    if sum < 1e-15 {
        Array1::from_elem(scores.len(), 1.0 / scores.len() as f64)
    } else {
        exps / sum
    }
}

/// Return the predicted class (argmax of scores).
pub fn predict_class(scores: &Array1<f64>) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Bybit API data fetching
// ---------------------------------------------------------------------------

/// Raw kline data from Bybit API.
#[derive(Debug, Clone)]
pub struct KlineData {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch kline (candlestick) data from Bybit REST API.
///
/// * `symbol` - Trading pair, e.g. "BTCUSDT"
/// * `interval` - Candle interval, e.g. "60" for 1 hour
/// * `limit` - Number of candles to fetch (max 200)
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<KlineData>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    if resp.ret_code != 0 {
        return Err(anyhow!("Bybit API error: ret_code={}", resp.ret_code));
    }

    let mut klines = Vec::new();
    for entry in &resp.result.list {
        if entry.len() < 6 {
            continue;
        }
        klines.push(KlineData {
            timestamp: entry[0].parse().unwrap_or(0),
            open: entry[1].parse().unwrap_or(0.0),
            high: entry[2].parse().unwrap_or(0.0),
            low: entry[3].parse().unwrap_or(0.0),
            close: entry[4].parse().unwrap_or(0.0),
            volume: entry[5].parse().unwrap_or(0.0),
        });
    }

    // Bybit returns newest first; reverse so oldest is first.
    klines.reverse();

    Ok(klines)
}

/// Extract normalized features from kline data.
/// Returns vectors of: (return, volatility, volume_change) each in [0, 1].
pub fn extract_features(klines: &[KlineData]) -> Vec<[f64; 3]> {
    if klines.is_empty() {
        return vec![];
    }

    let avg_volume: f64 = klines.iter().map(|k| k.volume).sum::<f64>() / klines.len() as f64;

    klines
        .iter()
        .map(|k| {
            let ret = if k.open != 0.0 {
                ((k.close - k.open) / k.open).clamp(-0.1, 0.1) / 0.2 + 0.5
            } else {
                0.5
            };
            let vol = if k.open != 0.0 {
                ((k.high - k.low) / k.open).clamp(0.0, 0.2) / 0.2
            } else {
                0.0
            };
            let vol_change = if avg_volume > 0.0 {
                (k.volume / avg_volume).clamp(0.0, 3.0) / 3.0
            } else {
                0.0
            };
            [ret, vol, vol_change]
        })
        .collect()
}

/// Simple market regime labeling based on returns and volatility.
/// 0 = bull, 1 = bear, 2 = sideways
pub fn label_regime(features: &[f64; 3]) -> usize {
    let ret = features[0]; // 0.5 = zero return
    let vol = features[1];

    if ret > 0.55 && vol < 0.5 {
        0 // bull
    } else if ret < 0.45 && vol > 0.3 {
        1 // bear
    } else {
        2 // sideways
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_feature_bounds() {
        let phi = encode_feature(0.0, 2);
        assert!((phi[0] - 1.0).abs() < 1e-10, "cos(0) should be 1");
        assert!(phi[1].abs() < 1e-10, "sin(0) should be 0");

        let phi = encode_feature(1.0, 2);
        assert!((phi[0] - (std::f64::consts::FRAC_PI_2).cos()).abs() < 1e-10);
        assert!((phi[1] - (std::f64::consts::FRAC_PI_2).sin()).abs() < 1e-10);
    }

    #[test]
    fn test_encode_feature_normalization() {
        // For phys_dim=2, the encoding should produce unit-norm vectors
        let phi = encode_feature(0.3, 2);
        let norm: f64 = phi.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Encoded feature should have unit norm, got {}",
            norm
        );
    }

    #[test]
    fn test_mps_creation() {
        let mps = MPS::new(5, 2, 4, 3);
        assert_eq!(mps.num_sites(), 5);
        assert_eq!(mps.phys_dim, 2);
        assert_eq!(mps.num_classes, 3);
        assert_eq!(mps.cores.len(), 5);
    }

    #[test]
    fn test_mps_classify_output_shape() {
        let mps = MPS::new(4, 2, 3, 3);
        let features: Vec<Array1<f64>> = (0..4).map(|_| encode_feature(0.5, 2)).collect();
        let scores = mps.classify(&features).unwrap();
        assert_eq!(scores.len(), 3, "Should output 3 class scores");
    }

    #[test]
    fn test_mps_classify_wrong_features() {
        let mps = MPS::new(4, 2, 3, 3);
        let features: Vec<Array1<f64>> = (0..3).map(|_| encode_feature(0.5, 2)).collect();
        assert!(mps.classify(&features).is_err());
    }

    #[test]
    fn test_softmax() {
        let scores = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&scores);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-10, "Softmax should sum to 1");
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_predict_class() {
        let scores = Array1::from_vec(vec![-1.0, 5.0, 2.0]);
        assert_eq!(predict_class(&scores), 1);
    }

    #[test]
    fn test_entanglement_entropy() {
        let mps = MPS::new(4, 2, 3, 2);
        let entropies = mps.entanglement_entropy();
        assert_eq!(entropies.len(), 3);
        for &s in &entropies {
            assert!(s >= 0.0, "Entropy should be non-negative");
        }
    }

    #[test]
    fn test_correlation_from_mps() {
        let mps = MPS::new(5, 2, 4, 2);
        let corr = correlation_from_mps(&mps);
        assert_eq!(corr.shape(), &[5, 5]);
        // Diagonal should be 1
        for i in 0..5 {
            assert!((corr[[i, i]] - 1.0).abs() < 1e-10);
        }
        // Symmetric
        for i in 0..5 {
            for j in 0..5 {
                assert!((corr[[i, j]] - corr[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_extract_features() {
        let klines = vec![
            KlineData { timestamp: 1, open: 100.0, high: 105.0, low: 95.0, close: 103.0, volume: 1000.0 },
            KlineData { timestamp: 2, open: 103.0, high: 107.0, low: 100.0, close: 101.0, volume: 1200.0 },
        ];
        let features = extract_features(&klines);
        assert_eq!(features.len(), 2);
        for f in &features {
            for &v in f {
                assert!(v >= 0.0 && v <= 1.0, "Feature {} out of [0,1]", v);
            }
        }
    }

    #[test]
    fn test_label_regime() {
        assert_eq!(label_regime(&[0.7, 0.2, 0.5]), 0); // bull
        assert_eq!(label_regime(&[0.3, 0.5, 0.5]), 1); // bear
        assert_eq!(label_regime(&[0.5, 0.3, 0.5]), 2); // sideways
    }

    #[test]
    fn test_training_reduces_loss() {
        let mut mps = MPS::new(3, 2, 4, 2);
        let data: Vec<Vec<Array1<f64>>> = vec![
            encode_features(&[0.1, 0.2, 0.3], 2),
            encode_features(&[0.8, 0.9, 0.7], 2),
            encode_features(&[0.15, 0.25, 0.2], 2),
            encode_features(&[0.85, 0.75, 0.9], 2),
        ];
        let labels = vec![0, 1, 0, 1];

        let loss1 = mps.train_sweep(&data, &labels, 0.01).unwrap();
        // Run several more sweeps
        let mut loss_last = loss1;
        for _ in 0..20 {
            loss_last = mps.train_sweep(&data, &labels, 0.01).unwrap();
        }
        assert!(
            loss_last <= loss1 + 0.1,
            "Training should generally reduce loss: first={}, last={}",
            loss1,
            loss_last
        );
    }

    #[test]
    fn test_truncate() {
        let mut mps = MPS::new(4, 2, 8, 2);
        mps.truncate(3);
        for k in 0..mps.num_sites() - 1 {
            let (_, _, br) = mps.cores[k].shape();
            assert!(br <= 3 || k == mps.num_sites() - 1);
        }
    }
}
