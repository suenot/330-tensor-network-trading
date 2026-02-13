//! Trading Example: Tensor Network Market Regime Detection
//!
//! This example demonstrates:
//! 1. Fetching multiple crypto assets from Bybit
//! 2. Encoding price features into tensor network representation
//! 3. Training an MPS classifier for market regime detection
//! 4. Analyzing inter-asset correlations captured by bond dimensions

use anyhow::Result;
use tensor_network_trading::*;

/// Assets to analyze
const SYMBOLS: &[&str] = &["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"];

/// Number of candles to fetch per asset
const NUM_CANDLES: usize = 100;

/// Physical dimension for feature encoding
const PHYS_DIM: usize = 2;

/// Maximum bond dimension
const MAX_BOND: usize = 4;

/// Number of market regime classes (bull, bear, sideways)
const NUM_CLASSES: usize = 3;

/// Number of training sweeps
const NUM_SWEEPS: usize = 50;

/// Learning rate
const LEARNING_RATE: f64 = 0.005;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Tensor Network Trading: Market Regime Detection ===\n");

    // Step 1: Fetch data from Bybit
    println!("Step 1: Fetching market data from Bybit...");
    let mut all_features = Vec::new();
    let mut asset_names = Vec::new();

    for &symbol in SYMBOLS {
        match fetch_bybit_klines(symbol, "60", NUM_CANDLES).await {
            Ok(klines) => {
                let features = extract_features(&klines);
                println!(
                    "  {} - {} candles fetched, {} features extracted",
                    symbol,
                    klines.len(),
                    features.len()
                );
                all_features.push((symbol.to_string(), features));
                asset_names.push(symbol.to_string());
            }
            Err(e) => {
                println!("  {} - Error fetching data: {}. Using synthetic data.", symbol, e);
                // Generate synthetic data for demonstration
                let features = generate_synthetic_features(NUM_CANDLES);
                all_features.push((symbol.to_string(), features));
                asset_names.push(symbol.to_string());
            }
        }
    }

    // Step 2: Prepare training data
    println!("\nStep 2: Preparing training data...");
    let num_assets = all_features.len();
    let num_features_per_sample = num_assets * 3; // 3 features per asset (return, vol, vol_change)

    // Find minimum number of candles across all assets
    let min_candles = all_features
        .iter()
        .map(|(_, f)| f.len())
        .min()
        .unwrap_or(0);

    if min_candles < 10 {
        println!("Not enough data for training. Need at least 10 candles.");
        return Ok(());
    }

    let mut training_data: Vec<Vec<ndarray::Array1<f64>>> = Vec::new();
    let mut training_labels: Vec<usize> = Vec::new();

    for t in 0..min_candles {
        // Concatenate features from all assets at time t
        let mut raw_features = Vec::new();
        let mut primary_features = [0.5, 0.0, 0.0]; // features of first asset for labeling

        for (i, (_, features)) in all_features.iter().enumerate() {
            let f = &features[t];
            raw_features.push(f[0]); // return
            raw_features.push(f[1]); // volatility
            raw_features.push(f[2]); // volume change
            if i == 0 {
                primary_features = *f;
            }
        }

        let encoded = encode_features(&raw_features, PHYS_DIM);
        let label = label_regime(&primary_features);
        training_data.push(encoded);
        training_labels.push(label);
    }

    let regime_counts = [
        training_labels.iter().filter(|&&l| l == 0).count(),
        training_labels.iter().filter(|&&l| l == 1).count(),
        training_labels.iter().filter(|&&l| l == 2).count(),
    ];
    println!(
        "  Training samples: {} (Bull: {}, Bear: {}, Sideways: {})",
        training_data.len(),
        regime_counts[0],
        regime_counts[1],
        regime_counts[2]
    );
    println!("  Features per sample: {}", num_features_per_sample);

    // Step 3: Create and train MPS classifier
    println!("\nStep 3: Training MPS classifier...");
    println!(
        "  Sites: {}, Physical dim: {}, Max bond: {}, Classes: {}",
        num_features_per_sample, PHYS_DIM, MAX_BOND, NUM_CLASSES
    );

    let mut mps = MPS::new(num_features_per_sample, PHYS_DIM, MAX_BOND, NUM_CLASSES);

    for sweep in 0..NUM_SWEEPS {
        let loss = mps.train_sweep(&training_data, &training_labels, LEARNING_RATE)?;
        if sweep % 10 == 0 || sweep == NUM_SWEEPS - 1 {
            println!("  Sweep {}/{}: loss = {:.6}", sweep + 1, NUM_SWEEPS, loss);
        }
    }

    // Step 4: Evaluate classifier
    println!("\nStep 4: Evaluating classifier...");
    let mut correct = 0;
    let mut total = 0;
    let mut class_correct = [0usize; 3];
    let mut class_total = [0usize; 3];

    for (sample, &label) in training_data.iter().zip(training_labels.iter()) {
        let scores = mps.classify(sample)?;
        let probs = softmax(&scores);
        let predicted = predict_class(&scores);

        class_total[label] += 1;
        total += 1;
        if predicted == label {
            correct += 1;
            class_correct[label] += 1;
        }
    }

    println!(
        "  Overall accuracy: {}/{} ({:.1}%)",
        correct,
        total,
        100.0 * correct as f64 / total as f64
    );
    for (c, name) in ["Bull", "Bear", "Sideways"].iter().enumerate() {
        if class_total[c] > 0 {
            println!(
                "  {} accuracy: {}/{} ({:.1}%)",
                name,
                class_correct[c],
                class_total[c],
                100.0 * class_correct[c] as f64 / class_total[c] as f64
            );
        }
    }

    // Step 5: Analyze correlations via entanglement entropy
    println!("\nStep 5: Analyzing inter-asset correlations via entanglement entropy...");
    let entropies = mps.entanglement_entropy();

    println!("  Bond entanglement entropies:");
    for (i, &s) in entropies.iter().enumerate() {
        // Map bond index back to asset
        let asset_idx = i / 3;
        let feature_idx = i % 3;
        let feature_name = match feature_idx {
            0 => "return",
            1 => "volatility",
            _ => "volume",
        };
        let asset = if asset_idx < asset_names.len() {
            &asset_names[asset_idx]
        } else {
            "unknown"
        };
        println!(
            "    Bond {}: S = {:.4} ({} - {})",
            i, s, asset, feature_name
        );
    }

    // Correlation matrix between feature groups
    let corr = correlation_from_mps(&mps);
    println!("\n  Correlation structure (feature-group level):");
    println!("  Features are grouped by asset: 3 per asset (return, vol, vol_change)");

    // Print asset-level correlation summary
    for i in 0..num_assets {
        for j in i + 1..num_assets {
            let mut avg_corr = 0.0;
            let mut count = 0;
            for fi in 0..3 {
                for fj in 0..3 {
                    avg_corr += corr[[i * 3 + fi, j * 3 + fj]];
                    count += 1;
                }
            }
            avg_corr /= count as f64;
            println!(
                "    {} <-> {}: avg correlation = {:.4}",
                asset_names[i], asset_names[j], avg_corr
            );
        }
    }

    // Step 6: Predict current regime
    println!("\nStep 6: Current regime prediction...");
    if let Some(last_sample) = training_data.last() {
        let scores = mps.classify(last_sample)?;
        let probs = softmax(&scores);
        let predicted = predict_class(&scores);
        let regime_name = match predicted {
            0 => "BULL",
            1 => "BEAR",
            _ => "SIDEWAYS",
        };
        println!("  Predicted regime: {}", regime_name);
        println!(
            "  Probabilities: Bull={:.3}, Bear={:.3}, Sideways={:.3}",
            probs[0], probs[1], probs[2]
        );
    }

    println!("\n=== Done ===");
    Ok(())
}

/// Generate synthetic market features for demonstration when API is unavailable.
fn generate_synthetic_features(n: usize) -> Vec<[f64; 3]> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut features = Vec::with_capacity(n);

    let mut regime = 0; // Start with bull
    let mut regime_counter = 0;

    for _ in 0..n {
        // Switch regime every ~20 candles
        regime_counter += 1;
        if regime_counter > 15 + rng.gen_range(0..10) {
            regime = rng.gen_range(0..3);
            regime_counter = 0;
        }

        let (ret, vol, vol_chg) = match regime {
            0 => (
                // Bull: positive returns, low vol
                0.5 + rng.gen_range(0.05..0.15),
                rng.gen_range(0.1..0.3),
                rng.gen_range(0.3..0.6),
            ),
            1 => (
                // Bear: negative returns, high vol
                0.5 - rng.gen_range(0.05..0.15),
                rng.gen_range(0.4..0.7),
                rng.gen_range(0.5..0.8),
            ),
            _ => (
                // Sideways: near-zero returns, moderate vol
                0.5 + rng.gen_range(-0.04..0.04),
                rng.gen_range(0.2..0.4),
                rng.gen_range(0.2..0.5),
            ),
        };

        features.push([ret, vol, vol_chg]);
    }

    features
}
