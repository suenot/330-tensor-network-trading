# Chapter 200: Tensor Network Trading

## 1. Introduction

Tensor networks (TN) have emerged from quantum physics as one of the most powerful mathematical frameworks for representing and manipulating high-dimensional data. In condensed matter physics, they enabled breakthroughs in simulating quantum many-body systems that were previously intractable. Now, the same mathematical machinery is finding its way into machine learning and, specifically, into quantitative trading.

The core insight is simple yet profound: real-world data, despite living in exponentially large spaces, often has structure that can be captured by tensor networks with far fewer parameters than a naive representation would require. In trading, we deal with precisely this kind of high-dimensional structured data: hundreds of correlated assets, each with multiple features evolving over time. A covariance matrix for 500 assets already has 125,000 entries, and higher-order correlations explode combinatorially. Tensor networks provide a principled way to compress these structures while preserving the essential information.

This chapter introduces tensor networks from the ground up, connects them to machine learning, and demonstrates their application to trading problems including market regime classification, correlation modeling, and portfolio construction. All implementations are provided in Rust for production-grade performance, with data sourced from the Bybit exchange API.

## 2. Mathematical Foundation

### Tensors

A tensor is a multi-dimensional array of numbers. A scalar is a 0th-order tensor, a vector is 1st-order, a matrix is 2nd-order, and so on. An N-th order tensor T with dimensions d_1 x d_2 x ... x d_N has entries T(i_1, i_2, ..., i_N) where each index i_k ranges from 1 to d_k.

The total number of entries grows exponentially with the order N. For instance, if each dimension is d = 10 and N = 20, the tensor has 10^20 entries — clearly impossible to store or manipulate directly. This is the "curse of dimensionality" that tensor networks are designed to overcome.

### Tensor Decomposition

Tensor decomposition factorizes a high-order tensor into a network of lower-order tensors. The most important decompositions for our purposes are:

**CP Decomposition (CANDECOMP/PARAFAC):** Expresses a tensor as a sum of rank-one tensors:

```
T(i_1, ..., i_N) = sum_r A_1(i_1, r) * A_2(i_2, r) * ... * A_N(i_N, r)
```

where r ranges from 1 to R (the rank). This requires only N * d * R parameters instead of d^N.

**Tucker Decomposition:** Generalizes CP by introducing a core tensor G:

```
T(i_1, ..., i_N) = sum_{r_1,...,r_N} G(r_1,...,r_N) * U_1(i_1, r_1) * ... * U_N(i_N, r_N)
```

### Matrix Product States (MPS) / Tensor Train (TT)

The Matrix Product State (MPS), also known as the Tensor Train (TT) decomposition, is the most widely used tensor network format. It represents a tensor as a chain of 3rd-order tensors:

```
T(i_1, i_2, ..., i_N) = sum_{a_1,...,a_{N-1}} A_1(i_1, a_1) * A_2(a_1, i_2, a_2) * ... * A_N(a_{N-1}, i_N)
```

Here, each A_k is a 3-index tensor (a "core") with:
- One "physical" index i_k of dimension d_k (the actual data dimension)
- Two "bond" indices a_{k-1} and a_k of dimensions r_{k-1} and r_k (the internal dimensions)

The boundary cores A_1 and A_N have only two indices (one bond index is trivially 1).

The total number of parameters is approximately N * d * r^2, where r is the maximum bond dimension. This is linear in N rather than exponential — a dramatic compression.

### Bond Dimension

The bond dimension r controls the expressiveness of the MPS. A larger bond dimension can capture more correlations between sites but requires more memory and computation:

- **r = 1:** The tensor is a product state with no correlations between sites.
- **r = d^(N/2):** The MPS can represent any tensor exactly (no compression).
- **Intermediate r:** Captures the most important correlations while compressing.

The optimal bond dimension depends on the entanglement structure of the data. This is where the connection to quantum physics becomes concrete.

### Entanglement Entropy

Given an MPS, we can perform a bipartition at any bond and compute the entanglement entropy S via the singular values of the corresponding matrix:

```
S = -sum_i lambda_i^2 * log(lambda_i^2)
```

where lambda_i are the singular values. The entanglement entropy quantifies how much correlation exists across the bipartition. If S is small, a small bond dimension suffices. This gives us a principled way to choose bond dimensions adaptively.

In trading, the entanglement entropy at different bipartitions reveals which groups of assets are most correlated — providing insights beyond pairwise correlation matrices.

## 3. TN for Machine Learning

### TN as Classifiers

Stoudenmire and Schwab (2016) showed that tensor networks can serve as powerful supervised classifiers. The idea is elegant:

1. **Feature encoding:** Map each input feature x_j to a local d-dimensional vector via a feature map phi(x_j). A common choice is:

```
phi(x) = [cos(pi*x/2), sin(pi*x/2)]
```

This maps a scalar feature into a 2-dimensional local Hilbert space.

2. **Tensor product:** The full feature vector is the tensor product of all local feature vectors:

```
Phi(x) = phi(x_1) tensor phi(x_2) tensor ... tensor phi(x_N)
```

This lives in a d^N-dimensional space but is structured as a product state.

3. **Classification:** The classifier is an MPS W that contracts with the feature vector to produce a label:

```
f(x) = W . Phi(x)
```

where the dot denotes full tensor contraction. The MPS W has an additional label index that gives one output per class.

4. **Training:** Optimize W by sweeping through the MPS cores one at a time, solving local optimization problems. This is analogous to the DMRG algorithm in physics.

### Connection to Quantum Circuits

There is a deep connection between tensor networks and quantum computing. An MPS with bond dimension r can be prepared by a quantum circuit of depth O(N * log(r)). Conversely, the output of a quantum circuit with bounded entanglement can be efficiently represented as an MPS. This means tensor network classifiers can be thought of as "classical simulations" of certain quantum machine learning models.

### Advantages over Traditional ML

Tensor network classifiers offer several unique advantages:

- **Interpretability:** The bond dimension at each cut reveals how much information flows between groups of features.
- **Controllable complexity:** The bond dimension directly controls the model capacity.
- **No barren plateaus:** Unlike deep neural networks, the optimization landscape of MPS is relatively smooth.
- **Guaranteed convergence:** The sweeping optimization is related to alternating least squares, which has well-understood convergence properties.

## 4. Trading Applications

### Modeling Correlations Between Many Assets

Traditional covariance matrices capture pairwise correlations but miss higher-order dependencies. For example, three assets may have near-zero pairwise correlations but strong three-way interactions (e.g., a triangular arbitrage relationship). Tensor networks naturally capture these higher-order correlations.

An MPS representation of the joint probability distribution of N asset returns:

```
P(r_1, r_2, ..., r_N) = ||MPS(r_1, r_2, ..., r_N)||^2
```

captures correlations up to order r^2, where r is the bond dimension. By examining the entanglement entropy at each bond, we can identify clusters of highly correlated assets.

### Compressing Large Covariance Structures

For a universe of N assets with T time steps, the full correlation tensor has N^2 * T^2 entries for pairwise time-lagged correlations alone. An MPS compression reduces this to O(N * r^2 * d) parameters, making it feasible to model thousands of assets with complex temporal dependencies.

The key insight is that financial correlations are often approximately low-rank. The leading singular values of empirical covariance matrices decay rapidly, suggesting that a modest bond dimension captures most of the relevant structure. This is analogous to the area law of entanglement entropy in physics: most physically relevant states have low entanglement.

### Time Series Modeling with MPS

An MPS can model a time series by treating each time step as a "site" in the network. The bond dimension then controls how much memory the model has of past events:

- **r = 1:** Memoryless (Markov) model.
- **r > 1:** The model captures temporal correlations up to length O(log(r)) for exponentially decaying correlations, or longer for algebraically decaying ones.

This provides a natural framework for market regime detection: different regimes correspond to different MPS representations, and transitions between regimes manifest as changes in the bond dimension structure.

### Market Regime Detection

We can train an MPS classifier to distinguish between market regimes:

1. **Bull regime:** Characterized by positive drift, low volatility, positive autocorrelation.
2. **Bear regime:** Negative drift, high volatility, volatility clustering.
3. **Sideways regime:** Near-zero drift, moderate volatility, mean reversion.

The MPS classifier encodes price features (returns, volatility, volume changes) into local tensor spaces and learns the correlations that distinguish regimes. The bond dimension reveals which features carry the most information about regimes.

### Portfolio Construction

Tensor networks enable a novel approach to portfolio construction:

1. Model the joint return distribution as an MPS.
2. Compute marginal and conditional distributions efficiently via partial contraction.
3. Optimize portfolio weights by maximizing expected utility subject to the MPS-encoded distribution.

This approach naturally handles non-Gaussian distributions and higher-order moments without the exponential cost of full moment tensors.

## 5. TN vs Deep Learning

Tensor networks and deep neural networks have complementary strengths:

| Aspect | Tensor Networks | Deep Learning |
|--------|----------------|---------------|
| **Parameters** | Controlled by bond dimension | Controlled by architecture |
| **Interpretability** | High (entanglement structure) | Low (black box) |
| **Training** | Sweeping (convex subproblems) | Gradient descent (non-convex) |
| **Data efficiency** | Often better with small data | Requires large datasets |
| **Higher-order correlations** | Explicit by construction | Implicit in layers |
| **Computational cost** | O(N * r^3 * d) per sweep | Varies widely |
| **Overfitting risk** | Controlled by bond dimension | Requires regularization |

Tensor networks tend to outperform deep learning when:

- The number of features is moderate (10-100) but correlations are complex.
- Training data is limited (hundreds to low thousands of samples).
- Interpretability is important (understanding which feature correlations drive predictions).
- The data has a natural one-dimensional ordering (time series, ordered assets).

Deep learning tends to win when:

- Massive datasets are available.
- The features have spatial or hierarchical structure (images, text).
- The task requires learning highly nonlinear transformations.

In practice, hybrid approaches can combine the strengths of both: use tensor networks for feature extraction and correlation modeling, then feed the compressed representations into neural networks for final prediction.

## 6. Implementation Walkthrough

Our Rust implementation provides a complete tensor network trading system. The core components are:

### MPS Core Structure

Each core in the MPS is a 3-index tensor stored as a flattened array with dimensions (bond_left, physical, bond_right). The `MPSCore` struct manages the core tensors and provides methods for contraction, SVD truncation, and feature encoding.

```rust
pub struct TensorCore {
    pub data: Array3<f64>,
}

pub struct MPS {
    pub cores: Vec<TensorCore>,
    pub bond_dims: Vec<usize>,
    pub phys_dim: usize,
}
```

### Feature Encoding

Features are encoded into local tensor spaces using a trigonometric feature map:

```rust
pub fn encode_feature(x: f64, phys_dim: usize) -> Array1<f64> {
    let mut phi = Array1::zeros(phys_dim);
    phi[0] = (std::f64::consts::FRAC_PI_2 * x).cos();
    phi[1] = (std::f64::consts::FRAC_PI_2 * x).sin();
    phi
}
```

### Contraction and Classification

The MPS contracts with encoded features by sequentially multiplying through the chain:

```rust
// Start with the first core contracted with the first feature
// Then propagate through the chain, contracting each core with its feature
// The final result is a vector over the label index
```

### SVD Truncation

Bond dimensions are controlled via SVD truncation after each update step. The truncation error provides a natural measure of information loss.

### Training

Training proceeds by sweeping through the MPS cores, optimizing each core while holding the others fixed. Each local optimization is a linear least-squares problem, which can be solved exactly.

The full implementation is in `rust/src/lib.rs` with a complete trading example in `rust/examples/trading_example.rs`.

## 7. Bybit Data Integration

The implementation fetches real market data from the Bybit API:

```rust
pub async fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize)
    -> Result<Vec<KlineData>>
```

This function retrieves OHLCV (Open, High, Low, Close, Volume) data for any trading pair available on Bybit. The data is then processed into features suitable for tensor network encoding:

- **Normalized returns:** (close - open) / open, scaled to [0, 1].
- **Volatility proxy:** (high - low) / open, scaled to [0, 1].
- **Volume change:** Ratio of current to average volume, scaled to [0, 1].

Multiple assets can be fetched in parallel and their features concatenated to form the input for the MPS classifier or correlation analysis.

The Bybit REST API endpoint used is:

```
GET https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={interval}&limit={limit}
```

No authentication is required for public market data endpoints.

## 8. Key Takeaways

1. **Tensor networks provide exponential compression** of high-dimensional data by exploiting low-rank structure. For N assets with d features each, an MPS reduces the parameter count from d^N to O(N * r^2 * d).

2. **Bond dimension is the key hyperparameter.** It controls the trade-off between expressiveness and efficiency. The entanglement entropy provides a principled way to choose it.

3. **MPS classifiers are interpretable.** The bond structure reveals which features and which groups of assets carry the most predictive information about market regimes.

4. **Higher-order correlations are captured naturally.** Unlike covariance matrices, tensor networks model interactions among arbitrary subsets of assets.

5. **Training is robust.** The sweeping optimization avoids many pitfalls of gradient-based training (vanishing gradients, local minima, barren plateaus).

6. **Tensor networks complement deep learning.** They excel in low-data regimes, with moderate feature counts, and when interpretability matters. Hybrid approaches can leverage both.

7. **Rust implementation ensures production readiness.** The combination of performance, memory safety, and ergonomic error handling makes Rust ideal for tensor network computations in trading systems.

8. **Bybit integration provides real market data.** The implementation demonstrates end-to-end workflow from data fetching to regime classification and correlation analysis.
