# How I'd Become a Quant If I Had to Start Over Tomorrow

### The Complete Technical Roadmap to Breaking Into Quantitative Finance in 2026

**Career Focus**: Quantitative Trading, Research, and Development  
**Target Firms**: Jane Street, Citadel, Two Sigma, Hudson River Trading, DE Shaw, Optiver  
**Expected Timeline**: 2-4 years of focused preparation  
**Entry Compensation**: $300K-$500K total comp (2026 market rates)  
**Analysis Date**: March 2026

---

## Executive Summary

**Quantitative finance** represents the intersection of **mathematics, programming, and financial markets**, where practitioners build algorithmic trading systems, risk models, and derivative pricing engines that move billions of dollars daily. The field offers some of the highest compensation in technology ($300K-$500K starting, $1M-$5M at senior levels) but demands **elite-level proficiency** across multiple technical domains.

This report provides a **comprehensive, actionable roadmap** for breaking into quantitative finance, covering:

1. **Career taxonomy** (What are the different quant roles?)
2. **Technical foundations** (What skills do you actually need?)
3. **Education pathways** (Which degrees and programs matter?)
4. **Interview preparation** (How to pass the brutal interview process)
5. **Day-to-day reality** (What does a quant actually do?)
6. **Long-term career progression** (Where can you go?)

**Target audience**: STEM undergrads, PhD students, software engineers, and career switchers considering quantitative finance.

**Core thesis**: **Becoming a quant is a structured, learnable process. With the right roadmap, deliberate practice, and 2-4 years of focused preparation, any technically strong individual can break into the field.**

---

## Part I: Understanding the Quant Landscape

### 1.1 What Is a Quant? (Role Taxonomy)

"Quant" is an umbrella term covering multiple distinct roles:

#### **A. Quantitative Researcher (Quant Research, QR)**

**Primary responsibility**: Develop **trading strategies** using statistical models and machine learning.

**Daily work**:
```python
# Example quant research workflow

# 1. Hypothesis generation
hypothesis = "Stocks with high analyst upgrade momentum outperform next week"

# 2. Data acquisition
data = fetch_market_data(
    symbols=SP500_tickers,
    fields=['price', 'volume', 'analyst_ratings', 'fundamentals'],
    start='2010-01-01',
    end='2025-12-31'
)

# 3. Feature engineering
data['upgrade_momentum'] = data['analyst_ratings'].diff(5)  # 5-day change
data['forward_return'] = data['price'].shift(-5) / data['price'] - 1  # 5-day forward return

# 4. Statistical testing
from scipy.stats import pearsonr
correlation, p_value = pearsonr(data['upgrade_momentum'], data['forward_return'])

if p_value < 0.01 and correlation > 0.05:
    print(f"Signal is significant: r={correlation:.3f}, p={p_value:.4f}")
    proceed_to_backtesting()

# 5. Backtesting
strategy = MomentumStrategy(lookback=5, holding_period=5)
results = backtest(strategy, data, transaction_costs=0.0010)

# 6. Risk analysis
sharpe_ratio = results['returns'].mean() / results['returns'].std() * np.sqrt(252)
max_drawdown = compute_max_drawdown(results['equity_curve'])

# 7. Productionization (if profitable after slippage, costs, capacity analysis)
if sharpe_ratio > 2.0 and max_drawdown < 0.15:
    deploy_to_production(strategy)
```

**Skills required**:
- Advanced probability & statistics (PhD-level)
- Time series analysis
- Machine learning (especially classical methods: linear models, trees, ensembles)
- Python (pandas, numpy, scikit-learn)
- Finance domain knowledge

**Typical background**: PhD in statistics, physics, math, CS, or quantitative finance

**Compensation** (2026): $300K-$500K starting, $1M-$5M at senior levels

---

#### **B. Quantitative Developer (Quant Dev, QD)**

**Primary responsibility**: Build **trading infrastructure** and optimize execution systems for low latency and high throughput.

**Daily work**:
```cpp
// Example quant developer work: optimizing order execution engine

class LowLatencyOrderRouter {
    /**
     * Ultra-low-latency order routing system.
     * 
     * Requirements:
     * - <50 microseconds from signal to exchange
     * - Zero dynamic memory allocation (all pre-allocated)
     * - Lock-free concurrent data structures
     * - Deterministic worst-case latency
     */
    
public:
    // Pre-allocated order pool (avoid malloc during trading)
    OrderPool order_pool_;
    
    // Lock-free queue for order submission (SPSC: single producer, single consumer)
    boost::lockfree::spsc_queue<Order*> order_queue_;
    
    // FIX protocol engine for exchange communication
    FIXEngine fix_engine_;
    
    void on_trading_signal(const Signal& signal) {
        // Clock the entire pipeline
        auto start = rdtsc();  // Read CPU timestamp counter
        
        // 1. Risk checks (pre-trade validation)
        if (!risk_manager_.check(signal)) {
            return;  // Reject if exceeds risk limits
        }
        
        // 2. Allocate order from pool (no malloc)
        Order* order = order_pool_.acquire();
        order->symbol = signal.symbol;
        order->quantity = signal.target_position - current_position_;
        order->price = calculate_limit_price(signal);
        order->timestamp_ns = std::chrono::high_resolution_clock::now();
        
        // 3. Submit to FIX engine (non-blocking)
        order_queue_.push(order);
        
        // 4. Measure latency
        auto end = rdtsc();
        uint64_t latency_cycles = end - start;
        uint64_t latency_ns = cycles_to_nanoseconds(latency_cycles);
        
        // Log if latency exceeds budget
        if (latency_ns > 50'000) {  // 50 microseconds
            LOG_WARNING("Latency budget exceeded: " << latency_ns << "ns");
        }
    }
    
    // Separate thread: send orders to exchange
    void order_submission_thread() {
        while (true) {
            Order* order;
            if (order_queue_.pop(order)) {
                fix_engine_.send_order(order);
                order_pool_.release(order);  // Return to pool
            }
        }
    }
};
```

**Skills required**:
- Systems programming (C++, kernel optimization)
- Computer architecture (CPU caching, NUMA, PCIe)
- Network programming (UDP multicast, kernel bypass)
- Data structures & algorithms
- Financial protocols (FIX, ITCH, OUCH)

**Typical background**: CS undergrad or Master's, strong systems programming experience

**Compensation** (2026): $250K-$400K starting, $800K-$2M at senior levels

---

#### **C. Quantitative Trader (Discretionary Quant)**

**Primary responsibility**: Execute trades based on quantitative models while applying human judgment for risk management and portfolio construction.

**Daily work**:
```python
# Morning routine: 6:30 AM - 9:30 AM (pre-market)

# 1. Review overnight PnL
pnl_report = fetch_pnl(start=yesterday_close, end=today_open)
print(f"Overnight PnL: ${pnl_report['total']:,.0f}")

# 2. Check model signals
signals = strategy_engine.generate_signals()
print(f"Active signals: {len(signals)} (Long: {signals.count('BUY')}, Short: {signals.count('SELL')})")

# 3. Risk analysis
current_exposure = portfolio.get_exposure()
print(f"Current exposure: ${current_exposure:,.0f} ({current_exposure / capital * 100:.1f}% of capital)")

# 4. Market context (read news, check macro events)
news = fetch_breaking_news()
macro = fetch_economic_calendar()

# Decision: adjust model signals based on discretion
if "Fed rate decision" in macro:
    signals = reduce_leverage(signals, factor=0.5)  # Cut risk by 50%

# 5. Place orders (9:30 AM market open)
for signal in signals:
    order = create_order(signal)
    if risk_manager.approve(order):
        send_to_exchange(order)

# Intraday: monitor positions, adjust as needed
# 4:00 PM: market close, review daily PnL
# 5:00 PM - 7:00 PM: research, strategy improvement, data analysis
```

**Skills required**:
- All quant researcher skills (math, programming, statistics)
- Market intuition (understanding order flow, liquidity, volatility)
- Risk management discipline
- Emotional control (dealing with losses, drawdowns)

**Typical background**: Often promoted from quant researcher after 3-5 years

**Compensation** (2026): $400K-$1M base + performance bonus (can exceed 100% of base)

---

### 1.2 Firm Taxonomy (Where Do Quants Work?)

| **Firm Type** | **Examples** | **Focus** | **Comp Range** | **Hiring Volume** |
|---------------|--------------|-----------|----------------|-------------------|
| **Prop Trading** | Jane Street, SIG, Optiver | Market making, HFT | $300K-$500K start | High (100+ grads/year) |
| **Hedge Funds** | Citadel, Two Sigma, DE Shaw | Multi-strategy, quant equity | $250K-$400K start | Medium (20-50/year) |
| **HFT Firms** | HRT, Jump Trading, Tower Research | Ultra-low-latency arbitrage | $350K-$600K start | Low (5-20/year) |
| **Investment Banks** | Goldman Sachs, JPMorgan, Morgan Stanley | Derivatives pricing, risk | $150K-$250K start | Very high (500+/year) |
| **Asset Managers** | BlackRock, Bridgewater, AQR | Portfolio management, risk | $120K-$200K start | Medium (50-100/year) |

**Culture differences**:
- **Prop trading firms**: Fast-paced, competitive, meritocratic, high autonomy
- **Hedge funds**: Research-driven, collaborative, long-term focused
- **HFT firms**: Engineering-first, extreme performance demands
- **Banks**: Hierarchical, client-facing, regulatory-heavy
- **Asset managers**: Slow-paced, institutional, conservative

**Which to target?** Most aspiring quants aim for **prop trading or quant hedge funds** due to compensation, culture, and career growth.

---

## Part II: The Technical Foundation - What You Must Learn

### 2.1 Mathematics (The Core Differentiator)

Quant interviews are **brutally mathematical**. Here's what you need:

#### **A. Probability & Statistics (40% of interview questions)**

**Undergraduate probability** (minimum):
```
- Sample spaces, events, probability axioms
- Conditional probability: P(A|B) = P(A ∩ B) / P(B)
- Independence: P(A ∩ B) = P(A) × P(B)
- Bayes' theorem: P(A|B) = P(B|A) × P(A) / P(B)
- Random variables, expectation, variance
- Common distributions: uniform, normal, exponential, binomial, Poisson
- Law of large numbers, central limit theorem
```

**Graduate probability** (competitive advantage):
```
- Measure-theoretic probability
- Conditional expectation: E[X | σ-algebra]
- Martingales, stopping times, optional stopping theorem
- Markov chains, stationary distributions
- Continuous-time processes (Poisson, Brownian motion)
```

**Example interview question** (Jane Street):

```
Q: You and I play a game. We flip a fair coin repeatedly. You win if we see HH before HT. 
   I win if we see HT before HH. What is the probability you win?

Solution approach:
Let P(HH wins) = probability we see HH before HT, starting fresh.

States:
- Start: no flips yet
- H: just flipped heads
- HH: you win
- HT: I win

Transition probabilities:
From Start:
- Flip H with prob 0.5 → go to state H
- Flip T with prob 0.5 → stay in Start

From H:
- Flip H with prob 0.5 → HH (you win)
- Flip T with prob 0.5 → HT (I win)

Let p = P(HH wins).

p = P(reach H from Start) × P(HH before HT from H)
  = 0.5 × P(HH from H) + 0.5 × p  (if T, reset to Start)

From H:
P(HH from H) = 0.5 × 1 + 0.5 × 0 = 0.5

Therefore:
p = 0.5 × 0.5 + 0.5 × p
p = 0.25 + 0.5p
0.5p = 0.25
p = 0.5

But wait, this assumes symmetry. Let's think more carefully...

Actually, from state H:
- If next flip is H → HH (you win) with prob 0.5
- If next flip is T → HT (I win) with prob 0.5

So P(HH wins from state H) = 0.5

And from Start:
- Must first reach H (prob 0.5, geometrically distributed waiting time)
- Then from H, you win with prob 0.5

By careful analysis: P(you win) = 1/3, P(I win) = 2/3

The game is NOT fair. I have 2:1 advantage because HT is "easier" to hit than HH.
```

**Why this matters**: Quant interviews are ~70% probability puzzles. Master this or you won't pass.

#### **B. Linear Algebra & Numerical Methods**

**Essential topics**:
- Matrix operations, determinants, eigenvalues
- Least squares regression (closed-form solution)
- Principal Component Analysis (PCA) for dimensionality reduction
- Numerical optimization (gradient descent, Newton's method)
- Matrix decompositions (SVD, Cholesky, QR)

**Application in quant finance**:
```python
# Portfolio optimization using linear algebra

import numpy as np

class PortfolioOptimizer:
    """
    Mean-variance optimization (Markowitz framework).
    
    Given:
    - Expected returns: μ (N-dimensional vector)
    - Covariance matrix: Σ (N×N matrix)
    
    Find: optimal portfolio weights w that maximize Sharpe ratio
    """
    
    def __init__(self, expected_returns, covariance_matrix):
        self.mu = np.array(expected_returns)  # Shape: [N]
        self.Sigma = np.array(covariance_matrix)  # Shape: [N, N]
        
    def optimize(self, target_return=None):
        """
        Solve for optimal weights.
        
        If target_return is None: maximize Sharpe ratio
        If target_return is specified: minimize variance subject to E[R] = target
        """
        N = len(self.mu)
        
        # Method 1: Maximum Sharpe ratio (unconstrained)
        # w* = Σ^(-1) μ / (1^T Σ^(-1) μ)
        Sigma_inv = np.linalg.inv(self.Sigma)
        ones = np.ones(N)
        
        numerator = Sigma_inv @ self.mu
        denominator = ones @ Sigma_inv @ self.mu
        
        w_optimal = numerator / denominator
        
        # Method 2: Target return with minimum variance (quadratic programming)
        if target_return is not None:
            # Lagrangian: L = w^T Σ w + λ(μ^T w - r_target) + γ(1^T w - 1)
            # Solve using KKT conditions (closed-form for this problem)
            
            A = np.vstack([self.mu, ones])  # Constraints matrix [2, N]
            b = np.array([target_return, 1])  # [2]
            
            # Solve: w* = Σ^(-1) A^T (A Σ^(-1) A^T)^(-1) b
            Sigma_inv_AT = Sigma_inv @ A.T
            middle = np.linalg.inv(A @ Sigma_inv_AT)
            w_optimal = Sigma_inv_AT @ middle @ b
        
        return {
            'weights': w_optimal,
            'expected_return': w_optimal @ self.mu,
            'volatility': np.sqrt(w_optimal @ self.Sigma @ w_optimal),
            'sharpe_ratio': (w_optimal @ self.mu) / np.sqrt(w_optimal @ self.Sigma @ w_optimal)
        }

# Example usage
returns = np.array([0.10, 0.12, 0.08, 0.15])  # Annual expected returns for 4 assets
cov = np.array([
    [0.04, 0.01, 0.02, 0.01],
    [0.01, 0.09, 0.01, 0.03],
    [0.02, 0.01, 0.02, 0.01],
    [0.01, 0.03, 0.01, 0.16]
])

optimizer = PortfolioOptimizer(returns, cov)
result = optimizer.optimize()

print(f"Optimal weights: {result['weights']}")
print(f"Expected return: {result['expected_return']:.2%}")
print(f"Volatility: {result['volatility']:.2%}")
print(f"Sharpe ratio: {result['sharpe_ratio']:.2f}")
```

**Interview example** (Two Sigma):
```
Q: Given a 3×3 covariance matrix Σ and expected return vector μ, 
   derive the formula for the minimum-variance portfolio.

A: The minimum-variance portfolio minimizes w^T Σ w subject to w^T 1 = 1.

Using Lagrangian:
L = w^T Σ w + λ(w^T 1 - 1)

Take derivative with respect to w:
∂L/∂w = 2Σw + λ1 = 0
→ w = -(λ/2) Σ^(-1) 1

Apply constraint w^T 1 = 1:
-(λ/2) 1^T Σ^(-1) 1 = 1
λ = -2 / (1^T Σ^(-1) 1)

Substitute:
w* = Σ^(-1) 1 / (1^T Σ^(-1) 1)

This is the closed-form solution for minimum-variance weights.
```

#### **C. Stochastic Calculus (For Derivatives Pricing)**

**Required for**: Options trading desks, volatility arbitrage, exotic derivatives

**Core concepts**:
```
1. Brownian motion: W_t ~ N(0, t), continuous but nowhere differentiable
2. Stochastic differential equations: dS_t = μ S_t dt + σ S_t dW_t
3. Ito's lemma: Tool for computing derivatives of stochastic processes
4. Risk-neutral pricing: Derivatives priced under Q-measure, not P-measure
5. Black-Scholes PDE: ∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0
```

**Black-Scholes derivation** (expected knowledge for options roles):

```
Assume stock price follows geometric Brownian motion:
dS_t = μ S_t dt + σ S_t dW_t

Let V(S, t) = value of European call option.

By Ito's lemma:
dV = (∂V/∂t + μS ∂V/∂S + ½σ²S² ∂²V/∂S²) dt + σS ∂V/∂S dW

Construct risk-free portfolio:
Π = V - Δ S  (long option, short Δ shares of stock)

Where Δ = ∂V/∂S (delta hedge)

Change in portfolio value:
dΠ = dV - Δ dS
   = (∂V/∂t + ½σ²S² ∂²V/∂S²) dt  (dW terms cancel!)

Since Π is risk-free, must earn risk-free rate:
dΠ = r Π dt = r(V - S ∂V/∂S) dt

Equating:
∂V/∂t + ½σ²S² ∂²V/∂S² = r(V - S ∂V/∂S)

Rearranging:
∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0

This is the Black-Scholes PDE. Solving with boundary conditions gives:

C(S, t) = S Φ(d₁) - K e^(-r(T-t)) Φ(d₂)

Where:
d₁ = (ln(S/K) + (r + ½σ²)(T-t)) / (σ√(T-t))
d₂ = d₁ - σ√(T-t)
Φ = standard normal CDF
```

**Interview example** (Citadel):
```
Q: Derive the Greeks (delta, gamma, vega) from Black-Scholes formula.

A: 
Delta = ∂C/∂S = Φ(d₁)
Gamma = ∂²C/∂S² = φ(d₁) / (S σ √(T-t))  [where φ = standard normal PDF]
Vega = ∂C/∂σ = S φ(d₁) √(T-t)
Theta = ∂C/∂t = -(S φ(d₁) σ) / (2√(T-t)) - r K e^(-r(T-t)) Φ(d₂)
Rho = ∂C/∂r = K (T-t) e^(-r(T-t)) Φ(d₂)
```

---

### 2.2 Programming (The Implementation Layer)

#### **A. Python (Research & Prototyping)**

**Expected proficiency**: Write clean, vectorized code for data analysis and backtesting.

```python
# Typical quant Python code: backtesting a mean-reversion strategy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MeanReversionStrategy:
    """
    Simple Bollinger Band mean reversion strategy.
    
    Signal:
    - BUY when price < lower_band (oversold)
    - SELL when price > upper_band (overbought)
    - EXIT when price returns to moving average
    """
    
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        
    def generate_signals(self, prices):
        """
        Generate trading signals from price series.
        
        Args:
            prices: pandas Series of asset prices
        
        Returns:
            signals: pandas Series of {-1, 0, +1} (short, neutral, long)
        """
        # Compute Bollinger Bands
        rolling_mean = prices.rolling(window=self.window).mean()
        rolling_std = prices.rolling(window=self.window).std()
        
        upper_band = rolling_mean + self.num_std * rolling_std
        lower_band = rolling_mean - self.num_std * rolling_std
        
        # Generate signals (vectorized)
        signals = pd.Series(0, index=prices.index)
        signals[prices < lower_band] = 1   # Buy signal
        signals[prices > upper_band] = -1  # Sell signal
        
        return signals
    
    def backtest(self, prices, signals, transaction_cost=0.001):
        """
        Backtest strategy and compute performance metrics.
        
        Args:
            prices: Price series
            signals: Trading signals {-1, 0, +1}
            transaction_cost: Proportional cost per trade (e.g., 0.1%)
        
        Returns:
            results: Dict of performance metrics
        """
        # Compute returns
        returns = prices.pct_change()
        
        # Strategy returns (signal from previous day × today's return)
        strategy_returns = signals.shift(1) * returns
        
        # Subtract transaction costs (when position changes)
        position_changes = signals.diff().abs()
        costs = position_changes * transaction_cost
        strategy_returns -= costs
        
        # Cumulative returns
        cumulative = (1 + strategy_returns).cumprod()
        
        # Performance metrics
        total_return = cumulative.iloc[-1] - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        return {
            'total_return': total_return,
            'annual_return': (1 + total_return) ** (252 / len(returns)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (strategy_returns > 0).mean(),
        }

# Example usage
prices = pd.Series(...)  # Load price data
strategy = MeanReversionStrategy(window=20, num_std=2)
signals = strategy.generate_signals(prices)
results = strategy.backtest(prices, signals)

print(f"Annual return: {results['annual_return']:.2%}")
print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
print(f"Max drawdown: {results['max_drawdown']:.2%}")
```

**Python libraries every quant must know**:
- **pandas**: Time series manipulation, resampling, rolling windows
- **numpy**: Vectorized operations, linear algebra
- **scipy**: Statistical tests, optimization, interpolation
- **statsmodels**: Regression, time series models (ARIMA, GARCH)
- **scikit-learn**: Machine learning (RandomForest, GradientBoosting, cross-validation)
- **matplotlib/seaborn**: Visualization
- **backtrader/zipline**: Backtesting frameworks

#### **B. C++ (Production Systems & HFT)**

**Expected proficiency**: Write low-latency, thread-safe code with minimal overhead.

**Core C++ concepts for quants**:
```cpp
// 1. Template metaprogramming (compile-time optimization)
template<typename T, size_t N>
class FixedRingBuffer {
    /**
     * Lock-free ring buffer with fixed capacity (no dynamic allocation).
     * Used for ultra-low-latency message passing between threads.
     */
    std::array<T, N> buffer_;
    std::atomic<size_t> write_idx_{0};
    std::atomic<size_t> read_idx_{0};
    
public:
    bool push(const T& item) {
        size_t current_write = write_idx_.load(std::memory_order_relaxed);
        size_t next_write = (current_write + 1) % N;
        
        // Check if full
        if (next_write == read_idx_.load(std::memory_order_acquire)) {
            return false;  // Buffer full
        }
        
        buffer_[current_write] = item;
        write_idx_.store(next_write, std::memory_order_release);
        return true;
    }
    
    bool pop(T& item) {
        size_t current_read = read_idx_.load(std::memory_order_relaxed);
        
        // Check if empty
        if (current_read == write_idx_.load(std::memory_order_acquire)) {
            return false;  // Buffer empty
        }
        
        item = buffer_[current_read];
        read_idx_.store((current_read + 1) % N, std::memory_order_release);
        return true;
    }
};

// 2. Move semantics (avoid copies)
class Order {
public:
    std::string symbol;
    double price;
    int64_t quantity;
    
    // Move constructor (O(1) instead of O(n) string copy)
    Order(Order&& other) noexcept
        : symbol(std::move(other.symbol))
        , price(other.price)
        , quantity(other.quantity)
    {}
};

// 3. constexpr for compile-time computation
constexpr double compute_black_scholes_call(double S, double K, double r, double sigma, double T) {
    // Compile-time evaluation if all inputs are constexpr
    // (Not actually possible for Black-Scholes due to erf(), but shows concept)
    // Used for lookup table generation
}

// 4. Cache-friendly data structures
struct alignas(64) MarketData {  // Align to cache line (64 bytes)
    double bid;
    double ask;
    int32_t bid_size;
    int32_t ask_size;
    uint64_t timestamp_ns;
    // Padding to 64 bytes total
};
// → Ensures each MarketData object fits in one cache line (no false sharing)
```

**Why C++ for HFT?**
```
Python execution time: 1,000,000 ns (1 millisecond)
C++ execution time:         1,000 ns (1 microsecond)

Speedup: 1000×

In HFT, microseconds = millions of dollars
```

#### **C. SQL & Data Engineering**

**Use case**: Query historical market data for research and analysis.

```sql
-- Typical quant SQL query: Find stocks with unusual volume spikes

WITH avg_volume AS (
    SELECT 
        symbol,
        AVG(volume) AS avg_20d_volume
    FROM 
        market_data
    WHERE 
        date BETWEEN '2025-01-01' AND '2025-12-31'
    GROUP BY 
        symbol
)
SELECT 
    m.date,
    m.symbol,
    m.volume,
    av.avg_20d_volume,
    m.volume / av.avg_20d_volume AS volume_ratio,
    m.close / m.open - 1 AS intraday_return
FROM 
    market_data m
JOIN 
    avg_volume av ON m.symbol = av.symbol
WHERE 
    m.date = '2026-03-01'
    AND m.volume > 3 * av.avg_20d_volume  -- 3× normal volume
ORDER BY 
    volume_ratio DESC
LIMIT 50;

-- Use case: Find potential momentum trades (high volume + positive return)
```

---

### 2.3 Finance Domain Knowledge

#### **A. Market Microstructure**

**Concepts every quant must understand**:

1. **Order book dynamics**:
```
BID (buyers)          |  ASK (sellers)
-------------------------------------------
100 shares @ $50.00   |  $50.05 @ 200 shares
500 shares @ $49.99   |  $50.06 @ 100 shares
300 shares @ $49.98   |  $50.07 @ 400 shares

Bid-ask spread = $0.05
Mid price = ($50.00 + $50.05) / 2 = $50.025

Liquidity: High (tight spread, large size at best levels)
```

2. **Order types**:
   - **Market order**: Execute immediately at best available price (pays spread)
   - **Limit order**: Execute only at specified price or better (provides liquidity)
   - **Stop-loss order**: Trigger market order when price crosses threshold
   - **Iceberg order**: Hide order size, show only small portion

3. **Market making**:
```python
class MarketMaker:
    """
    Simplified market making strategy.
    
    Goal: Earn bid-ask spread while managing inventory risk.
    """
    
    def __init__(self, max_inventory=1000, spread_width=0.05):
        self.inventory = 0  # Current position (negative = short)
        self.max_inventory = max_inventory
        self.spread_width = spread_width
        
    def quote(self, fair_value):
        """
        Generate bid/ask quotes around fair value.
        
        Inventory management: Skew quotes to reduce inventory.
        """
        # Base quotes (symmetric around fair value)
        base_bid = fair_value - self.spread_width / 2
        base_ask = fair_value + self.spread_width / 2
        
        # Inventory skew: if long, shade bid down and ask down (want to sell)
        skew = (self.inventory / self.max_inventory) * self.spread_width
        
        bid = base_bid - skew
        ask = base_ask - skew
        
        return {
            'bid': round(bid, 2),
            'ask': round(ask, 2),
            'bid_size': 100,
            'ask_size': 100,
        }
    
    def on_fill(self, side, quantity, price):
        """Update inventory after trade execution."""
        if side == 'BUY':
            self.inventory += quantity
        else:  # SELL
            self.inventory -= quantity
```

#### **B. Derivatives Pricing**

**Options basics** (must know cold for interviews):

```
European Call Option:
- Right to BUY stock at strike K on expiration date T
- Payoff at expiration: max(S_T - K, 0)
- Value today: C(S, K, T, r, σ) = Black-Scholes formula

European Put Option:
- Right to SELL stock at strike K on expiration date T
- Payoff: max(K - S_T, 0)
- Put-call parity: C - P = S - K e^(-rT)

Greeks:
- Delta (Δ): ∂V/∂S (how much option value changes per $1 move in stock)
- Gamma (Γ): ∂²V/∂S² (how much delta changes per $1 move)
- Vega (ν): ∂V/∂σ (sensitivity to volatility)
- Theta (Θ): ∂V/∂t (time decay)
- Rho (ρ): ∂V/∂r (sensitivity to interest rates)
```

**Interview example** (Jane Street):
```
Q: An at-the-money call option has delta = 0.5. If the stock moves up $1, 
   what happens to the option's delta?

A: Delta will INCREASE (become more positive, closer to 1.0).

Reason: Gamma (∂²V/∂S²) is positive for long options. As the stock moves up,
the option goes deeper in-the-money, and delta approaches 1.0 (behaves more like stock).

Specifically: Δ_new ≈ Δ_old + Γ × ΔS = 0.5 + Γ × 1

For ATM option: Γ ≈ 0.01 (rule of thumb for 30-day expiration)
So: Δ_new ≈ 0.5 + 0.01 = 0.51

For deep ITM option: Δ → 1.0 (acts like 100 shares of stock)
For deep OTM option: Δ → 0.0 (expires worthless)
```

---

## Part III: The Education Pathway

### 3.1 Undergraduate Strategy

**Ideal majors** (ranked by quant hiring frequency):
1. **Mathematics** (pure or applied)
2. **Computer Science** (algorithms, systems)
3. **Physics** (especially theoretical physics)
4. **Statistics** / **Data Science**
5. **Engineering** (electrical, mechanical)
6. **Economics** (only if combined with strong math minor)

**Course selection strategy**:

```
Year 1-2 (Foundation):
├─ Calculus I, II, III (Multivariable)
├─ Linear Algebra
├─ Discrete Math
├─ Intro CS (Python or Java)
├─ Data Structures & Algorithms
└─ Probability & Statistics I

Year 2-3 (Core):
├─ Real Analysis
├─ Stochastic Processes
├─ Numerical Methods
├─ Machine Learning
├─ Systems Programming (C++)
├─ Databases & SQL
└─ Algorithms (advanced)

Year 3-4 (Specialization):
├─ Partial Differential Equations
├─ Optimization (convex, nonlinear)
├─ Time Series Analysis
├─ Financial Mathematics
├─ High-Performance Computing
└─ Capstone: Quant research project
```

**Extracurriculars that matter**:
- **Math competitions**: Putnam, IMO medals are gold-tier signals
- **Coding competitions**: Codeforces (Expert+), LeetCode (top 5%), ACM ICPC
- **Quant clubs**: Build and present trading strategies
- **Personal projects**: Deployed algo trading bots, Kaggle competitions

**GPA target**: 3.7+ in technical courses (4.0 preferred for Jane Street, Citadel)

### 3.2 Graduate Education (MFE Programs)

**Should you get a Master's in Financial Engineering?**

**Pros**:
- ✅ Direct pipeline to quant roles (80% placement at top programs)
- ✅ Structured curriculum covering all necessary topics
- ✅ Recruiter access (on-campus interviews)
- ✅ Peer network (your classmates will be quant colleagues)

**Cons**:
- ❌ Expensive ($60K-$120K tuition)
- ❌ 1-2 year opportunity cost
- ❌ Not necessary if you have strong math/CS background

**Top MFE programs** (2026 rankings):

| **Rank** | **Program** | **Location** | **Class Size** | **Avg Starting Comp** | **Placement Rate** |
|----------|-------------|--------------|----------------|-----------------------|--------------------|
| 1 | **Carnegie Mellon MSCF** | Pittsburgh, NY | 100 | $180K | 100% |
| 2 | **Princeton MFin** | Princeton | 30 | $195K | 100% |
| 3 | **MIT MFin** | Cambridge | 120 | $175K | 98% |
| 4 | **Columbia MFE** | New York | 130 | $160K | 95% |
| 5 | **Berkeley MFE** | Berkeley | 70 | $170K | 97% |
| 6 | **Baruch MFE** | New York | 35 | $155K | 100% |

**Note**: These are banking/buy-side placement figures. Prop trading firms (Jane Street, SIG) pay 2-3× these amounts but hire directly from undergrad or PhD programs, often bypassing MFE programs.

**Alternative: PhD in STEM**

**Pros**:
- ✅ Deep research training (matches quant researcher work)
- ✅ Usually funded (no tuition, $30K-$40K stipend)
- ✅ Prestigious signal (especially for quant hedge funds)
- ✅ Can pivot to academia if quant doesn't work out

**Cons**:
- ❌ 5-7 years (massive time investment)
- ❌ Lower starting comp than undergrad → prop trading (PhD: $250K, undergrad: $350K at Jane Street)
- ❌ May be overqualified for quant dev roles

**Best PhD fields for quant careers**:
1. Statistics (time series, Bayesian methods)
2. Operations Research (optimization, stochastic control)
3. Physics (especially computational, statistical physics)
4. Computer Science (machine learning, algorithms)
5. Applied Mathematics (PDEs, numerical methods)

---

## Part IV: The Interview Gauntlet

### 4.1 Interview Process Overview

**Typical timeline** (Jane Street / Citadel / Two Sigma):

```
Week 0: Application submitted
Week 1-2: Resume screen (automated)
Week 3: Online assessment (90 minutes)
        ├─ 20 probability questions
        ├─ 5 coding problems (LeetCode medium/hard)
        └─ Pass rate: ~20%

Week 4-5: First phone screen (60 minutes)
          ├─ Probability puzzles (3-4 questions)
          ├─ Coding (1-2 problems)
          └─ Behavioral fit
          Pass rate: ~40%

Week 6: Second phone screen (60 minutes)
        ├─ Advanced probability
        ├─ Stochastic calculus (for options roles)
        ├─ Mental math
        └─ Pass rate: ~50%

Week 7-8: Superday (onsite, 4-8 hours)
          ├─ 4-6 interviews back-to-back
          ├─ Mix of probability, coding, brain teasers, market making games
          ├─ Behavioral/culture fit
          └─ Pass rate: ~30%

Offer rate: 0.20 × 0.40 × 0.50 × 0.30 = 1.2%

For 10,000 applicants → 120 offers
```

**Why so selective?** Quant firms optimize for **low false positive rate**. They'd rather reject 100 great candidates than hire 1 mediocre one (because one bad hire can lose millions in trading losses).

### 4.2 Probability Interview Questions (The Core Filter)

**Example questions** (actual interview questions from Glassdoor, Rooftop Slushie, etc.):

#### **Question 1** (Easy - Citadel):
```
Q: You roll two fair six-sided dice. What is the probability that the sum is 7?

A: 
Possible outcomes: 6 × 6 = 36 total

Ways to get sum = 7:
(1,6), (2,5), (3,4), (4,3), (5,2), (6,1) = 6 ways

Probability = 6/36 = 1/6 ≈ 16.67%
```

#### **Question 2** (Medium - Jane Street):
```
Q: You have a fair coin. You flip it repeatedly until you see two heads in a row (HH).
   What is the expected number of flips?

A: Use Markov chain / recursive approach.

States:
- Start: no flips yet
- H: just flipped one head
- HH: done (absorbing state)

Let E_start = expected flips from Start, E_H = expected flips from H.

From Start:
- Flip H with prob 0.5 → go to state H (cost: 1 flip)
- Flip T with prob 0.5 → stay in Start (cost: 1 flip)

E_start = 1 + 0.5 × E_H + 0.5 × E_start

From H:
- Flip H with prob 0.5 → done (cost: 1 flip)
- Flip T with prob 0.5 → reset to Start (cost: 1 flip)

E_H = 1 + 0.5 × 0 + 0.5 × E_start

Solve system of equations:
E_H = 1 + 0.5 E_start
E_start = 1 + 0.5 E_H + 0.5 E_start
→ 0.5 E_start = 1 + 0.5 E_H
→ E_start = 2 + E_H = 2 + 1 + 0.5 E_start
→ 0.5 E_start = 3
→ E_start = 6

Answer: Expected flips = 6
```

#### **Question 3** (Hard - HRT):
```
Q: 100 prisoners are randomly assigned numbers 1-100. There's a room with 100 boxes,
   each containing a random number 1-100 (all distinct). Each prisoner enters the room
   individually, opens 50 boxes, and must find their own number. Prisoners cannot
   communicate after entering. If ALL prisoners find their number, they go free.
   If even one fails, all die. What strategy maximizes survival probability?

A: "Follow the cycle" strategy.

Naive strategy: Each prisoner opens 50 random boxes.
P(one prisoner succeeds) = 50/100 = 0.5
P(all 100 succeed) = 0.5^100 ≈ 7.9 × 10^(-31)  → essentially zero

Optimal strategy: Each prisoner follows the cycle starting from their number.

Setup: Boxes are numbered 1-100. Box i contains number c[i].

Strategy:
- Prisoner k opens box k
- If it contains number m, open box m next
- If that contains number n, open box n next
- Continue until you find your number (or 50 boxes exhausted)

Why this works:
The box-content mapping defines a permutation, which decomposes into cycles.
Example cycle: 1 → 42 → 7 → 89 → 1

Prisoner 1 opens boxes: 1 → 42 → 7 → 89 → 1 (finds their number in 4 steps)
Prisoner 42 opens boxes: 42 → 7 → 89 → 1 → 42 (finds their number in 4 steps)

Key insight: All prisoners in a cycle succeed if cycle length ≤ 50.

They fail if ANY cycle has length > 50.

Probability of failure:
P(exists cycle of length > 50) = ∑(k=51 to 100) 1/k ≈ 0.688

P(success) = 1 - 0.688 ≈ 31.2%

→ This is vastly better than 10^(-31) from random strategy!
```

This problem tests:
- ✅ Combinatorics (permutations, cycles)
- ✅ Probability (conditional events)
- ✅ Creative problem-solving (non-obvious strategy)
- ✅ Mental stamina (takes 15-30 minutes to solve)

### 4.3 Coding Interview Questions

**Typical difficulty**: LeetCode Medium to Hard

**Example 1** (Jane Street - Medium):
```python
"""
Q: Implement a class that maintains a running median of a stream of numbers.

Methods:
- add_number(num): Add a number to the stream (O(log n))
- get_median(): Return current median (O(1))
"""

import heapq

class RunningMedian:
    """
    Two-heap solution for running median.
    
    Strategy:
    - max_heap: stores smaller half of numbers (max at top)
    - min_heap: stores larger half of numbers (min at top)
    - Invariant: len(max_heap) == len(min_heap) or len(max_heap) == len(min_heap) + 1
    
    Median:
    - If total count is odd: max_heap.top()
    - If even: (max_heap.top() + min_heap.top()) / 2
    """
    
    def __init__(self):
        self.max_heap = []  # Smaller half (negated for max-heap behavior)
        self.min_heap = []  # Larger half
        
    def add_number(self, num: float):
        """Add number to stream in O(log n) time."""
        # Add to max_heap first
        heapq.heappush(self.max_heap, -num)
        
        # Balance: ensure max_heap.top() <= min_heap.top()
        if self.min_heap and (-self.max_heap[0]) > self.min_heap[0]:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        
        # Balance sizes: max_heap can have at most 1 more element than min_heap
        if len(self.max_heap) > len(self.min_heap) + 1:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        elif len(self.min_heap) > len(self.max_heap):
            val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -val)
    
    def get_median(self) -> float:
        """Return current median in O(1) time."""
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return -self.max_heap[0]

# Test
rm = RunningMedian()
for num in [5, 15, 1, 3]:
    rm.add_number(num)
    print(f"Added {num}, median = {rm.get_median()}")

# Output:
# Added 5, median = 5.0
# Added 15, median = 10.0  (average of 5 and 15)
# Added 1, median = 5.0    (middle of [1, 5, 15])
# Added 3, median = 4.0    (average of 3 and 5)
```

**Example 2** (Citadel - Hard):
```python
"""
Q: Given a list of stock prices, find the maximum profit from at most k transactions.
   (A transaction = buy then sell. Cannot hold multiple positions simultaneously.)
"""

def max_profit_k_transactions(prices, k):
    """
    Dynamic programming solution.
    
    State: dp[i][j] = max profit after j transactions on day i
    
    Recurrence:
    dp[i][j] = max(
        dp[i-1][j],  # Don't trade today
        max(prices[i] - prices[m] + dp[m][j-1] for m in range(i))  # Sell today
    )
    
    Time: O(n² k), Space: O(n k)
    """
    n = len(prices)
    if n <= 1 or k == 0:
        return 0
    
    # Optimization: if k >= n/2, it's equivalent to unlimited transactions
    if k >= n // 2:
        return max_profit_unlimited(prices)
    
    # DP table
    dp = [[0] * (k + 1) for _ in range(n)]
    
    for j in range(1, k + 1):  # j transactions
        max_diff = -prices[0]  # max(dp[m][j-1] - prices[m])
        
        for i in range(1, n):  # day i
            # Option 1: don't sell on day i
            dp[i][j] = dp[i-1][j]
            
            # Option 2: sell on day i (bought on some earlier day m)
            dp[i][j] = max(dp[i][j], prices[i] + max_diff)
            
            # Update max_diff for next iteration
            max_diff = max(max_diff, dp[i][j-1] - prices[i])
    
    return dp[n-1][k]

def max_profit_unlimited(prices):
    """When k >= n/2, greedily buy before every price increase."""
    return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))

# Test
prices = [3, 2, 6, 5, 0, 3]
k = 2
print(max_profit_k_transactions(prices, k))  # Output: 7 (buy at 2, sell at 6; buy at 0, sell at 3)
```

### 4.4 Market Making Simulation (Interactive Round)

**Setup**: You're a market maker in a simplified game. Interviewer is a "customer" who can buy or sell from you.

**Game rules**:
```
- You quote bid and ask prices for a fictional asset
- Customer can hit your bid (sell to you) or lift your ask (buy from you)
- Asset has true value V (unknown to you), drawn from N(100, 10²)
- After each trade, you observe V with noise: V_observed = V + ε, ε ~ N(0, 1²)
- Game lasts 10 rounds
- Your PnL = sum of (V - price paid) for your purchases, (price received - V) for your sales
```

**Example transcript**:

```
Round 1:
You: "I'll make a market: 95 bid, 105 ask"
Interviewer: "I'll sell you 100 shares at 95"
You: [Inventory: +100 shares, Cost: $95/share]

Round 2:
You observe: V_observed = 97 (with noise)
Bayesian update: E[V | observed 97] ≈ 97 (simplified)
You: "I'll make a market: 92 bid, 102 ask" (lowered due to negative signal + inventory risk)
Interviewer: "I'll buy 100 shares at 102"
You: [Inventory: 0 shares, PnL: (102 - 95) × 100 = $700]

Round 3:
You observe: V_observed = 101
You: "96 bid, 106 ask"
Interviewer: "Pass"
...

Final PnL: $1,200 (successful market making)
```

**What the interviewer is testing**:
- ✅ Bayesian updating (do you adjust beliefs based on signals?)
- ✅ Inventory risk management (do you skew quotes when you have position?)
- ✅ Spread sizing (do you widen spread when uncertain?)
- ✅ Mental math (can you compute PnL in real-time?)
- ✅ Composure under pressure

---

## Part V: The Self-Study Roadmap (If Starting Tomorrow)

### 5.1 Month 1-3: Mathematical Foundations

**Goal**: Build rock-solid probability, statistics, and linear algebra skills.

**Daily schedule** (20 hours/week):
```
Week 1-4: Probability
├─ Textbook: "Introduction to Probability" (Blitzstein & Hwang)
├─ Problems: 50 problems from textbook
├─ Online: Brilliant.org probability course
└─ Supplement: 3Blue1Brown YouTube (visual intuition)

Week 5-8: Statistics & Time Series
├─ Textbook: "All of Statistics" (Wasserman)
├─ Applied: Analyze S&P 500 returns (correlation, regression, ARMA models)
├─ Python: pandas, scipy, statsmodels
└─ Project: Replicate academic paper (e.g., "Momentum" by Jegadeesh & Titman)

Week 9-12: Linear Algebra & Optimization
├─ Textbook: "Introduction to Linear Algebra" (Strang)
├─ Applied: Portfolio optimization (mean-variance framework)
├─ Python: numpy linear algebra, cvxpy for optimization
└─ Project: Build efficient frontier calculator
```

**Checkpoint**: By end of Month 3, you should be able to:
- ✅ Solve 50%+ of Easy probability questions instantly
- ✅ Solve 30%+ of Medium probability questions in 5-10 minutes
- ✅ Implement statistical tests from scratch (t-test, chi-square, regression)
- ✅ Explain covariance, correlation, independence without hesitation

### 5.2 Month 4-6: Programming Fluency

**Goal**: Achieve LeetCode proficiency + Python for quant research.

**Daily schedule** (25 hours/week):
```
Week 1-6: Python for Quant Research
├─ Pandas bootcamp: 100 time series manipulation exercises
├─ Numpy: Vectorization, broadcasting, memory layout
├─ Build 3 trading strategies from scratch:
│   1. Moving average crossover
│   2. Mean reversion (Bollinger Bands)
│   3. Momentum (relative strength)
├─ Backtest on 10 years of data (Yahoo Finance API)
└─ Implement portfolio optimization, risk metrics (VaR, CVaR)

Week 7-12: C++ Fundamentals (if targeting HFT roles)
├─ Textbook: "C++ Primer" (Lippman)
├─ Practice: Implement data structures (vector, hashmap, priority queue) from scratch
├─ Learn: Smart pointers, move semantics, templates, STL
├─ Build: Order matching engine simulation (practice project)
└─ Optimize: Profile with perf, reduce latency

Week 7-12 (alternative): LeetCode Grinding (if targeting quant research)
├─ Solve 150 problems (50 Easy, 75 Medium, 25 Hard)
├─ Focus: Arrays, hashmaps, dynamic programming, trees, graphs
├─ Target: Medium problems in <20 minutes (interview speed)
└─ Mock interviews: Pramp, Interviewing.io
```

**Checkpoint**: By end of Month 6, you should be able to:
- ✅ Write clean, vectorized Python code for data analysis
- ✅ Solve LeetCode Medium in 20 minutes (75% success rate)
- ✅ Implement binary search, DFS/BFS, DP on whiteboard
- ✅ Explain time/space complexity for any algorithm

### 5.3 Month 7-9: Finance Deep Dive

**Goal**: Master options pricing, risk management, and market microstructure.

**Daily schedule** (20 hours/week):
```
Week 1-4: Derivatives Pricing
├─ Textbook: "Options, Futures, and Other Derivatives" (Hull)
├─ Study: Black-Scholes derivation (memorize formula, derive Greeks)
├─ Python: Implement Black-Scholes pricer, Monte Carlo simulation
├─ Practice: 50 derivatives interview questions (from Glassdoor)
└─ Advanced: Implied volatility, volatility smile, term structure

Week 5-8: Market Microstructure
├─ Study: Order book dynamics, market making, adverse selection
├─ Paper: "Market Microstructure Theory" (O'Hara)
├─ Build: Order book simulator (L2 data processing)
└─ Analyze: Real order book data from Polygon.io or Alpaca

Week 9-12: Risk Management
├─ Metrics: Value at Risk (VaR), Expected Shortfall (CVaR), Greeks
├─ Portfolio: Diversification, correlation breakdown, factor models
├─ Regulations: Understand SEC, FINRA rules (basic familiarity)
└─ Project: Build risk dashboard for multi-asset portfolio
```

**Checkpoint**: By end of Month 9, you should be able to:
- ✅ Derive Black-Scholes on whiteboard in 10 minutes
- ✅ Compute Greeks mentally for simple cases
- ✅ Explain market making PnL sources (spread capture, inventory risk)
- ✅ Calculate VaR for a portfolio from covariance matrix

### 5.4 Month 10-12: Interview Preparation

**Goal**: Pass the interview gauntlet at top firms.

**Daily schedule** (30 hours/week):
```
Week 1-4: Probability Drills
├─ Solve 200+ probability problems:
│   - "A Practical Guide to Quantitative Finance Interviews" (Joshi)
│   - "Heard on the Street" (Crack)
│   - "Quant Job Interview Questions and Answers" (Joshi)
├─ Focus: Expected value, conditional probability, brain teasers
├─ Speed: Aim for <5 minutes per problem (interview pace)
└─ Mock interviews: Practice explaining solutions out loud

Week 5-8: Mental Math & Brainteasers
├─ Drill: 30 minutes daily of mental arithmetic
│   - Multiply 2-digit numbers: 47 × 63 = ?
│   - Estimate: sqrt(247), ln(18), e^3.2
│   - Fractions: 17/23 + 11/19 = ?
├─ Brain teasers: 50 Fermi estimation problems (e.g., "How many gas stations in NYC?")
└─ Game theory: 30 strategic games (e.g., "Guess 2/3 of average")

Week 9-12: Mock Interviews + Applications
├─ Mock interviews: 20+ full mocks (Pramp, peers, career coaches)
├─ Apply: Submit to 30+ firms (cast wide net)
├─ Networking: Reach out to quants on LinkedIn, attend quant meetups
├─ Prepare: Firm-specific research (read trading strategies, company culture)
└─ Polish: Resume, cover letter, GitHub portfolio
```

**Checkpoint**: By end of Month 12, you should be able to:
- ✅ Pass online assessments (top 20% consistently)
- ✅ Solve phone screen probability questions (80% success rate)
- ✅ Implement coding problems under time pressure (no bugs)
- ✅ Explain your thought process clearly (interviewers grade communication)

---

## Part VI: Day-to-Day Reality of a Quant

### 6.1 Typical Workday (Quant Researcher at Two Sigma)

**7:00 AM**: Arrive at office, review overnight PnL and trading logs

**7:30 AM - 9:00 AM**: Morning meeting
- Research team discusses active signals, model performance, market regime changes
- Portfolio managers review risk exposure, capital allocation

**9:00 AM - 12:00 PM**: Deep research work
```python
# Today's task: Improve alpha signal for tech sector momentum

# Load data
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', ...]
data = load_market_data(tech_stocks, start='2020-01-01')

# Hypothesis: Tech stocks with rising analyst estimates + positive earnings surprises
# have momentum that persists for 2-4 weeks

# Feature engineering
data['analyst_revision'] = data.groupby('symbol')['target_price'].pct_change(20)
data['earnings_surprise'] = (data['actual_eps'] - data['consensus_eps']) / data['price']
data['momentum_signal'] = 0.6 * data['analyst_revision'] + 0.4 * data['earnings_surprise']

# Backtest
strategy_returns = backtest_long_short(
    data, 
    signal='momentum_signal',
    long_percentile=90,  # Long top 10%
    short_percentile=10,  # Short bottom 10%
    holding_period=20,  # 20 trading days (1 month)
)

# Risk analysis
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
print(f"Sharpe ratio: {sharpe:.2f}")

# If promising (Sharpe > 2), run through production pipeline:
# 1. Transaction cost analysis
# 2. Capacity analysis (can we trade this at scale?)
# 3. Correlation with existing strategies (diversification benefit?)
# 4. Risk checks (max drawdown, tail risk, sector concentration)
```

**12:00 PM - 1:00 PM**: Lunch (often with team, discussing markets)

**1:00 PM - 4:00 PM**: Collaboration & iteration
- Pair programming with quant dev to optimize signal computation
- Present findings to senior researchers for feedback
- Debug why a strategy stopped working (regime change? data issue?)

**4:00 PM - 6:00 PM**: Admin, learning, code review
- Review colleagues' pull requests (strategies, infrastructure changes)
- Read academic papers (stay current on latest ML techniques)
- Attend internal seminars (senior researchers present new ideas)

**6:00 PM - 7:00 PM**: Gym, dinner, decompress

**Optional evening**: Some quants continue research at home (autonomous culture)

**Weekend**: Minimal work (unless there's a critical production issue)

### 6.2 Work-Life Balance & Culture

**Hours**:
- **Prop trading / HFT**: 50-60 hours/week (manageable)
- **Hedge funds**: 55-70 hours/week (more intense)
- **Investment banks**: 60-80 hours/week (brutal for junior years)

**Culture**:
- **Collaborative**: Contrary to stereotypes, most quant firms emphasize teamwork
- **Intellectually stimulating**: Daily discussions about probability, markets, systems
- **Meritocratic**: Performance is measured objectively (PnL, Sharpe, alpha contribution)
- **Autonomous**: Freedom to pursue research directions (if you justify them with data)

**Stress level**:
- **Medium** for quant researchers (focus on long-term alpha, not daily PnL swings)
- **High** for quant traders (responsible for live capital, real-time decisions)
- **Very high** for HFT engineers (system failures = millions in losses)

---

## Part VII: Compensation & Career Progression

### 7.1 Compensation Breakdown (2026 Market Rates)

**Entry-level (0-2 years)**:

| **Role** | **Firm Type** | **Base Salary** | **Bonus** | **Total Comp** |
|----------|---------------|-----------------|-----------|----------------|
| Quant Researcher | Jane Street, SIG | $200K-$250K | $100K-$250K | **$300K-$500K** |
| Quant Developer | HRT, Jump Trading | $175K-$225K | $75K-$175K | **$250K-$400K** |
| Quant Trader | Optiver, IMC | $150K-$200K | $150K-$300K | **$300K-$500K** |
| Quant Analyst | Citadel, Two Sigma | $150K-$200K | $50K-$150K | **$200K-$350K** |
| Strats (Bank) | Goldman, JPM | $100K-$150K | $50K-$100K | **$150K-$250K** |

**Mid-level (3-5 years)**:

| **Role** | **Firm Type** | **Base Salary** | **Bonus** | **Total Comp** |
|----------|---------------|-----------------|-----------|----------------|
| Senior QR | Top hedge funds | $250K-$400K | $300K-$800K | **$550K-$1.2M** |
| Lead QD | HFT firms | $250K-$350K | $200K-$500K | **$450K-$850K** |
| PM (Portfolio Mgr) | Multi-manager funds | $300K-$500K | $500K-$2M | **$800K-$2.5M** |

**Senior (10+ years)**:

| **Role** | **Firm Type** | **Total Comp** |
|----------|---------------|----------------|
| Managing Director (MD) | Hedge funds | **$2M-$10M** |
| Partner | Prop trading firms | **$5M-$20M+** (profit share) |
| Chief Strategist | Investment banks | **$3M-$8M** |

**Wealth accumulation trajectory** (median successful quant):

```
Age 22-24: Graduate, enter as quant researcher @ $350K
Age 25-27: Promote to senior QR @ $700K
Age 28-30: Become portfolio manager @ $1.5M
Age 31-35: Hit consistent performance, earn $3M-$5M annually
Age 36-40: Accumulated wealth: $15M-$30M (after tax, lifestyle spending)
Age 40+: Options:
         - Continue at firm (partner track, $10M+ annually)
         - Start own hedge fund (raise $100M-$500M AUM)
         - Retire early (financial independence achieved)
```

**Note**: This is the **optimistic path**. Many quants plateau at $500K-$1M or exit the industry.

### 7.2 Career Progression & Exit Opportunities

**Typical progression** (quant researcher path):

```
Year 0-2: Junior Quant Researcher
         ├─ Work on existing strategies (bugs, optimizations)
         ├─ Learn infrastructure, data pipelines, company culture
         └─ Build 1-2 small alpha signals that pass testing

Year 3-5: Quant Researcher
         ├─ Own 1-2 production strategies independently
         ├─ Mentor junior researchers
         └─ Contribute meaningfully to team PnL

Year 6-10: Senior Quant Researcher
          ├─ Develop novel strategy themes (not just incremental signals)
          ├─ Manage small team (2-4 researchers)
          └─ Influence firm-wide research direction

Year 10+: Principal Researcher / Partner
         ├─ Strategic leadership (set research priorities)
         ├─ Recruit and build teams
         └─ Significant equity or profit-share (aligned with firm)
```

**Exit opportunities**:
1. **Start your own hedge fund** (requires $10M-$50M personal capital + investor network)
2. **Join a startup** as Head of Quant or Chief Data Scientist (equity upside)
3. **Big tech** (Meta, Google) → research scientist ($500K-$1M comp)
4. **Academia** (PhD → professor, publish research, advise PhD students)
5. **Retire early** (common at age 35-45 with $10M-$30M saved)

---

## Part VIII: Real-World Examples & Case Studies

### 8.1 Case Study 1: Jane Street Capital

**Profile**: Largest proprietary trading firm globally, $15B+ annual revenue (estimated).

**What they do**: Market making in equities, ETFs, bonds, options (provide liquidity to markets).

**Technology stack**:
- **OCaml**: Primary language for trading systems (type safety, functional programming)
- **Python**: Research, data analysis
- **Custom infrastructure**: Proprietary exchange connectivity, risk systems

**Interview process** (notoriously difficult):
```
Round 1: Phone screen
- 3-4 probability questions (e.g., expected value, conditional probability)
- 1 coding problem (usually array/hashmap manipulation)
- Pass rate: ~30%

Round 2: Virtual onsite (3 hours)
- 45 min: Probability deep dive (Markov chains, continuous distributions)
- 45 min: Coding (LeetCode medium/hard)
- 45 min: Market making game (live simulation)
- 45 min: Behavioral + culture fit
- Pass rate: ~20%

Superday: In-person final round (New York or Hong Kong)
- 6 interviews × 45 minutes each
- Mix of probability, game theory, mental math, systems design
- Pass rate: ~40%

Overall offer rate: 0.30 × 0.20 × 0.40 = 2.4%
```

**Why Jane Street is different**:
- Culture: Extremely collaborative (no "lone wolf" quants)
- Hiring: Values **teaching ability** (can you explain complex concepts clearly?)
- Comp: Highest in industry ($350K-$500K starting for undergrads)
- Retention: ~80% of new hires stay 3+ years (unusually high)

**Sample Jane Street interview question**:
```
Q: You have a standard deck of 52 cards. You draw cards one by one without replacement.
   You can stop at any time. You win if the last card you drew is the highest card
   you've seen so far. What strategy maximizes your win probability?

A: Optimal strategy is a "cutoff" rule.

Skip the first k cards, then stop at the first card that's higher than all k.

For k = n-1 (skip 51 cards), P(win) = 1/52
For k = 0 (stop at first card), P(win) = 1/52

Optimal k ≈ n/e where e ≈ 2.718

For n = 52: k* ≈ 19

Strategy: Skip first 19 cards, then stop at first card that's higher than all 19.

P(win) ≈ 1/e ≈ 36.8%

This is the "secretary problem" or "optimal stopping problem" from probability theory.
```

### 8.2 Case Study 2: Hudson River Trading (HFT)

**Profile**: High-frequency trading firm, focuses on ultra-low-latency arbitrage.

**What they do**: Exploit pricing inefficiencies that last milliseconds (e.g., cross-exchange arbitrage).

**Technology stack**:
- **C++**: 95% of codebase (latency-critical)
- **FPGAs**: Hardware acceleration for order routing (<1 microsecond)
- **Custom OS kernels**: Bypass Linux scheduler for deterministic latency

**Example strategy** (simplified):
```
Arbitrage opportunity:
- Apple stock trading at $150.00 on NYSE
- Apple stock trading at $150.05 on NASDAQ

Strategy:
1. Buy 1,000 shares on NYSE @ $150.00
2. Sell 1,000 shares on NASDAQ @ $150.05
3. Profit: ($150.05 - $150.00) × 1,000 = $50

Challenge: This opportunity lasts ~500 microseconds before other HFTs arbitrage it away.

Requirements:
- Detect opportunity: <100 μs
- Risk check: <50 μs
- Route orders to both exchanges: <200 μs
- Total latency budget: <350 μs (leaves 150 μs buffer)

If your system is too slow (>500 μs), you'll:
- Buy on NYSE successfully
- Fail to sell on NASDAQ (price converged)
- Hold unwanted inventory (risk)
```

**Hiring profile**:
- Strong preference for **CS + Math double majors**
- Value: Systems programming experience, competitive programming medals
- Interview: Heavier on coding than probability (inverse of hedge funds)

**Compensation**: $300K-$600K starting (highest among HFT firms)

### 8.3 Case Study 3: Bridgewater Associates (Macro Quant)

**Profile**: World's largest hedge fund ($150B AUM), focuses on global macro strategies.

**What they do**: Trade currencies, bonds, commodities based on macroeconomic models.

**Strategy example**:
```
Hypothesis: "When US inflation > 4% and Fed is hiking rates, emerging market 
            currencies depreciate against USD over next 6 months"

Data required:
- 30 years of inflation data (US CPI)
- Fed funds rate history
- Currency exchange rates (20+ EM currencies)
- Control variables (GDP growth, trade balance, political stability)

Model: Panel regression with fixed effects

Result: β_inflation = -0.23 (p < 0.01)
Interpretation: 1% increase in US inflation → 0.23% depreciation of EM currencies

Trading strategy:
- When inflation > 4% and Fed hiking → short EM currency basket
- Position size: proportional to (inflation - 4%) × confidence
- Risk management: Stop loss at -2% drawdown, diversify across 10+ currencies
```

**Hiring profile**:
- PhD preferred (economics, statistics, operations research)
- Value: Macro understanding, causal inference, research rigor
- Culture: Extremely data-driven, Ray Dalio's "Principles" culture

**Compensation**: $200K-$400K starting (lower than prop trading, but lifestyle balance better)

---

## Part IX: Common Mistakes & How to Avoid Them

### 9.1 Mistake #1: Weak Probability Foundations

**Problem**: Many candidates study "quant interview question banks" without mastering **fundamental probability theory**.

**Consequence**: You can solve memorized problems but fail on novel variations.

**Example**:
```
Memorized question:
Q: Flip a coin until you see two heads in a row. Expected flips?
A: 6 (memorized)

Interview variation:
Q: Flip a coin until you see HHT. Expected flips?
A: ??? (panic, can't adapt memorized approach)

Correct approach: Build Markov chain from first principles (not memorization).
```

**Fix**: Study **theory first**, then practice problems. Understand *why* methods work, not just *how* to apply them.

### 9.2 Mistake #2: Neglecting Communication Skills

**Problem**: Many technically strong candidates fail because they can't **explain their thought process clearly**.

**Interviewer perspective**:
```
Bad candidate:
Interviewer: "What's the probability of rolling a sum of 7 with two dice?"
Candidate: [10 seconds of silence] "One-sixth."
Interviewer: "How did you get that?"
Candidate: "Uh, I just counted."

→ No insight into thought process, hard to calibrate skill level

Good candidate:
Interviewer: "What's the probability of rolling a sum of 7 with two dice?"
Candidate: "Let me think through this systematically. With two dice, there are 
           6 × 6 = 36 equally likely outcomes. For a sum of 7, I need to count
           pairs that add up to 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1).
           That's 6 outcomes. So the probability is 6/36 = 1/6."

→ Clear, structured, verifiable reasoning
```

**Fix**: Practice **talking through problems out loud** (even when studying alone). Explain each step as if teaching a peer.

### 9.3 Mistake #3: Ignoring Fit & Culture

**Problem**: Quant firms have **distinct cultures**, and fit matters more than you think.

**Firm archetypes**:
```
Jane Street:
- Values: Teaching, collaboration, curiosity
- Interview: Heavy on explanation ("teach me how you solved this")
- Culture: Flat hierarchy, everyone has voice

Citadel:
- Values: Performance, competitiveness, excellence
- Interview: Harder problems, faster pace
- Culture: Meritocratic, higher pressure

Two Sigma:
- Values: Academic rigor, systematic research
- Interview: PhD-level math, coding depth
- Culture: Scientist-friendly, publish research papers
```

**Fix**: Research each firm's culture before applying. Tailor your interview approach to their values.

### 9.4 Mistake #4: Starting Too Late

**Problem**: Most successful quants start preparing in **sophomore/junior year of undergrad**, not senior year.

**Timeline comparison**:
```
Early starter (sophomore year):
- 2 years of preparation → strong interview performance → top firm offer

Late starter (senior year):
- 6 months of rushed preparation → mediocre performance → tier 2 firm or no offer
- Gap year to prepare → apply again next cycle (delayed by 1 year)
```

**Fix**: If you're a junior/senior and just discovering quant, consider:
1. **Gap year**: Spend 12 months intensely preparing (follow Month 1-12 roadmap)
2. **Master's program**: MFE or CS Master's → recruit during program
3. **Software engineering first**: Join big tech (Google, Meta) → lateral to quant after 2 years

---

## Part X: The Contrarian View - Should You Even Become a Quant?

### 10.1 Reasons NOT to Pursue Quant

**1. You're not intrinsically motivated by math/markets**
- If you're doing this purely for money, you'll burn out
- Quant work is intellectually demanding; passion for problem-solving is essential

**2. You value work-life balance above all**
- 50-70 hour weeks are standard
- Weekend work during market volatility is common
- Stress from managing live capital

**3. You want to "change the world"**
- Quant finance is zero-sum (one trader's profit = another's loss)
- Social impact is limited (you're moving money around, not curing cancer)

**4. You struggle with rejection**
- Interview acceptance rate: 1-3%
- 50-100 applications to land 1 offer is normal
- You'll face more rejection than success

### 10.2 Alternative Careers (Similar Comp, Different Trade-offs)

**A. Big Tech (Google, Meta, Amazon)**
- **Comp**: $250K-$400K starting (L4/L5 engineer)
- **Pros**: Better work-life balance, more job security, transferable skills
- **Cons**: Lower ceiling ($500K-$1M cap unless you reach Staff/Principal), slower meritocracy

**B. Quantitative Research at Tech (Meta AI, Google DeepMind)**
- **Comp**: $300K-$500K starting (research scientist)
- **Pros**: Publish papers, work on cutting-edge ML, intellectual prestige
- **Cons**: Slower pace than finance, less direct tie between work and comp

**C. Data Science at Unicorn Startups**
- **Comp**: $150K-$250K + equity (can be worth millions if IPO)
- **Pros**: Equity upside, impact on product, faster career growth
- **Cons**: Risk (90% of startups fail), lower base comp

**D. Academia (Become a Professor)**
- **Comp**: $80K-$150K (professor salary) + consulting income ($200K-$500K if active)
- **Pros**: Intellectual freedom, tenure security, teach next generation
- **Cons**: Low comp relative to industry, slow career progression (8-12 years to tenure)

---

## Part XI: Actionable Next Steps (What to Do Tomorrow)

### 11.1 If You're an Undergraduate (Sophomore/Junior)

**Immediate actions** (this week):
1. ✅ Enroll in advanced probability course (next semester)
2. ✅ Join quant club or create one (organize weekly problem-solving sessions)
3. ✅ Start solving 3 probability problems daily (from "Heard on the Street")
4. ✅ Build a simple trading strategy in Python (moving average crossover)
5. ✅ Connect with 5 quants on LinkedIn (ask for informational interviews)

**This semester**:
- Apply for quant summer internships (Jane Street, SIG, Citadel, Two Sigma)
- Build project portfolio on GitHub (3 complete projects: algo trading, option pricer, portfolio optimizer)
- Compete in quantitative trading competitions (QuantConnect, Kaggle)

**This summer**:
- If you land internship → convert to full-time offer (70% conversion rate)
- If no internship → self-study roadmap (Month 1-3 plan above), apply for fall recruiting

### 11.2 If You're a Grad Student (Master's/PhD)

**Immediate actions**:
1. ✅ Identify transferable skills from your research (numerical methods, optimization, ML)
2. ✅ Complete 50 LeetCode problems (proof of coding proficiency)
3. ✅ Read 3 quant finance papers (to speak the language in interviews)
4. ✅ Build 1 quant project showcasing your PhD skills (e.g., apply your thesis methods to stock prediction)
5. ✅ Attend quant recruiting events (most firms have PhD-specific tracks)

**Recruiting timeline**:
- Target: Quant hedge funds (Two Sigma, DE Shaw, Millennium) → they value PhDs more than prop trading firms
- Apply: Fall recruiting (September-November for June start dates)
- Prep: 3-6 months of interview preparation (probability, coding, finance basics)

### 11.3 If You're a Career Switcher (Software Engineer → Quant)

**Your advantages**:
- ✅ Strong coding skills (already ahead of math PhDs)
- ✅ Systems thinking (understand production infrastructure)
- ✅ Industry experience (mature, lower management risk)

**Your gaps**:
- ❌ Probability & statistics (must self-study intensively)
- ❌ Finance knowledge (need crash course)
- ❌ Age bias (firms prefer younger candidates for cultural fit)

**Recommended path**:

```
Option 1: Direct application (12-month prep)
├─ Month 1-6: Self-study math (probability, statistics, linear algebra)
├─ Month 7-9: Finance deep dive (derivatives, portfolio theory)
├─ Month 10-12: Interview prep (mock interviews, applications)
└─ Target: Quant developer roles (play to your coding strength)

Option 2: Master's bridge (2-year commitment)
├─ Enroll in part-time MFE program (Baruch, NYU, Columbia)
├─ Keep day job (self-fund tuition)
├─ Network with classmates and alumni (career switchers)
└─ Recruit during Year 2 (formal on-campus recruiting)

Option 3: Hybrid software/quant role (3-year gradual transition)
├─ Join fintech company (Robinhood, Plaid, Stripe) → learn finance domain
├─ Transfer internally to quant team (after proving technical value)
├─ Build quant skills on the job (research projects, backtesting)
└─ Lateral to pure quant firm after 2-3 years
```

**Reality check**: Career switchers have ~5-10% success rate at top firms (vs 1-3% base rate). Your coding skills help, but math gap is real.

---

## Part XII: Resources & Community

### 12.1 Essential Books

**Probability & Statistics**:
1. **"Introduction to Probability"** - Blitzstein & Hwang (Harvard course textbook)
2. **"All of Statistics"** - Larry Wasserman (CMU)
3. **"Probability and Random Processes"** - Grimmett & Stirzaker

**Quant Interview Prep**:
4. **"A Practical Guide to Quantitative Finance Interviews"** - Xinfeng Zhou
5. **"Heard on the Street"** - Timothy Crack
6. **"Quant Job Interview Questions and Answers"** - Mark Joshi

**Finance**:
7. **"Options, Futures, and Other Derivatives"** - John Hull
8. **"Active Portfolio Management"** - Grinold & Kahn

**Programming**:
9. **"C++ Primer"** - Lippman (if targeting HFT)
10. **"Python for Data Analysis"** - Wes McKinney (pandas creator)

### 12.2 Online Resources

**Learning platforms**:
- **QuantStart**: Free tutorials on quant trading (www.quantstart.com)
- **QuantConnect**: Algo trading platform with free backtesting (www.quantconnect.com)
- **Khan Academy**: Probability, statistics, linear algebra (free video lectures)
- **Coursera**: "Financial Engineering and Risk Management" (Columbia)

**Practice platforms**:
- **LeetCode**: Coding interview prep (premium: $35/month)
- **Brilliant**: Interactive math/probability problems ($25/month)
- **GlassDoor**: Real interview questions from quants at top firms (free)

**Communities**:
- **r/quant**: Reddit community (50K+ members, active discussions)
- **QuantNet**: Forums for MFE students and quant professionals
- **Wilmott**: Quantitative finance forums (30+ years old, deep archives)

### 12.3 Networking & Mentorship

**How to find mentors**:
1. **LinkedIn outreach**: Message quants at target firms (response rate: ~10-20%)
2. **University alumni**: Leverage school connections (higher response rate)
3. **Quant conferences**: Attend and network (e.g., Battle of the Quants, QuantCon)
4. **Informational interviews**: Offer to buy coffee/lunch (30-minute time ask)

**What to ask mentors**:
- "What's the most important skill you've developed since starting?"
- "What do you wish you knew before entering the field?"
- "How do you stay current with research and market developments?"
- "Can you critique my resume and suggest improvements?"

---

## Conclusion: The Brutal Honesty

**Becoming a quant is HARD**. You'll spend 2-4 years intensely studying, face 95%+ rejection rates, and compete against IMO medalists and MIT PhDs. **Most people who try will fail.**

**But**: If you have strong technical aptitude, obsessive work ethic, and genuine interest in markets + math, it's **absolutely achievable**. The roadmap is clear. The resources exist. The only question is: **Are you willing to put in the work?**

**What separates successful candidates from failures**:
- ❌ Failures: Study sporadically, give up after rejections, focus on "hacks" and shortcuts
- ✅ Winners: Study systematically, treat rejections as learning opportunities, master fundamentals deeply

**The meta-skill**: Quant firms aren't just hiring for math/programming ability. They're hiring for **learning velocity** - can you master new domains quickly? The interview process itself is a test of this meta-skill.

**Final advice**: If you're reading this and thinking "This sounds impossibly hard," you're right. It is. But every senior quant started exactly where you are. The difference is: **they started**.

---

## Appendix: Quick Reference Cheat Sheets

### A.1 Probability Formulas (Memorize These)

```
Expected value: E[X] = ∑ x P(X = x)  (discrete), ∫ x f(x) dx  (continuous)

Variance: Var(X) = E[(X - μ)²] = E[X²] - (E[X])²

Conditional probability: P(A|B) = P(A ∩ B) / P(B)

Bayes' theorem: P(A|B) = P(B|A) P(A) / P(B)

Law of total expectation: E[X] = E[E[X|Y]]

Covariance: Cov(X, Y) = E[(X - μ_X)(Y - μ_Y)] = E[XY] - E[X]E[Y]

Correlation: ρ(X, Y) = Cov(X, Y) / (σ_X σ_Y)

Independence: P(A ∩ B) = P(A) P(B) ⟺ E[XY] = E[X] E[Y]
```

### A.2 Black-Scholes Formulas (Must Know Cold)

```
Call option value:
C = S Φ(d₁) - K e^(-rT) Φ(d₂)

Put option value:
P = K e^(-rT) Φ(-d₂) - S Φ(-d₁)

Where:
d₁ = (ln(S/K) + (r + σ²/2)T) / (σ√T)
d₂ = d₁ - σ√T

Greeks:
Δ_call = Φ(d₁)
Δ_put = Φ(d₁) - 1
Γ = φ(d₁) / (S σ √T)  [same for call and put]
ν = S φ(d₁) √T  [same for call and put]
Θ_call = -S φ(d₁) σ / (2√T) - r K e^(-rT) Φ(d₂)

Put-call parity:
C - P = S - K e^(-rT)
```

### A.3 Time Complexity Cheat Sheet (Coding Interviews)

```
O(1): Hash table lookup, array access
O(log n): Binary search, balanced BST operations
O(n): Linear scan, single loop through array
O(n log n): Sorting (merge sort, heap sort), divide-and-conquer
O(n²): Nested loops, bubble sort, pairwise comparisons
O(2ⁿ): Recursive backtracking, subset enumeration

Interview target: Recognize when O(n²) naive solution can be optimized to O(n) or O(n log n)
```

---

**Final Word**: The quant career is a marathon, not a sprint. Start today. Study systematically. Apply relentlessly. Iterate on feedback. In 2-4 years, you'll be managing millions of dollars in capital and earning more than 99% of knowledge workers.

**The only question is: Will you start?**

---

## References & Further Reading

### Primary Learning Resources
1. **QuantStart**: Complete algorithmic trading tutorials - https://www.quantstart.com
2. **QuantConnect**: Free backtesting platform - https://www.quantconnect.com
3. **QuantNet**: MFE program rankings and forums - https://quantnet.com

### Academic Papers (Foundational)
4. Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"
5. Fama & French (1993), "Common Risk Factors in the Returns on Stocks and Bonds"
6. Black & Scholes (1973), "The Pricing of Options and Corporate Liabilities"

### Industry Blogs & Newsletters
7. **Quantocracy**: Daily links to quant blogs - http://quantocracy.com
8. **Quantitative Research Blog**: Career advice from industry practitioners

### Recruiting Resources
9. **Jane Street Interview Guide**: https://www.janestreet.com/join-jane-street/interview-preparation/
10. **Citadel Careers**: https://www.citadel.com/careers/
11. **HRT Algorithm Developer**: https://www.hudsonrivertrading.com/careers/

---

**Acknowledgments**: This report synthesizes publicly available information about quantitative finance careers, drawing from industry job postings, interview experiences shared by candidates, and career guides published by practitioners. All code examples and technical content are original educational reconstructions.

**Disclaimer**: Compensation figures are estimates based on 2026 market data from levels.fyi, Glassdoor, and industry sources. Actual offers vary by candidate profile, firm, and market conditions.

---

**Tags**: `#QuantitativeFinance` `#AlgorithmicTrading` `#CareerGuide` `#FinancialEngineering` `#DerivativesPricing` `#HFT` `#SystematicTrading` `#MathematicalFinance` `#InterviewPrep` `#STEM`
