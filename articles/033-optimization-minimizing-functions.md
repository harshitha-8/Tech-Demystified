# The Mathematics of Optimization: From Gradient Descent to Newton's Method

### A comprehensive guide to first-order and second-order optimization algorithms for machine learning and deep learning

Optimization is the **mathematical engine** that powers every machine learning system. Whether training a simple logistic regression model or fine-tuning a 175-billion parameter language model, the core task remains the same: **minimize a loss function** by iteratively adjusting parameters.

Yet the difference between naive gradient descent and sophisticated optimizers can mean the difference between:
- Training that converges in hours vs. weeks
- Models that generalize well vs. overfit badly
- Reaching global minima vs. getting trapped in poor local minima
- Memory-efficient training vs. computational explosion

This article presents the complete mathematical foundation of optimization for machine learning, from first principles through modern algorithms used in production systems.

**Based on MIT 18.065 Lecture 21** ([source](https://ickma.dev/Math/MIT18.065/mit18065-lecture21-minimizing-function.html)) and contemporary research, we'll cover:

1. **Mathematical Foundations**: Taylor expansion, gradients, Hessian, and Jacobian matrices
2. **First-Order Methods**: Gradient Descent, SGD, Momentum, RMSProp, Adam
3. **Second-Order Methods**: Newton's Method, Quasi-Newton (L-BFGS), Conjugate Gradient
4. **Convexity Theory**: Why convex optimization is "easy" and how it relates to deep learning
5. **Practical Considerations**: When to use what, computational trade-offs, hyperparameter tuning

**Why this matters**: The choice of optimizer is **not** a minor implementation detail. It fundamentally determines:
- Training time and cost (GPU hours = money)
- Model performance (convergence quality affects generalization)
- Scalability (some methods don't scale to billions of parameters)

## Part 1: Mathematical Foundations

### The Taylor Series Expansion

**Every optimization algorithm** is fundamentally based on approximating a function locally and stepping toward its minimum. The mathematical tool that makes this possible is the **Taylor series expansion**.

#### One-Dimensional Case

For a scalar function \( F: \mathbb{R} \to \mathbb{R} \), the Taylor expansion around point \( x \) is:

\[
F(x + \Delta x) \approx F(x) + \Delta x \cdot \frac{dF}{dx} + \frac{1}{2}(\Delta x)^2 \cdot \frac{d^2F}{dx^2}
\]

**Components**:
- **Zeroth-order term**: \( F(x) \) — function value at current point
- **First-order term**: \( \Delta x \cdot \frac{dF}{dx} \) — linear approximation (gradient)
- **Second-order term**: \( \frac{1}{2}(\Delta x)^2 \cdot \frac{d^2F}{dx^2} \) — curvature correction

**Intuition**: 
- The first derivative tells you the **direction** to move (downhill)
- The second derivative tells you **how fast the direction changes** (curvature)

#### Multivariable Generalization

For \( F: \mathbb{R}^n \to \mathbb{R} \) (the case in machine learning where \( n \) is the number of parameters):

\[
F(x + \Delta x) \approx F(x) + (\Delta x)^\top \nabla_x F + \frac{1}{2}(\Delta x)^\top H \, \Delta x
\]

Where:
- **Gradient** \( \nabla_x F \in \mathbb{R}^n \): Vector of first partial derivatives
  \[
  \nabla_x F = \begin{bmatrix}
  \frac{\partial F}{\partial x_1} \\
  \frac{\partial F}{\partial x_2} \\
  \vdots \\
  \frac{\partial F}{\partial x_n}
  \end{bmatrix}
  \]

- **Hessian** \( H \in \mathbb{R}^{n \times n} \): Matrix of second partial derivatives
  \[
  H_{ij} = \frac{\partial^2 F}{\partial x_i \partial x_j}
  \]

**Critical property**: Under standard smoothness assumptions (twice-differentiable functions), the Hessian is **symmetric**: \( H_{ij} = H_{ji} \).

### The Gradient: Direction of Steepest Ascent

**Theorem**: The gradient \( \nabla F(x) \) points in the direction of **maximum increase** of \( F \).

**Corollary**: To **minimize** \( F \), move in the direction of \( -\nabla F(x) \) (negative gradient).

**Proof sketch**: For any unit vector \( u \) (\( \|u\| = 1 \)), the directional derivative is:
\[
\frac{\partial F}{\partial u} = \nabla F(x)^\top u = \|\nabla F(x)\| \cdot \|u\| \cdot \cos(\theta) = \|\nabla F(x)\| \cos(\theta)
\]

This is maximized when \( \cos(\theta) = 1 \), i.e., \( u \) is parallel to \( \nabla F(x) \).

### The Hessian: Curvature Information

The Hessian captures **local curvature** of the loss surface. It tells us:

**1. Whether a critical point is a minimum, maximum, or saddle point**:
- If \( \nabla F(x^*) = 0 \) and \( H \) is positive definite (\( v^\top H v > 0 \) for all \( v \neq 0 \)) → **local minimum**
- If \( H \) is negative definite → **local maximum**
- If \( H \) has mixed eigenvalues (some positive, some negative) → **saddle point**

**2. Condition number and convergence speed**:
- The **condition number** \( \kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}} \) determines how "well-shaped" the loss surface is
- High condition number → "ravine-like" surface → slow convergence for gradient descent
- Low condition number → "bowl-like" surface → fast convergence

**Example**: Quadratic function \( F(x) = \frac{1}{2} x^\top A x - b^\top x \)

The Hessian is constant: \( H = A \)

If \( A \) has eigenvalues \( \lambda_1 = 100, \lambda_2 = 1 \):
- Gradient descent converges slowly (zigzags in ravine)
- Condition number \( \kappa = 100/1 = 100 \) indicates ill-conditioning

### The Jacobian: Derivatives of Vector Functions

For a vector-valued function \( f: \mathbb{R}^n \to \mathbb{R}^m \), the **Jacobian matrix** \( J \in \mathbb{R}^{m \times n} \) is:

\[
J_{ik} = \frac{\partial f_i}{\partial x_k}
\]

**Role in optimization**: When solving \( f(x) = 0 \) (root-finding), the Jacobian is used to linearize:
\[
f(x + \Delta x) \approx f(x) + J(x) \Delta x
\]

**Connection to Newton's Method**: For minimization, set \( f = \nabla F \). Then:
- The Jacobian of \( \nabla F \) is the Hessian of \( F \)
- \( J(\nabla F) = H(F) \)

This is why Newton's method for minimizing \( F \) uses the Hessian!

## Part 2: First-Order Methods (Gradient-Based)

### Vanilla Gradient Descent

**Update rule**:
\[
x_{k+1} = x_k - \alpha \nabla F(x_k)
\]

Where \( \alpha > 0 \) is the **learning rate** (step size).

**Algorithm**:

```python
def gradient_descent(f, grad_f, x0, alpha, max_iters=1000, tol=1e-6):
    """
    Minimize function f using gradient descent.
    
    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        alpha: Learning rate
        max_iters: Maximum iterations
        tol: Convergence tolerance
    """
    x = x0.copy()
    history = [f(x)]
    
    for k in range(max_iters):
        grad = grad_f(x)
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {k} iterations")
            break
        
        # Update
        x = x - alpha * grad
        history.append(f(x))
    
    return x, history
```

**Convergence analysis** for convex, \( L \)-Lipschitz smooth functions (\( \|\nabla F(x) - \nabla F(y)\| \leq L \|x - y\| \)):

With learning rate \( \alpha = \frac{1}{L} \):
\[
F(x_k) - F(x^*) \leq \frac{L \|x_0 - x^*\|^2}{2k}
\]

This is \( O(1/k) \) convergence — **sublinear**.

**Challenges**:
- ❌ **Learning rate selection**: Too large → divergence; too small → slow convergence
- ❌ **Saddle points**: Can get stuck where \( \nabla F = 0 \) but not at minimum
- ❌ **Ravines**: Oscillates perpendicular to optimal direction

### Stochastic Gradient Descent (SGD)

**Key innovation**: Use **mini-batches** instead of full dataset.

For loss \( L(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i) \):

**Full-batch GD**:
\[
\theta_{k+1} = \theta_k - \alpha \cdot \frac{1}{N} \sum_{i=1}^N \nabla_\theta \ell(f_\theta(x_i), y_i)
\]

**Mini-batch SGD**:
\[
\theta_{k+1} = \theta_k - \alpha \cdot \frac{1}{B} \sum_{i \in \mathcal{B}_k} \nabla_\theta \ell(f_\theta(x_i), y_i)
\]

Where \( \mathcal{B}_k \subset \{1, ..., N\} \) is a randomly sampled mini-batch of size \( B \ll N \).

**Why this works**:

```python
import numpy as np

def sgd(loss_fn, grad_fn, X, y, theta0, alpha, batch_size, epochs):
    """
    Stochastic Gradient Descent with mini-batches.
    
    Args:
        loss_fn: Loss function(theta, X, y)
        grad_fn: Gradient function(theta, X_batch, y_batch)
        X, y: Training data
        theta0: Initial parameters
        alpha: Learning rate
        batch_size: Mini-batch size
        epochs: Number of passes through data
    """
    theta = theta0.copy()
    N = len(X)
    history = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(N)
        
        for i in range(0, N, batch_size):
            # Get mini-batch
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            # Compute gradient on mini-batch
            grad = grad_fn(theta, X_batch, y_batch)
            
            # Update
            theta = theta - alpha * grad
        
        # Track progress
        loss = loss_fn(theta, X, y)
        history.append(loss)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
    
    return theta, history
```

**Benefits**:
- ✅ **Computational efficiency**: \( O(B) \) instead of \( O(N) \) per update
- ✅ **Stochastic noise**: Helps escape sharp local minima
- ✅ **Memory efficiency**: Don't need to load entire dataset

**Trade-offs**:
- ⚠️ **Noisy updates**: Higher variance than full-batch
- ⚠️ **Requires learning rate decay**: Noise prevents exact convergence

### Momentum: Accelerating Convergence

**Problem with vanilla SGD**: Oscillates in directions of high curvature.

**Solution**: Add **momentum** — accumulate velocity from past gradients.

**Update rule**:
\[
\begin{aligned}
v_{k+1} &= \beta v_k + \nabla F(x_k) \\
x_{k+1} &= x_k - \alpha v_{k+1}
\end{aligned}
\]

Where \( \beta \in [0, 1) \) is the momentum coefficient (typically \( \beta = 0.9 \)).

**Physical analogy**: Ball rolling down a hill
- Gradient = slope of hill
- Velocity = speed of ball (accumulates over time)
- Momentum = resistance to changing direction

**Rewritten as exponential moving average**:
\[
v_{k+1} = \beta v_k + (1 - \beta) \nabla F(x_k) \quad \text{(if we rescale)}
\]

But standard formulation doesn't normalize, so:
\[
v_k = \nabla F(x_k) + \beta \nabla F(x_{k-1}) + \beta^2 \nabla F(x_{k-2}) + ...
\]

**Implementation**:

```python
def sgd_momentum(loss_fn, grad_fn, X, y, theta0, alpha, beta, batch_size, epochs):
    """
    SGD with Momentum.
    
    Args:
        beta: Momentum coefficient (typically 0.9)
    """
    theta = theta0.copy()
    v = np.zeros_like(theta)  # Initialize velocity
    N = len(X)
    
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            grad = grad_fn(theta, X_batch, y_batch)
            
            # Momentum update
            v = beta * v + grad
            theta = theta - alpha * v
    
    return theta
```

**Why it works**:
- ✅ **Dampens oscillations**: Averages out oscillatory components
- ✅ **Speeds up convergence**: Accumulates velocity in consistent directions
- ✅ **Better conditioning**: Effectively reduces condition number

**Nesterov Accelerated Gradient** (NAG): Look-ahead variant
\[
\begin{aligned}
v_{k+1} &= \beta v_k + \nabla F(x_k - \alpha \beta v_k) \\
x_{k+1} &= x_k - \alpha v_{k+1}
\end{aligned}
\]

Evaluates gradient at "predicted" future position → better convergence.

### RMSProp: Adaptive Learning Rates

**Problem**: Fixed learning rate treats all parameters equally, but loss surface may have different curvatures in different dimensions.

**Solution**: **Root Mean Square Propagation (RMSProp)** — adapt learning rate per parameter based on historical gradient magnitudes.

**Update rule**:
\[
\begin{aligned}
s_{k+1} &= \gamma s_k + (1 - \gamma) (\nabla F(x_k))^2 \\
x_{k+1} &= x_k - \frac{\alpha}{\sqrt{s_{k+1}} + \epsilon} \odot \nabla F(x_k)
\end{aligned}
\]

Where:
- \( s_k \): Exponential moving average of **squared gradients** (element-wise)
- \( \gamma \approx 0.9 \): Decay rate
- \( \epsilon \approx 10^{-8} \): Numerical stability constant
- \( \odot \): Element-wise multiplication

**Intuition**: 
- Parameters with **large historical gradients** → small effective learning rate (divide by large \( \sqrt{s} \))
- Parameters with **small historical gradients** → large effective learning rate

**Implementation**:

```python
def rmsprop(loss_fn, grad_fn, X, y, theta0, alpha, gamma, batch_size, epochs, epsilon=1e-8):
    """
    RMSProp optimizer.
    
    Args:
        gamma: Decay rate for squared gradient average (typically 0.9)
        epsilon: Numerical stability constant
    """
    theta = theta0.copy()
    s = np.zeros_like(theta)  # Initialize squared gradient average
    N = len(X)
    
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            grad = grad_fn(theta, X_batch, y_batch)
            
            # Update squared gradient average
            s = gamma * s + (1 - gamma) * (grad ** 2)
            
            # Adaptive update
            theta = theta - alpha * grad / (np.sqrt(s) + epsilon)
    
    return theta
```

**Benefits**:
- ✅ **Handles different scales**: Normalizes updates across dimensions
- ✅ **Reduces need for manual tuning**: Less sensitive to initial learning rate
- ✅ **Works well in non-stationary settings**: Adapts to changing gradients

### Adam: Adaptive Moment Estimation

**The gold standard** for deep learning optimization. Combines **Momentum** (first moment) and **RMSProp** (second moment).

**Update rule**:
\[
\begin{aligned}
m_{k+1} &= \beta_1 m_k + (1 - \beta_1) \nabla F(x_k) \\
v_{k+1} &= \beta_2 v_k + (1 - \beta_2) (\nabla F(x_k))^2 \\
\hat{m}_{k+1} &= \frac{m_{k+1}}{1 - \beta_1^{k+1}} \\
\hat{v}_{k+1} &= \frac{v_{k+1}}{1 - \beta_2^{k+1}} \\
x_{k+1} &= x_k - \frac{\alpha}{\sqrt{\hat{v}_{k+1}} + \epsilon} \hat{m}_{k+1}
\end{aligned}
\]

Where:
- \( m_k \): First moment (mean of gradients) — **momentum component**
- \( v_k \): Second moment (uncentered variance) — **RMSProp component**
- \( \hat{m}_k, \hat{v}_k \): **Bias-corrected** estimates (important in early iterations)
- \( \beta_1 \approx 0.9 \): Exponential decay for first moment
- \( \beta_2 \approx 0.999 \): Exponential decay for second moment
- \( \alpha \): Learning rate (often \( 10^{-3} \) or \( 10^{-4} \))
- \( \epsilon \approx 10^{-8} \): Numerical stability

**Why bias correction?** 
Initially, \( m_0 = 0, v_0 = 0 \). Without correction, estimates are biased toward zero in early iterations.

**Full implementation**:

```python
class Adam:
    def __init__(self, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam optimizer.
        
        Args:
            alpha: Learning rate
            beta1: Exponential decay rate for first moment (momentum)
            beta2: Exponential decay rate for second moment (RMSProp)
            epsilon: Numerical stability constant
        """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def update(self, theta, grad):
        """Update parameters given gradient."""
        # Initialize moments on first call
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        
        self.t += 1
        
        # Update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        theta_new = theta - self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return theta_new

# Usage
def train_with_adam(loss_fn, grad_fn, X, y, theta0, batch_size, epochs):
    optimizer = Adam(alpha=1e-3)
    theta = theta0.copy()
    N = len(X)
    
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            grad = grad_fn(theta, X_batch, y_batch)
            theta = optimizer.update(theta, grad)
    
    return theta
```

**Why Adam dominates**:
- ✅ **Combines best of both worlds**: Momentum + adaptive learning rates
- ✅ **Robust hyperparameters**: Default values (\( \beta_1=0.9, \beta_2=0.999, \alpha=10^{-3} \)) work well across problems
- ✅ **Bias correction**: Ensures good initial behavior
- ✅ **Per-parameter adaptation**: Different effective learning rates for each parameter

**Modern variants**:
- **AdamW**: Adam with **decoupled weight decay** (better regularization)
- **RAdam**: **Rectified Adam** with warmup (fixes early training instability)
- **Lion**: **Sign-based** momentum optimizer (memory-efficient for large models)

## Part 3: Second-Order Methods

### Newton's Method: Using Curvature

**Motivation**: First-order methods only use gradient (slope). What if we also use **curvature** (how fast slope changes)?

**From Taylor expansion**, the second-order approximation is:
\[
F(x + \Delta x) \approx F(x) + (\Delta x)^\top \nabla F + \frac{1}{2} (\Delta x)^\top H \, \Delta x
\]

To find the minimum of this quadratic approximation, set derivative to zero:
\[
\frac{\partial}{\partial (\Delta x)} \left[ F(x) + (\Delta x)^\top \nabla F + \frac{1}{2} (\Delta x)^\top H \, \Delta x \right] = \nabla F + H \, \Delta x = 0
\]

Solving for \( \Delta x \):
\[
\Delta x = -H^{-1} \nabla F
\]

**Newton's Method update**:
\[
x_{k+1} = x_k - H(x_k)^{-1} \nabla F(x_k)
\]

**Comparison with gradient descent**:
- **Gradient Descent**: \( x_{k+1} = x_k - \alpha \nabla F(x_k) \)
- **Newton's Method**: \( x_{k+1} = x_k - H^{-1} \nabla F(x_k) \)

**Key difference**: Newton's method uses \( H^{-1} \) as an **adaptive, data-dependent learning rate matrix**.

**Example: Quadratic function**

For \( F(x) = \frac{1}{2} x^\top A x - b^\top x \):
- Gradient: \( \nabla F = Ax - b \)
- Hessian: \( H = A \) (constant)

Newton's method:
\[
x_{k+1} = x_k - A^{-1}(A x_k - b) = x_k - x_k + A^{-1} b = A^{-1} b = x^*
\]

**Converges in ONE iteration!** (for quadratic functions)

**Implementation** (basic version):

```python
def newtons_method(f, grad_f, hessian_f, x0, max_iters=100, tol=1e-6):
    """
    Newton's Method for unconstrained optimization.
    
    Args:
        f: Objective function
        grad_f: Gradient function
        hessian_f: Hessian matrix function
        x0: Initial point
        max_iters: Maximum iterations
        tol: Convergence tolerance
    """
    x = x0.copy()
    
    for k in range(max_iters):
        grad = grad_f(x)
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {k} iterations")
            break
        
        H = hessian_f(x)
        
        # Solve H * delta_x = -grad
        # (More stable than computing H^(-1) explicitly)
        delta_x = -np.linalg.solve(H, grad)
        
        # Newton step
        x = x + delta_x
    
    return x
```

### Quadratic Convergence

**Theorem**: Near a minimum \( x^* \) where \( H(x^*) \) is positive definite, Newton's method has **quadratic convergence**:

\[
\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|^2
\]

**What this means**: Error squares each iteration → **extremely fast convergence**.

**Example**:
- Iteration 1: Error = \( 10^{-1} \)
- Iteration 2: Error ≈ \( (10^{-1})^2 = 10^{-2} \)
- Iteration 3: Error ≈ \( (10^{-2})^2 = 10^{-4} \)
- Iteration 4: Error ≈ \( (10^{-4})^2 = 10^{-8} \)
- Iteration 5: Error ≈ \( (10^{-8})^2 = 10^{-16} \) (machine precision!)

**Contrast with gradient descent**: \( O(1/k) \) convergence — much slower.

### Why Not Use Newton in Deep Learning?

**The computational bottleneck**:

For a neural network with \( n \) parameters:
- **Gradient**: \( O(n) \) storage, \( O(n) \) computation per layer (backprop)
- **Hessian**: \( O(n^2) \) storage, \( O(n^3) \) computation (matrix inversion)

**Example**: GPT-3 has **175 billion parameters**
- Gradient: 175B floats = 700 GB (FP32)
- Hessian: \( (175 \times 10^9)^2 \) floats = \( 3.06 \times 10^{22} \) floats = **122 million petabytes** 😱

**Even a "small" model** with 1 million parameters:
- Gradient: 4 MB
- Hessian: \( 10^{12} \) floats = **4 TB** of memory

**Additional problems**:
- ❌ **Saddle points**: Hessian may not be positive definite (has negative eigenvalues)
- ❌ **Stochastic gradients**: Mini-batch noise makes exact Hessian meaningless
- ❌ **Non-quadratic loss**: Taylor approximation breaks down far from optimum

### Quasi-Newton Methods: L-BFGS

**Idea**: Approximate \( H^{-1} \) without computing Hessian explicitly.

**BFGS** (Broyden-Fletcher-Goldfarb-Shanno) builds approximation \( B_k \approx H^{-1} \) by observing how gradients change:

\[
s_k = x_{k+1} - x_k, \quad y_k = \nabla F(x_{k+1}) - \nabla F(x_k)
\]

**BFGS update**:
\[
B_{k+1} = \left( I - \frac{s_k y_k^\top}{y_k^\top s_k} \right) B_k \left( I - \frac{y_k s_k^\top}{y_k^\top s_k} \right) + \frac{s_k s_k^\top}{y_k^\top s_k}
\]

**L-BFGS** (Limited-memory BFGS): Don't store full matrix \( B_k \), just last \( m \) pairs \( (s_i, y_i) \).

**Memory**: \( O(nm) \) instead of \( O(n^2) \) (typically \( m = 10 \))

**Use cases**:
- ✅ Medium-scale problems (thousands to millions of parameters)
- ✅ Full-batch optimization (not stochastic mini-batches)
- ✅ When gradient evaluation is expensive (e.g., physics simulations)

**Implementation** (simplified):

```python
from scipy.optimize import minimize

def optimize_lbfgs(f, grad_f, x0):
    """
    Optimize using L-BFGS via scipy.
    """
    result = minimize(
        fun=f,
        x0=x0,
        method='L-BFGS-B',
        jac=grad_f,
        options={'maxiter': 1000}
    )
    return result.x
```

### Conjugate Gradient: Efficient Linear System Solver

**Use case**: When Newton or Quasi-Newton methods require solving large linear systems:
\[
H \, \Delta x = -\nabla F
\]

Instead of direct inversion (\( O(n^3) \)), use **Conjugate Gradient** (\( O(n^2) \) or better with sparsity).

**Key idea**: Solve \( Ax = b \) by iteratively constructing \( x \) from "conjugate" directions.

**Algorithm** (high-level):

```python
def conjugate_gradient(A, b, x0, max_iters=100, tol=1e-6):
    """
    Solve Ax = b using Conjugate Gradient.
    Assumes A is symmetric positive definite.
    """
    x = x0.copy()
    r = b - A @ x  # Residual
    p = r.copy()   # Search direction
    
    for k in range(max_iters):
        # Check convergence
        if np.linalg.norm(r) < tol:
            break
        
        # Step size
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        
        # Update x
        x = x + alpha * p
        
        # Update residual
        r_new = r - alpha * Ap
        
        # Compute new search direction
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        
        r = r_new
    
    return x
```

**Why it works**: For an \( n \)-dimensional positive definite system, CG converges in **at most \( n \) iterations** (often much fewer).

## Part 4: Convexity Theory

### Convex Sets

A set \( C \subseteq \mathbb{R}^n \) is **convex** if:
\[
x, y \in C, \; \forall \theta \in [0, 1] \implies \theta x + (1 - \theta) y \in C
\]

**Intuition**: Any line segment connecting two points in \( C \) lies entirely within \( C \).

**Examples**:
- ✅ Convex: Ball, half-space, polyhedron, affine subspace
- ❌ Non-convex: Ring, star shape, disjoint regions

### Convex Functions

A function \( F: \mathbb{R}^n \to \mathbb{R} \) is **convex** if:
\[
F(\theta x + (1 - \theta) y) \leq \theta F(x) + (1 - \theta) F(y), \quad \forall x, y, \; \theta \in [0, 1]
\]

**Geometric interpretation**: The function lies below the line segment connecting any two points on its graph.

**Equivalent conditions** (for differentiable functions):

**1. First-order condition**:
\[
F(y) \geq F(x) + \nabla F(x)^\top (y - x)
\]
The function lies above its tangent plane at every point.

**2. Second-order condition**:
\[
H(x) \succeq 0 \quad \text{(Hessian is positive semidefinite)}
\]

**Examples**:
- Convex: \( x^2 \), \( e^x \), \( -\log(x) \), \( \|x\|_2 \)
- Non-convex: \( x^3 \), \( \sin(x) \), \( x^4 - x^2 \)

### Why Convexity Matters: Global Optimality

**Fundamental theorem of convex optimization**:

> For a convex function \( F \) on a convex domain, **every local minimum is a global minimum**.

**Proof**: Suppose \( x^* \) is a local minimum but not global. Then \( \exists y \) such that \( F(y) < F(x^*) \).

By convexity, for \( \theta \in (0, 1) \):
\[
F(\theta x^* + (1 - \theta) y) \leq \theta F(x^*) + (1 - \theta) F(y) < F(x^*)
\]

This contradicts \( x^* \) being a local minimum! ∎

**Practical implication**: For convex problems, gradient descent **will find the global optimum** (given appropriate learning rate and enough iterations).

### Strongly Convex Functions

A function \( F \) is **\( \mu \)-strongly convex** if:
\[
F(y) \geq F(x) + \nabla F(x)^\top (y - x) + \frac{\mu}{2} \|y - x\|^2
\]

Equivalently: \( H(x) \succeq \mu I \) (all eigenvalues \( \geq \mu \)).

**Why it matters**: Strong convexity implies **faster convergence**.

For \( \mu \)-strongly convex and \( L \)-smooth functions, gradient descent with \( \alpha = 1/L \) has:
\[
F(x_k) - F(x^*) \leq \left( 1 - \frac{\mu}{L} \right)^k \cdot [F(x_0) - F(x^*)]
\]

This is **linear convergence** (exponentially fast), not sublinear!

### Non-Convex Optimization (Deep Learning Reality)

**Bad news**: Neural network loss surfaces are **highly non-convex**:
- Multiple local minima
- Saddle points (Hessian has mixed eigenvalues)
- Plateaus (gradients near zero)

**Good news**: Empirical observations suggest:
- **Over-parametrization**: With enough parameters, many local minima have similar (good) loss values
- **SGD noise**: Stochastic gradients help escape poor local minima
- **Implicit regularization**: SGD/Adam find solutions that generalize well, even if not global optima

**Modern understanding**: We don't need global optima—we need solutions that **generalize to unseen data**.

## Part 5: Practical Considerations

### Decision Tree: Which Optimizer to Use?

```
Problem size < 10K parameters?
  └─ YES → Try L-BFGS (full-batch, second-order)
  └─ NO ↓

Training neural network?
  └─ YES → Use Adam (default choice)
       - Start with α = 1e-3, β₁ = 0.9, β₂ = 0.999
       - Add learning rate decay (cosine or step)
       - Consider AdamW for weight decay
  └─ NO ↓

Need guaranteed convergence (convex problem)?
  └─ YES → Gradient Descent with line search
  └─ NO ↓

Fine-tuning pre-trained model?
  └─ YES → SGD with momentum (often better generalization)
       - α = 1e-4 to 1e-3, β = 0.9
  └─ NO ↓

Memory-constrained (large models)?
  └─ YES → Lion optimizer (sign-based, memory-efficient)
```

### Hyperparameter Tuning

**Learning rate** (\( \alpha \)):
- **Too high**: Divergence, loss explodes
- **Too low**: Slow convergence, may not reach optimum
- **Sweet spot**: Largest value that doesn't diverge

**Finding good learning rate** (Leslie Smith's LR range test):

```python
def learning_rate_finder(model, train_loader, min_lr=1e-7, max_lr=10, num_iters=100):
    """
    Find good learning rate range by gradually increasing LR and tracking loss.
    """
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_iters)
    losses = []
    
    for lr in lrs:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss = train_one_batch(model, train_loader, optimizer)
        losses.append(loss)
        
        # Stop if loss explodes
        if len(losses) > 1 and losses[-1] > 4 * losses[-2]:
            break
    
    # Plot and choose LR where loss decreases fastest
    plt.plot(lrs[:len(losses)], losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()
    
    return lrs, losses
```

**Rule of thumb**: Choose LR slightly left of the minimum loss point.

### Learning Rate Schedules

**Step decay**:
\[
\alpha_k = \alpha_0 \cdot \gamma^{\lfloor k / s \rfloor}
\]
Reduce LR by factor \( \gamma \) every \( s \) epochs (e.g., \( \gamma = 0.5, s = 30 \)).

**Exponential decay**:
\[
\alpha_k = \alpha_0 \cdot e^{-\lambda k}
\]

**Cosine annealing**:
\[
\alpha_k = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min}) \left( 1 + \cos\left( \frac{k}{T} \pi \right) \right)
\]

Smooth decay from \( \alpha_{\max} \) to \( \alpha_{\min} \) over \( T \) iterations.

**1cycle policy** (Leslie Smith):
- Warmup: Linearly increase LR from low to high
- Cooldown: Decrease LR to very low value
- Used in FastAI, achieves fast convergence

```python
from torch.optim.lr_scheduler import OneCycleLR

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=100,
    steps_per_epoch=len(train_loader)
)

for epoch in range(100):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update LR every batch
```

### Gradient Clipping

**Problem**: Gradients can explode in RNNs or very deep networks.

**Solution**: Clip gradient norm to maximum value.

```python
# Gradient clipping by norm
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Gradient clipping by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Batch Size Effects

**Trade-offs**:
- **Large batches** (e.g., 256-1024):
  - ✅ Faster training (better GPU utilization)
  - ✅ More stable gradients (less noise)
  - ❌ May generalize worse ("sharp minima")
  - ❌ Requires more memory

- **Small batches** (e.g., 16-64):
  - ✅ Better generalization ("flat minima")
  - ✅ Less memory usage
  - ❌ Slower training (more iterations)
  - ❌ Noisier gradients

**Linear scaling rule** (Goyal et al., Facebook AI):
> When multiplying batch size by \( k \), multiply learning rate by \( k \).

**Example**: If batch size 32 works with LR = 0.001, then batch size 256 should use LR = 0.008.

## Summary: The Optimization Landscape

### Algorithmic Hierarchy

```
First-Order Methods (O(n) memory):
├─ Gradient Descent (baseline)
├─ SGD (stochastic, mini-batch)
├─ Momentum (accelerated, dampened)
├─ RMSProp (adaptive per-parameter LR)
└─ Adam (momentum + adaptive LR) ← DEFAULT CHOICE

Second-Order Methods (O(n²) or approximations):
├─ Newton's Method (exact Hessian, O(n³) compute)
├─ Quasi-Newton (L-BFGS, O(n) memory approximation)
└─ Conjugate Gradient (efficient linear solver)
```

### Convergence Rates Summary

| Method | Convex | Strongly Convex | Non-Convex |
|--------|--------|-----------------|------------|
| **Gradient Descent** | O(1/k) | Linear O(ρᵏ) | Local convergence |
| **Momentum** | O(1/k²) (Nesterov) | Accelerated linear | Better than GD |
| **Adam** | O(1/√k) (stochastic) | Practical linear | Robust |
| **Newton** | Quadratic (near opt.) | Superlinear | Requires PD Hessian |

### Key Takeaways

**1. Default recommendation**: **Adam** with learning rate decay
- Works well out-of-the-box
- Robust across problem types
- Minimal hyperparameter tuning

**2. For fine-tuning**: **SGD with momentum**
- Often generalizes better than Adam on pre-trained models
- Requires more careful LR tuning

**3. For small-scale**: **L-BFGS**
- Fast convergence on problems with < 10K parameters
- Not suitable for mini-batch stochastic optimization

**4. Theory vs. Practice**:
- Convex theory provides intuition
- Deep learning is non-convex, but SGD/Adam work empirically
- Generalization matters more than finding exact optima

**5. Learning rate is critical**:
- Use LR range test or cyclical schedules
- Always use LR decay for final convergence

---

*This article is part of the Tech Demystified series. For more articles on ML engineering, optimization, and deep learning fundamentals, see the [Tech Demystified repository](https://github.com/harshitha-8/Tech-Demystified).*

## References and Further Reading

**Primary Source**:
- MIT 18.065 Lecture 21 - Minimizing a Function: https://ickma.dev/Math/MIT18.065/mit18065-lecture21-minimizing-function.html
- MIT OpenCourseWare 18.065: Matrix Methods in Data Analysis, Signal Processing, and Machine Learning

**Foundational Papers**:
- Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization" - https://arxiv.org/abs/1412.6980
- Nesterov (1983): "A method for solving the convex programming problem with convergence rate O(1/k²)"
- Ruder (2016): "An overview of gradient descent optimization algorithms" - https://ruder.io/optimizing-gradient-descent/

**Modern Developments**:
- Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization" (AdamW)
- Liu & Chen (2023): "Symbolic Discovery of Optimization Algorithms" (Lion)
- Smith (2017): "Cyclical Learning Rates for Training Neural Networks"

**Textbooks**:
- Boyd & Vandenberghe: "Convex Optimization" (free PDF available)
- Nocedal & Wright: "Numerical Optimization" (definitive reference)
- Goodfellow et al.: "Deep Learning" Chapter 8 (Optimization)

**Practical Resources**:
- PyTorch Optimizers: https://pytorch.org/docs/stable/optim.html
- FastAI Learning Rate Finder: https://docs.fast.ai/callback.schedule.html
- Sebastian Ruder's Blog: http://ruder.io/
