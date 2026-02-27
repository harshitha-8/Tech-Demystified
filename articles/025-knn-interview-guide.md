# Crash Course to Crack Machine Learning Interview – Part 9: k-Nearest Neighbors

### Mastering distance metrics, the curse of dimensionality, and geometric intuition for ML interviews

k-Nearest Neighbors (k-NN) is deceptively simple. The entire algorithm can be explained in one sentence: **find the k closest training examples and use their labels to predict**. Yet this simplicity makes k-NN one of the most revealing interview topics in machine learning.

Interviewers don't ask about k-NN because it's the most widely deployed production algorithm (it's not). They ask about it because k-NN forces you to reason about **foundational concepts** without the complexity of optimization, gradients, or ensemble mechanics:

- **Distance and similarity**: What does "closeness" mean geometrically?
- **Feature scaling**: Why does it matter more for k-NN than almost any other algorithm?
- **Bias-variance tradeoff**: How does k control model flexibility?
- **Curse of dimensionality**: Why do high-dimensional spaces break nearest neighbor logic?
- **Computational efficiency**: What's the cost of non-parametric learning?

A strong k-NN answer demonstrates that you understand **the geometry of machine learning** — how algorithms operate in feature space, why distance metrics shape behavior, and how dimensionality affects learning. These insights transfer to other algorithms: kernel methods, clustering, embeddings, and more.

In this guide, we'll build k-NN from first principles. We'll explore how different distance metrics create different geometric neighborhoods, why the choice of k represents a fundamental bias-variance tradeoff, what happens when dimensionality increases, and how to handle k-NN's computational challenges. We'll also cover the most common interview questions and how to answer them with geometric intuition.

## The Core Algorithm: Lazy Learning from Neighborhoods

k-Nearest Neighbors belongs to a category called **non-parametric** or **instance-based learning**. Unlike parametric models (linear regression, neural networks) that learn fixed parameters during training, k-NN simply **memorizes the training data** and defers all computation until prediction time.

#### The k-NN Process

**Training Phase**:
```python
def fit(X_train, y_train):
    # Store the entire dataset
    self.X_train = X_train
    self.y_train = y_train
    # That's it!
```

**No learning happens**. No parameters estimated, no optimization performed, no model fitted. This is why k-NN is called a "lazy learner" — it procrastinates until absolutely necessary.

**Prediction Phase**:

For a new point x:

1. **Compute distances** to all training points:
   $$d_i = \text{distance}(x, x_i) \quad \forall i \in \{1, 2, ..., N\}$$

2. **Find k nearest neighbors**:
   - Sort distances: $d_{(1)} \leq d_{(2)} \leq ... \leq d_{(N)}$
   - Select k smallest: $\{x_{(1)}, x_{(2)}, ..., x_{(k)}\}$

3. **Aggregate neighbor labels**:
   - **Classification**: Majority vote
     $$\hat{y} = \text{mode}(y_{(1)}, y_{(2)}, ..., y_{(k)})$$
   
   - **Regression**: Average
     $$\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_{(i)}$$

#### Implementation Example

```python
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        # Compute distances (Euclidean)
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        
        # Find k nearest indices
        k_indices = np.argsort(distances)[:self.k]
        
        # Get k nearest labels
        k_nearest_labels = self.y_train[k_indices]
        
        # Return majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
```

#### Key Characteristics

**Non-parametric**:
- No fixed parameter count
- Model complexity grows with data size
- Can represent arbitrarily complex decision boundaries

**Instance-based**:
- Stores all training data
- Makes decisions by comparing to stored examples
- Memory grows linearly with dataset size

**Lazy learning**:
- No training time (O(1))
- All work happens at prediction (O(N × D) per query)
- Opposite of eager learners (neural nets, trees)

**Local method**:
- Predictions depend only on nearby training points
- Different regions of space use different neighbors
- Can adapt to local patterns without global assumptions

#### Interview Key Point

**Question**: "Explain how k-NN makes predictions differently from models like logistic regression."

**Strong answer**:
> "k-NN is a non-parametric, instance-based learner that memorizes training data rather than learning parameters. At prediction time, it finds the k closest training examples using a distance metric and aggregates their labels. Unlike logistic regression which learns a global linear decision boundary, k-NN is a local method that adapts to different patterns in different regions of feature space. This flexibility comes at a cost: k-NN requires storing all training data and computing distances to every point at prediction time, making it memory-intensive and slow for inference compared to parametric models."

## Distance Metrics: The Geometry of Similarity

**Distance metrics define what "nearest" means**. This choice fundamentally shapes k-NN behavior, yet it's often overlooked. Understanding distance metrics requires thinking geometrically about feature space.

#### Euclidean Distance: Circular Neighborhoods

**Definition**:
$$d_{\text{Euclidean}}(x, y) = \sqrt{\sum_{i=1}^{d} (x_i - y_i)^2}$$

**Geometric interpretation**: Straight-line distance, "as the crow flies"

**Neighborhood shape**: Circular (in 2D), spherical (in 3D+)

**Properties**:
- All points on circle/sphere are equally "near"
- Large change in one feature can be compensated by small changes in others
- Sensitive to feature scales

**Example**:
```
Point A: [1000 sqft, 2 bedrooms]
Point B: [1050 sqft, 2 bedrooms]  (50 sqft difference)
Point C: [1000 sqft, 3 bedrooms]  (1 bedroom difference)

Without scaling:
distance(A, B) = sqrt((50)^2 + (0)^2) = 50
distance(A, C) = sqrt((0)^2 + (1)^2) = 1

Point C appears much closer, even though 50 sqft may be less significant than 1 bedroom!
```

**When to use**: Continuous features on similar scales, smooth underlying geometry

#### Manhattan Distance: Diamond Neighborhoods

**Definition**:
$$d_{\text{Manhattan}}(x, y) = \sum_{i=1}^{d} |x_i - y_i|$$

**Geometric interpretation**: Distance along grid lines (city blocks)

**Neighborhood shape**: Diamond/square rotated 45° (in 2D), hypercube (in higher D)

**Properties**:
- Movement constrained to axis-aligned directions
- Large change in one feature can't be fully offset by changes in others
- More intuitive when features represent independent attributes

**When to use**: Grid-like problems, when features shouldn't compensate for each other

#### Minkowski Distance: Generalization

**Definition**:
$$d_{\text{Minkowski}}(x, y) = \left(\sum_{i=1}^{d} |x_i - y_i|^p\right)^{1/p}$$

**Special cases**:
- p = 1: Manhattan distance
- p = 2: Euclidean distance
- p → ∞: Chebyshev distance (max of absolute differences)

**Effect of p**: Controls how differences across dimensions combine

#### Cosine Similarity: Directional Neighborhoods

**Definition**:
$$\text{similarity}(x, y) = \frac{x \cdot y}{||x|| \cdot ||y||} = \frac{\sum_{i=1}^{d} x_i y_i}{\sqrt{\sum x_i^2} \cdot \sqrt{\sum y_i^2}}$$

**Distance**: $d_{\text{cosine}} = 1 - \text{similarity}$

**Geometric interpretation**: Measures angle between vectors, ignores magnitude

**Properties**:
- Vectors pointing in same direction are similar (even if different lengths)
- Scale-invariant
- Values in [-1, 1] (similarity) or [0, 2] (distance)

**When to use**: 
- Text data (document similarity based on word frequencies)
- When magnitude shouldn't matter, only relative proportions
- High-dimensional sparse data

**Example**:
```
Document A: [10, 20, 0]  (10x word1, 20x word2)
Document B: [100, 200, 0]  (100x word1, 200x word2)

Euclidean distance: Large (very different magnitudes)
Cosine similarity: 1.0 (identical direction → identical topics)
```

#### Hamming Distance: Categorical Neighborhoods

**Definition**:
$$d_{\text{Hamming}}(x, y) = \sum_{i=1}^{d} \mathbb{1}[x_i \neq y_i]$$

**Counts number of differing features**

**When to use**: Binary or categorical features where arithmetic operations don't make sense

**Example**:
```
Person A: [Male, USA, Engineer]
Person B: [Male, Canada, Engineer]
Person C: [Female, USA, Doctor]

Hamming(A, B) = 1  (only country differs)
Hamming(A, C) = 2  (gender and occupation differ)
```

#### Choosing the Right Distance Metric

| Data Type | Recommended Metric | Reasoning |
|-----------|-------------------|-----------|
| Continuous, similar scales | Euclidean | Standard choice, smooth neighborhoods |
| Continuous, independent attributes | Manhattan | Features don't compensate each other |
| Text (TF-IDF, word counts) | Cosine | Direction matters, magnitude less so |
| Binary/categorical | Hamming | Counts differences, no arithmetic |
| Mixed types | Custom or Gower | Combines appropriate metrics per feature type |

#### Implementing Different Metrics

```python
from sklearn.neighbors import KNeighborsClassifier

# Euclidean (default)
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Manhattan
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

# Minkowski (p=3)
knn_minkowski = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)

# Cosine (requires preprocessing)
from sklearn.neighbors import KNeighborsClassifier
knn_cosine = KNeighborsClassifier(n_neighbors=5, metric='cosine')

# Custom distance function
def custom_distance(x, y):
    # Weight first feature 2x more
    return np.sqrt(2 * (x[0] - y[0])**2 + np.sum((x[1:] - y[1:])**2))

knn_custom = KNeighborsClassifier(n_neighbors=5, metric=custom_distance)
```

#### Interview Key Point

**Question**: "Why does the choice of distance metric matter in k-NN?"

**Strong answer**:
> "Distance metric defines what 'similar' means geometrically. k-NN blindly trusts this definition — if the metric says two points are close, they'll influence each other's predictions regardless of whether they should. Euclidean distance assumes all features are continuous and commensurable, treating feature space as smooth and isotropic. Manhattan works better when features represent independent quantities. Cosine ignores magnitude and only compares direction, essential for text data. Using the wrong metric can make genuinely similar points appear distant or vice versa. Since k-NN doesn't learn to weight features, choosing the right distance metric is critical for capturing true similarity."

## Feature Scaling: Non-Negotiable for k-NN

Unlike decision trees and many other algorithms, **k-NN is extremely sensitive to feature scales**. This isn't a minor implementation detail — it's a fundamental consequence of how distance-based algorithms work.

#### The Problem

Consider predicting house prices with two features:

```
Feature 1: square_footage [500 - 5000] (range: 4500)
Feature 2: num_bedrooms [1 - 6] (range: 5)
```

**Without scaling**, Euclidean distance for two houses:

```python
House A: [2000 sqft, 3 bedrooms]
House B: [2100 sqft, 3 bedrooms]
House C: [2000 sqft, 4 bedrooms]

distance(A, B) = sqrt((100)^2 + (0)^2) = 100
distance(A, C) = sqrt((0)^2 + (1)^2) = 1
```

House C appears 100x closer than House B, purely because of scale differences!

**The number of bedrooms (1 unit difference) dominates square footage (100 unit difference)**, even though 100 sqft might be more meaningful for price prediction.

**k-NN has no way to learn that square footage matters more** — it mechanically uses the geometry defined by your features and distance metric.

#### Why k-NN is Scale-Sensitive

**Distance-based algorithms** (k-NN, k-means, SVM with RBF kernel) compute distances in feature space. Larger-scale features contribute more to distance:

$$d = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}$$

If $x_1$ ranges 0-1000 and $x_2$ ranges 0-1, the first term dominates.

**Contrast with scale-invariant algorithms**:
- **Decision trees**: Only care about relative ordering (thresholds)
- **Linear models**: Learn weights that compensate for scale
- **k-NN**: Uses raw feature values directly in distance computation

#### Scaling Methods

**Standardization (Z-score normalization)**:
$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

**Result**: Mean = 0, standard deviation = 1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CRITICAL: fit on train, transform both train and test
# Never fit on test data (data leakage!)
```

**When to use**: Features have different units, want to preserve outlier information

**Min-Max Scaling**:
$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**Result**: Range [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**When to use**: Want bounded range, data has hard limits, no extreme outliers

**Robust Scaling**:
$$x_{\text{scaled}} = \frac{x - \text{median}}{\text{IQR}}$$

**When to use**: Data has outliers that would distort standard/min-max scaling

#### Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Pipeline ensures scaling is always applied consistently
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# Fit scales and trains in one step
pipeline.fit(X_train, y_train)

# Predict automatically scales new data
predictions = pipeline.predict(X_test)
```

**Critical**: Pipelines prevent data leakage by ensuring scaler is fit only on training data.

#### Interview Key Point

**Question**: "Why is feature scaling important for k-NN but not for decision trees?"

**Strong answer**:
> "k-NN computes distances directly using raw feature values. Features with larger scales contribute proportionally more to distance, effectively getting higher weight. A feature ranging 0-1000 will dominate one ranging 0-1, even if the smaller-scale feature is more predictive. Since k-NN doesn't learn feature weights, we must manually ensure features are on comparable scales through standardization or normalization. Decision trees, in contrast, are scale-invariant because they only compare values within a single feature at a time when choosing splits. Whether income is in dollars or thousands doesn't change which side of a threshold a sample falls on."

## Choosing k: The Bias-Variance Tradeoff Visualized

The value of k is the **single most important hyperparameter** in k-NN. It directly controls the bias-variance tradeoff in an intuitive geometric way.

#### Small k (k=1): High Variance, Low Bias

**k=1** (nearest neighbor only):
- Prediction = label of single closest point
- **Bias**: Very low — can perfectly fit any training pattern
- **Variance**: Very high — prediction highly sensitive to individual training points

**Decision boundary**: Extremely irregular, wraps around every training point

```python
# k=1: Perfect training accuracy, poor generalization
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, y_train)
print(f"Train accuracy: {knn1.score(X_train, y_train)}")  # Often 1.0!
print(f"Test accuracy: {knn1.score(X_test, y_test)}")    # Much lower
```

**Problem**: Memorizes noise, overfits badly

**Visualization**: Decision boundary has jagged, complex regions that conform to every training point, including outliers.

#### Large k (k→N): High Bias, Low Variance

**k=N** (all training points):
- Prediction = majority class of entire dataset
- **Bias**: Very high — can't capture any local patterns
- **Variance**: Very low — prediction is constant

**Decision boundary**: Constant prediction (single class)

**Problem**: Underfits, ignores local structure

#### Optimal k: Balance

**Typical values**: 3, 5, 7, 10, 15 (often odd numbers for binary classification to avoid ties)

**As k increases**:
- Decision boundaries become smoother
- Predictions less sensitive to individual outliers
- More "democratic" averaging over larger neighborhoods
- But: loses ability to capture fine-grained patterns

#### Mathematical Intuition

Prediction variance for k-NN regression:

$$\text{Var}(\hat{y}) \approx \frac{\sigma^2}{k}$$

Where σ² is noise variance in labels.

**Doubling k approximately halves variance** (assuming independent neighbors).

#### Choosing k in Practice

**Cross-validation approach**:

```python
from sklearn.model_selection import cross_val_score

k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())
    print(f"k={k}: CV accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")

# Plot
import matplotlib.pyplot as plt
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('k (number of neighbors)')
plt.ylabel('Cross-validation accuracy')
plt.title('k-NN: Choosing k')
plt.axvline(k_values[np.argmax(cv_scores)], color='r', linestyle='--', 
            label=f'Best k={k_values[np.argmax(cv_scores)]}')
plt.legend()
plt.show()
```

**Typical pattern**:
- Very small k: Overfits (high CV variance across folds)
- Optimal k: Best test performance
- Very large k: Underfits (high bias)

**Rule of thumb**: $k \approx \sqrt{N}$ as starting point, then cross-validate

#### Odd vs Even k

For **binary classification**, prefer **odd k** to avoid ties:
```
k=4, neighbors: [Class A, Class A, Class B, Class B]
→ Tie! Need tiebreaker rule

k=5, neighbors: [Class A, Class A, Class A, Class B, Class B]
→ Class A wins (no tie)
```

For multi-class or regression, this matters less.

#### Interview Key Point

**Question**: "How does the value of k affect the bias-variance tradeoff in k-NN?"

**Strong answer**:
> "k directly controls neighborhood size and thus model flexibility. With k=1, the model has very low bias — it can fit any training pattern perfectly by memorizing individual points. But it has high variance because predictions depend on single, potentially noisy neighbors. As k increases, we average over more neighbors, which smooths predictions and reduces variance. However, large k introduces bias by forcing the model to average over points that may have different underlying patterns. Optimal k balances these: large enough to average out noise but small enough to preserve local structure. I'd choose k via cross-validation, typically trying values from 3 to sqrt(N)."

## The Curse of Dimensionality: When Distance Breaks Down

The **curse of dimensionality** is one of the most important concepts in machine learning, and k-NN provides the clearest demonstration of why high-dimensional spaces are fundamentally different from low-dimensional intuition.

#### The Problem: Distance Becomes Meaningless

In high dimensions, **all points become approximately equidistant**.

**Intuition**: Consider a unit hypercube [0,1]^d with random points.

As d increases:
- Volume concentrates in corners (far from center)
- Average distance between random points grows as $\sqrt{d}$
- Relative difference between nearest and farthest neighbor shrinks

**Mathematical result**:

$$\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0 \text{ as } d \to \infty$$

**In English**: The ratio of farthest to nearest neighbor approaches 1. **All neighbors are approximately the same distance away.**

#### Empirical Demonstration

```python
import numpy as np

def distance_concentration(n_samples=1000, dimensions=[2, 10, 50, 100, 500]):
    results = []
    
    for d in dimensions:
        # Generate random points
        X = np.random.randn(n_samples, d)
        
        # Pick a query point
        query = X[0]
        
        # Compute distances to all other points
        distances = np.sqrt(np.sum((X[1:] - query)**2, axis=1))
        
        d_min = distances.min()
        d_max = distances.max()
        d_mean = distances.mean()
        
        # Relative difference
        contrast = (d_max - d_min) / d_min
        
        results.append({
            'dimensions': d,
            'min_dist': d_min,
            'max_dist': d_max,
            'mean_dist': d_mean,
            'contrast': contrast
        })
        
        print(f"d={d:4d}: min={d_min:.2f}, max={d_max:.2f}, "
              f"mean={d_mean:.2f}, contrast={contrast:.3f}")
    
    return results

results = distance_concentration()

# Output example:
# d=   2: min=0.15, max=4.82, mean=1.59, contrast=31.133
# d=  10: min=1.89, max=6.73, mean=4.02, contrast=2.562
# d=  50: min=6.51, max=11.92, mean=9.20, contrast=0.831
# d= 100: min=9.85, max=15.21, mean=12.53, contrast=0.544
# d= 500: min=22.15, max=27.93, mean=25.04, contrast=0.261
```

**Observation**: As dimensionality increases, contrast ratio collapses. In 500 dimensions, max distance is only 26% farther than min distance!

**Implication for k-NN**: **"Nearest" neighbor isn't actually near**, and isn't much closer than the "farthest" neighbor.

#### Why This Breaks k-NN

1. **Loss of locality**: Neighborhoods no longer contain truly similar points

2. **Increased noise sensitivity**: With so many features, irrelevant ones add noise to distance

3. **Sample sparsity**: Volume of d-dimensional space grows as $V \propto r^d$. To maintain same density, need samples exponentially: $N \propto (\frac{1}{\epsilon})^d$

**Example**: To have same neighbor density:
- 2D: Need 100 samples
- 10D: Need 100^5 = 10 billion samples!

#### Mitigating Dimensionality

**1. Feature Selection**
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Keep only k most informative features
selector = SelectKBest(f_classif, k=10)
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)
```

**2. Dimensionality Reduction**
```python
from sklearn.decomposition import PCA

# Reduce to lower-dimensional space
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Use k-NN in reduced space
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
```

**3. Use Different Algorithm**

When d > 50-100, consider:
- Linear models (scale well with dimensionality)
- Tree-based methods (naturally select important features)
- Neural networks with embeddings

#### Interview Key Point

**Question**: "What is the curse of dimensionality and how does it affect k-NN?"

**Strong answer**:
> "In high-dimensional spaces, distance loses its discriminative power — all points become approximately equidistant. This happens because small differences across many dimensions accumulate, and the volume of space grows exponentially. For k-NN, this means 'nearest' neighbors aren't actually similar; they're just the least dissimilar among many distant points. The algorithm breaks down because its core assumption — that nearby points should have similar labels — no longer holds. Practically, k-NN works well up to maybe 20-50 dimensions with sufficient data, but beyond that, you need dimensionality reduction or should switch to algorithms that scale better with dimensionality."

## Weighted k-NN: Distance-Based Voting

Standard k-NN treats all k neighbors equally. **Weighted k-NN** gives closer neighbors more influence, improving predictions.

#### The Motivation

**Problem with uniform voting**:

```
Query point: x
3-NN with distances:
- Neighbor 1: distance=0.1, class=A
- Neighbor 2: distance=0.12, class=A  
- Neighbor 3: distance=2.5, class=B

Uniform k-NN: 2 votes for A, 1 for B → predict A
```

Neighbor 3 is **25x farther** than neighbor 1, yet gets equal vote. Should it?

#### Weighted Voting

**Inverse distance weighting**:

$$w_i = \frac{1}{d_i + \epsilon}$$

Where ε is small constant to avoid division by zero when d=0.

**Prediction**:
- **Classification**: Weighted vote
  $$\hat{y} = \arg\max_c \sum_{i=1}^{k} w_i \cdot \mathbb{1}[y_i = c]$$

- **Regression**: Weighted average
  $$\hat{y} = \frac{\sum_{i=1}^{k} w_i \cdot y_i}{\sum_{i=1}^{k} w_i}$$

**Example**:
```
Weights:
w1 = 1/0.1 = 10.0
w2 = 1/0.12 = 8.33
w3 = 1/2.5 = 0.4

Weighted votes:
Class A: 10.0 + 8.33 = 18.33
Class B: 0.4

→ Predict A (with much higher confidence)
```

#### Implementation

```python
from sklearn.neighbors import KNeighborsClassifier

# Uniform weighting (default)
knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# Distance weighting
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Custom weighting function
def gaussian_weight(distances):
    return np.exp(-distances**2 / (2 * 1.0**2))

knn_custom = KNeighborsClassifier(n_neighbors=5, weights=gaussian_weight)
```

#### Benefits

1. **Reduces sensitivity to k**: Distant neighbors get low weight, so exact k value matters less

2. **Better decision boundaries**: Smoother transitions between classes

3. **Handles noisy neighbors**: Outliers at edge of neighborhood don't dominate

4. **More robust**: Less affected by border cases

#### Trade-offs

- **More complex**: Additional hyperparameter (weighting function)
- **Less interpretable**: Weights aren't as intuitive as "5 neighbors"
- **Computation**: Slightly more expensive (weight calculation)

Most practitioners use **distance weighting by default** since it improves performance with minimal downside.

## Computational Complexity: The Cost of Laziness

k-NN's simplicity during training comes at a cost: **expensive prediction time**.

#### Complexity Analysis

**Training**:
- Time: O(1) — just store data
- Space: O(N × D) — must keep all training samples

**Prediction (brute force)**:
- **Per query**: O(N × D + N log k)
  - O(N × D): Compute distance to all N points, each with D features
  - O(N log k): Find k smallest distances (partial sort)
- **For M queries**: O(M × N × D)

**Comparison with other algorithms**:

| Algorithm | Training Time | Prediction Time (per sample) |
|-----------|---------------|----------------------------|
| k-NN (brute force) | O(1) | O(N × D) |
| k-NN (KD-tree) | O(N log N) | O(log N) to O(N) |
| Decision Tree | O(N log N × D) | O(depth) ≈ O(log N) |
| Linear Model | O(N × D) | O(D) |
| Neural Network | O(iterations × N × D) | O(D) |

**Problem**: k-NN prediction time **grows with dataset size**. With 1 million training samples, every prediction requires 1 million distance computations!

#### Optimization 1: KD-Trees

**Idea**: Organize data into tree structure for faster search

**Construction**:
- Recursively split data along median of alternating features
- Creates axis-aligned partitions
- Build time: O(N log N)

**Query**:
- Traverse tree to find leaf containing query
- Search nearby regions using branch-and-bound
- Best case: O(log N), worst case: O(N)

```python
from sklearn.neighbors import KNeighborsClassifier

# Automatically uses KD-tree when beneficial
knn_kdtree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='kd_tree',
    leaf_size=30
)
```

**Limitation**: **Breaks down in high dimensions** (d > 20). Distance concentration makes pruning ineffective.

#### Optimization 2: Ball Trees

**Idea**: Organize data into nested hyperspheres

**Construction**:
- Recursively partition into balls that minimize radius
- More flexible than axis-aligned splits
- Build time: O(N log N)

**Query**:
- Traverse tree using triangle inequality to prune branches
- Better than KD-tree for high dimensions
- Still degrades as d increases

```python
knn_balltree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='ball_tree',
    leaf_size=30
)
```

**Works better than KD-tree up to d ≈ 50**, but still struggles beyond that.

#### Optimization 3: Approximate Nearest Neighbors (ANN)

**Idea**: Sacrifice exactness for massive speedup

**Methods**:
- **Locality-Sensitive Hashing (LSH)**: Hash similar points to same buckets
- **HNSW (Hierarchical Navigable Small Worlds)**: Graph-based navigation
- **FAISS (Facebook AI Similarity Search)**: GPU-accelerated approximate search

```python
import faiss

# Build index
d = X_train.shape[1]
index = faiss.IndexFlatL2(d)  # Simple version
index.add(X_train.astype('float32'))

# Search
k = 5
distances, indices = index.search(X_query.astype('float32'), k)

# Returns approximate k nearest neighbors (very fast)
```

**Trade-off**: 90-99% recall (find most of true nearest neighbors) with 10-100x speedup

**When to use**: Large-scale systems (millions+ samples), high dimensions, real-time inference

#### Practical Implications

**k-NN is rarely used alone in production** for large datasets because:
1. Inference latency grows with data size
2. Memory requirements (store all training data)
3. No compression or abstraction

**Where k-NN appears in practice**:
- Small to medium datasets (< 100K samples)
- Offline batch predictions (not real-time)
- As similarity component in larger systems
- Research/prototyping baselines

**Modern alternative**: Learn embeddings (neural network) that map data to low-dimensional space where Euclidean distance is meaningful, then use approximate nearest neighbors in that space.

#### Interview Key Point

**Question**: "What are the computational limitations of k-NN?"

**Strong answer**:
> "k-NN's simplicity during training creates computational challenges at prediction time. Each query requires computing distances to all N training points, making prediction O(N × D) per sample. As datasets grow, this becomes prohibitively slow. KD-trees and ball trees can accelerate search to O(log N) in low dimensions, but they degrade to O(N) in high dimensions due to distance concentration. For production systems with millions of samples or high-dimensional data, approximate nearest neighbor methods like LSH or FAISS provide 10-100x speedups by trading exact neighbors for approximate ones. k-NN also requires storing all training data in memory, which doesn't scale. This is why k-NN is rarely used alone for large-scale production inference."

## Handling Imbalanced Classes

k-NN struggles with **class imbalance** because majority class naturally dominates neighborhoods.

#### The Problem

```
Dataset: 95% Class A (majority), 5% Class B (minority)

Query point: Actually Class B
k=10 nearest neighbors found:
- 9 are Class A (just due to prevalence)
- 1 is Class B

Prediction: Class A (wrong!)
```

**Why**: In most regions of space, majority class points are denser. Even when query is truly minority class, most neighbors will be majority class due to pure statistics.

#### Solutions

**1. Use Smaller k**
```python
# Smaller neighborhood preserves local minority regions
knn = KNeighborsClassifier(n_neighbors=3)  # Instead of k=10
```

**Downside**: Higher variance

**2. Distance Weighting**
```python
# Closer neighbors (potentially true class) get more weight
knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
```

If true minority class neighbor is very close, its weighted vote can outweigh distant majority neighbors.

**3. Class-Balanced Sampling**
```python
from imblearn.over_sampling import SMOTE

# Create synthetic minority class samples
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train k-NN on balanced data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_resampled, y_resampled)
```

**SMOTE**: Creates synthetic minority samples by interpolating between existing ones.

**4. Adjust Decision Threshold**
```python
# Get probability predictions
probas = knn.predict_proba(X_test)

# Use lower threshold for minority class
threshold = 0.3  # Instead of 0.5
predictions = (probas[:, 1] >= threshold).astype(int)
```

**5. Use Different Metric**

Don't optimize accuracy — use metrics that respect imbalance:
- **F1-score**: Balances precision and recall
- **Balanced accuracy**: Average of per-class accuracies
- **ROC-AUC**: Threshold-independent performance

```python
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]

print(f"F1: {f1_score(y_test, y_pred):.3f}")
print(f"Balanced Acc: {balanced_accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
```

## When to Use k-NN vs. Other Algorithms

Understanding when k-NN is appropriate demonstrates practical judgment.

#### When k-NN Excels

**1. Small to Medium Datasets** (100 - 100K samples)
- Computation and memory are manageable
- Can store all training data

**2. Low to Moderate Dimensionality** (< 20-30 features)
- Distance remains meaningful
- Curse of dimensionality doesn't dominate

**3. Complex, Non-Linear Decision Boundaries**
- No assumptions about global structure
- Adapts to local patterns naturally

**4. Continuous Learning / Online Updates**
- Just add new points to training set (no retraining!)
- Remove old points easily

**5. Need Simple Baseline**
- No hyperparameters to tune extensively
- Easy to understand and explain
- Quick to implement

**6. Similarity is Well-Defined**
- Features naturally capture meaningful similarity
- Domain knowledge suggests distance-based reasoning

#### When to Avoid k-NN

**1. Large Datasets** (> 100K samples)
- Prediction becomes prohibitively slow
- Memory requirements too high
- Consider approximate methods or different algorithm

**2. High-Dimensional Data** (> 50 features)
- Distance becomes meaningless
- Use dimensionality reduction or alternative

**3. Real-Time Inference Requirements**
- Latency grows with dataset size
- Need constant-time predictions
- Use parametric model (linear, neural net)

**4. Many Irrelevant Features**
- Noise dimensions pollute distance
- Feature selection critical (but expensive)
- Trees naturally handle this better

**5. Categorical Features Dominate**
- Distance metrics less natural
- Consider trees or encoding strategies

**6. Severely Imbalanced Classes**
- Majority class dominates neighborhoods
- Need significant preprocessing
- Consider cost-sensitive models

#### Comparison: k-NN vs. Other Algorithms

| Aspect | k-NN | Decision Tree | Linear Model | Neural Network |
|--------|------|---------------|--------------|----------------|
| Training Time | O(1) | O(N log N × D) | O(N × D) | O(epochs × N × D) |
| Prediction Time | O(N × D) | O(depth) | O(D) | O(D) |
| Memory | O(N × D) | O(nodes) | O(D) | O(weights) |
| Non-linear | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| Feature Scaling | ⚠️ Required | ✅ Not needed | ⚠️ Helps | ⚠️ Required |
| Interpretability | Medium | High | High | Low |
| High Dimensions | ❌ Poor | ✅ Good | ✅ Good | ⚠️ Needs architecture |
| Handles Irrelevant Features | ❌ Poor | ✅ Good | ⚠️ Regularization helps | ⚠️ With dropout |
| Online Learning | ✅ Trivial | ❌ Hard | ⚠️ Possible | ⚠️ Possible |

## Common Interview Questions and Answers

#### 1. Explain how k-NN makes predictions. How does it differ for classification vs. regression?

k-NN makes predictions by finding the k closest training examples to a query point using a distance metric, then aggregating their labels. For **classification**, it uses majority voting — the most common class among the k neighbors becomes the prediction. For **regression**, it averages the target values of the k neighbors. The core algorithm is identical; only the aggregation step differs. Both rely on the assumption that nearby points in feature space should have similar outputs.

#### 2. Why is feature scaling critical for k-NN?

k-NN computes distances using raw feature values. Features with larger numeric ranges contribute proportionally more to distance calculations. For example, if income ranges 20K-200K and age ranges 20-80, a 10K income difference creates distance 10,000 while a 30-year age difference creates distance 30. Income completely dominates, even if age is more predictive. Unlike algorithms that learn feature weights during training, k-NN blindly uses the geometry you provide. Scaling ensures all features contribute proportionally based on their variance, not their arbitrary units. This makes scaling non-optional for k-NN.

#### 3. How does the choice of k affect the bias-variance tradeoff?

k directly controls model complexity. **Small k** (like k=1) creates low bias because the model can fit any pattern by memorizing individual training points. But it has high variance — predictions are unstable and sensitive to noise. **Large k** reduces variance by averaging over many neighbors, smoothing predictions. But it increases bias by forcing the model to average over regions that may have different true patterns. Optimal k balances these: large enough to reduce noise but small enough to preserve local structure. In practice, k is chosen via cross-validation, often in the range 3-15.

#### 4. What is the curse of dimensionality and why does it affect k-NN?

In high-dimensional spaces, distance loses meaning because all points become approximately equidistant. This happens because: (1) Small differences across many dimensions accumulate, making all pairwise distances large; (2) The ratio of farthest to nearest neighbor approaches 1; (3) Volume grows exponentially, making data sparse. For k-NN, this breaks the core assumption that "nearest" neighbors are actually similar. In 100+ dimensions, your "nearest" neighbor might be very far away and not meaningfully similar. Practically, k-NN works up to ~20-50 dimensions, beyond which you need dimensionality reduction or a different algorithm.

#### 5. What are the advantages and disadvantages of k-NN?

**Advantages**: (1) Simple to understand and implement; (2) No training phase — add/remove data instantly; (3) No assumptions about data distribution; (4) Naturally handles multi-class problems; (5) Can capture complex non-linear patterns; (6) Effective baseline for small datasets. **Disadvantages**: (1) Slow prediction time (O(N) per query); (2) Memory intensive (stores all training data); (3) Requires feature scaling; (4) Sensitive to irrelevant features; (5) Breaks down in high dimensions; (6) Struggles with imbalanced classes; (7) No learned representation or feature importance.

#### 6. When would you use Manhattan distance instead of Euclidean?

Use Manhattan distance when: (1) **Features represent independent quantities** that shouldn't compensate for each other (e.g., counting different types of events); (2) **Want robustness to outliers** — Manhattan is less sensitive to large differences in single features because it doesn't square them; (3) **Grid-like problem structure** where movement is constrained to axis-aligned directions; (4) **Computational efficiency** — no square root calculation. Euclidean assumes smooth continuous space where diagonal movement is meaningful. Manhattan treats each dimension more independently. For many real-world problems, the difference is small, but for interpretability or when features have clear physical meaning, Manhattan can be more appropriate.

#### 7. How do you choose the optimal value of k?

Use **cross-validation** to systematically test different k values. Try a range (e.g., 1, 3, 5, 7, 11, 15, 21, 31, 51) and plot CV accuracy vs. k. The curve typically shows: (1) Small k: Lower CV score (high variance); (2) Optimal k: Peak performance; (3) Large k: Decreasing score (high bias). Choose k at the peak. For starting points, try k = sqrt(N) or the square root of the number of samples. For binary classification, prefer odd k to avoid ties. Also consider: dataset size (larger data can support larger k), noise level (noisy data benefits from larger k), and computational budget (larger k is slightly slower).

#### 8. Why is k-NN called a "lazy learner"?

k-NN is called "lazy" because it defers all computation until prediction time. During training, it does literally nothing except store the data — no parameters learned, no patterns identified, no model fitted. When a prediction is needed, k-NN must compute distances to all training points and find nearest neighbors. This is opposite to "eager learners" (neural networks, decision trees) that invest heavily in training to learn compact representations, making inference fast. k-NN's laziness makes training instant but predictions expensive, especially as data grows.

#### 9. What happens when k equals 1? What about when k equals N?

**k=1**: Predict using only the single nearest neighbor. This achieves **perfect training accuracy** (each point is its own nearest neighbor) but severely overfits. Decision boundaries are extremely irregular, wrapping around every training point including noise and outliers. High variance, low bias. **k=N**: Predict using all training points. For classification, this predicts the majority class for every query (constant prediction). For regression, it predicts the mean of all training targets. Completely ignores local structure. High bias, low variance. Both extremes are useless; optimal k is in between.

#### 10. How does weighted k-NN differ from standard k-NN?

Standard k-NN gives all k neighbors equal votes. Weighted k-NN assigns higher influence to closer neighbors, typically using **inverse distance weighting**: weight = 1/distance. For classification, this means closer neighbors' votes count more; for regression, closer neighbors contribute more to the average. **Benefits**: (1) Less sensitive to exact k value; (2) Smoother decision boundaries; (3) Reduces impact of distant neighbors that happen to fall within k; (4) Often improves performance. The algorithm still selects the same k neighbors; only the aggregation changes. Most practitioners use distance weighting by default since it typically helps without significant downside.

#### 11. Can k-NN be used for regression? How?

Yes! k-NN regression works identically to classification except the aggregation step. Instead of majority voting, it **averages the target values** of k nearest neighbors: ŷ = (1/k) × Σ yᵢ. With distance weighting: ŷ = Σ(wᵢ × yᵢ) / Σwᵢ. k-NN regression naturally handles non-linear relationships and makes no distributional assumptions. However, it **cannot extrapolate** — predictions are bounded by the min/max of training targets. If test data extends beyond training range, k-NN will underpredict or overpredict. For this reason, k-NN regression is less common than classification in practice.

#### 12. How do KD-trees improve k-NN efficiency?

KD-trees organize training data into a binary tree by recursively splitting along feature medians in alternating dimensions. This creates axis-aligned partitions. During query, the algorithm traverses the tree toward the region containing the query point, finding candidate neighbors quickly. Branches are pruned using distance bounds — if the closest point in a branch is farther than the current k-th nearest neighbor, that entire branch is skipped. This reduces prediction from O(N) to O(log N) in best case. **Limitation**: Effective only in low dimensions (< 20). In high dimensions, distance concentration prevents effective pruning, degrading to O(N) anyway.

#### 13. What preprocessing steps are essential for k-NN?

**(1) Feature scaling** (standardization or min-max) — non-negotiable since k-NN is distance-based. **(2) Handle missing values** — k-NN can't process NaNs; impute using median/mean or k-NN imputer. **(3) Encode categorical features** — use one-hot encoding or ordinal encoding depending on whether categories have natural order. **(4) Feature selection** — remove irrelevant/redundant features that add noise to distance. **(5) Consider dimensionality reduction** (PCA) if features > 20-30. **(6) Remove or cap outliers** — extreme values distort distance calculations. The preprocessing pipeline is more critical for k-NN than most algorithms because it doesn't learn to adapt to data issues.

#### 14. How does k-NN handle multi-class classification?

k-NN naturally handles multi-class problems without modification. For k neighbors, it counts votes for each class and predicts the class with the most votes. With distance weighting, each neighbor contributes its weighted vote to its class. Unlike one-vs-rest or other multi-class strategies needed for binary classifiers, k-NN's voting mechanism inherently supports any number of classes. This makes it very convenient for multi-class problems. However, in cases with many classes and class imbalance, some classes may never appear in top-k neighbors, effectively being impossible to predict.

#### 15. What is the role of the distance metric in k-NN?

The distance metric **defines the geometry of feature space** and thus completely determines which points are considered "near." k-NN doesn't learn which features matter — it trusts the distance metric's judgment. **Euclidean** creates circular neighborhoods, good for continuous features; **Manhattan** creates diamond-shaped neighborhoods, more robust to outliers; **Cosine** measures angle not distance, essential for high-dimensional sparse data like text; **Hamming** counts differences, appropriate for categorical data. Wrong metric choice can make truly similar points appear distant or vice versa. Since k-NN has no learning phase to correct bad geometry, choosing the right metric is critical.

#### 16. Why doesn't k-NN require a training phase?

k-NN is a **non-parametric, instance-based learner**. It doesn't learn a function or parameters that generalize patterns — it simply memorizes all training examples. The "training" phase is just storing the data. Predictions are made by directly comparing new points to stored examples. This is fundamentally different from parametric models that compress training data into learned parameters (weights, trees, etc.). The trade-off: instant "training" but expensive prediction, and memory grows with dataset size. It's why k-NN is called a "lazy" learner — it defers all work until prediction time.

#### 17. How would you speed up k-NN for a large dataset?

Several strategies: **(1) Use tree-based exact search** (KD-tree, ball tree) to reduce prediction from O(N) to O(log N) in low dimensions. **(2) Approximate nearest neighbors** (FAISS, HNSW, LSH) for 10-100x speedup with 90-99% accuracy. **(3) Dimensionality reduction** (PCA) to reduce d, making distance computation faster. **(4) Feature selection** to remove irrelevant features. **(5) Prototype selection** — store only representative points, not all training data. **(6) GPU acceleration** for parallel distance computation. **(7) Hybrid approach**: Learn embeddings via neural network, then use approximate k-NN in embedding space. For production systems, I'd likely use learned embeddings + FAISS rather than raw k-NN.

#### 18. What are the decision boundaries produced by k-NN?

k-NN creates **Voronoi tessellation-like decision boundaries**. For k=1, boundaries are exactly the Voronoi diagram — regions where a single training point is closest. For k>1, boundaries smooth out, forming piecewise linear segments. Unlike linear models (straight lines) or trees (axis-aligned rectangles), k-NN boundaries can be arbitrarily complex and curved, adapting to local data density. As k increases, boundaries become smoother and more "averaged." In 2D, you can visualize this as a patchwork where each region is dominated by the local majority class. The boundaries are non-parametric, meaning complexity isn't predetermined — it's entirely determined by training data distribution.

#### 19. How does k-NN perform with irrelevant features?

k-NN performs **poorly with irrelevant features** because it treats all features equally in distance computation. Each irrelevant feature adds random noise to distances, obscuring the signal from relevant features. With enough noise features, the "nearest" neighbors are determined more by random variation in irrelevant dimensions than by true similarity in relevant ones. Unlike decision trees that can ignore bad features or linear models with regularization that zero out useless coefficients, k-NN has no mechanism to learn feature relevance. **Solution**: Aggressive feature selection or dimensionality reduction before applying k-NN, or use algorithms that naturally handle irrelevant features.

#### 20. Explain the relationship between k-NN and the Bayes optimal classifier.

In the limit of infinite data, **k-NN with k→∞ and k/N→0** (k grows but remains small fraction of N) approaches the **Bayes optimal classifier** — the theoretically best possible classifier. Intuition: With infinite data, any arbitrarily small neighborhood contains infinitely many samples, perfectly representing the true conditional distribution P(y|x) at that location. k-NN essentially estimates this local distribution empirically. However, with finite data, k-NN is suboptimal: small k has high variance, large k introduces bias. This theoretical connection shows why k-NN can be surprisingly effective — it's approximating the Bayes optimal solution by estimating local class distributions. It also reveals the fundamental limitation: finite sample sizes force a bias-variance tradeoff that the theoretical optimum doesn't face.

## Summary: Mastering k-NN

k-Nearest Neighbors represents one of the simplest yet most instructive algorithms in machine learning. Its elegance lies in a single principle: **similar inputs should produce similar outputs**. By avoiding assumptions about global structure and reasoning directly from local neighborhoods, k-NN can capture complex patterns with minimal code.

**Core Concepts to Remember**:

1. **Instance-Based Learning**: k-NN memorizes training data rather than learning parameters — no training phase, all work at prediction time

2. **Distance Metric Defines Geometry**: Choice of distance (Euclidean, Manhattan, cosine) fundamentally shapes what "similar" means; wrong choice breaks the algorithm

3. **Feature Scaling is Mandatory**: Large-scale features dominate distance; standardization ensures all features contribute proportionally

4. **k Controls Bias-Variance**: Small k = high variance (sensitive to noise), large k = high bias (over-smooths); optimal k balances via cross-validation

5. **Curse of Dimensionality**: High dimensions make distance meaningless — all points become equidistant; k-NN effective only up to ~20-50 dimensions

6. **Computational Cost**: O(N × D) prediction time per query; doesn't scale to large datasets without approximation methods (KD-trees, FAISS)

7. **Weighted Voting**: Distance-weighted neighbors improve performance by giving closer points more influence

8. **No Extrapolation**: Predictions bounded by training data min/max; problematic for time series or when test extends beyond training range

**For Interviews**:

Focus on **geometric intuition**. Explain that k-NN operates in feature space where distance encodes similarity. Discuss how different metrics create different neighborhood shapes (circles, diamonds, angles). Show understanding that k-NN doesn't learn — it delegates intelligence to your choice of distance and scaling. Explain the curse of dimensionality as distance concentration, not just "high dimensions are bad." Acknowledge computational limitations and modern solutions (approximate methods, embeddings). Connect k to bias-variance explicitly.

**Interview Red Flags to Avoid**:
- Ignoring feature scaling importance
- Not explaining why distance metric matters
- Missing curse of dimensionality discussion
- Suggesting k-NN for large-scale or high-dimensional problems
- Forgetting computational complexity

**In Practice**:

k-NN rarely appears alone in production systems. Its value is as: (1) **Baseline**: Quick sanity check for whether similarity-based reasoning works; (2) **Small data**: When you have < 10K samples and need something fast; (3) **Component**: Modern systems learn embeddings (neural networks) then use approximate k-NN for retrieval (recommendation systems, semantic search); (4) **Research**: Prototyping and experimentation. For production ML, most teams use tree-based methods (Random Forests, XGBoost) or neural networks, keeping k-NN as the conceptual foundation for understanding distance, similarity, and geometric learning.

k-NN proves that **simplicity doesn't mean superficiality**. By forcing you to think about distance, scale, dimensionality, and geometry explicitly, it reveals fundamental principles that more complex algorithms handle implicitly. Master k-NN, and you master the geometric foundations of machine learning.

---

*This article is part of the "Crash Course to Crack Machine Learning Interviews" series. For more articles on ML algorithms and interview preparation, see the [Tech Demystified repository](https://github.com/harshitha-8/Tech-Demystified).*

**References and Further Reading:**
- Cover, T. & Hart, P. (1967). "Nearest neighbor pattern classification" - Original paper
- Hastie, T., Tibshirani, R., & Friedman, J. - "The Elements of Statistical Learning" (Chapter 13)
- Scikit-learn k-NN Documentation: https://scikit-learn.org/stable/modules/neighbors.html
- Beyer et al. (1999). "When Is 'Nearest Neighbor' Meaningful?" - Curse of dimensionality
- FAISS: https://github.com/facebookresearch/faiss - Fast approximate nearest neighbors
