# I Messed Up My Amazon Interview – Let’s Make Sure You Don’t!

*By BuildML, reconstructed and reorganized for Tech Demystified*

---

## Introduction

In 2020, I got my first callback for a FAANG company—Amazon. With limited experience in machine learning interviews, I prepped mostly on Data Structures and Algorithms, brushed up on my own projects, and reviewed Amazon Leadership Principles. I hoped it would be enough.

The interview started well enough, but soon, the interviewer dove into basic machine learning questions, beginning with:

> "**Besides linearity, give me some of the other assumptions of linear regression**."

I was caught off guard—after years spent working on XGBoost, RNNs, Transformers, and deep architectures, I had forgotten the simple foundations!

Fast forward one week: HR called. Not this time.

Now, after six years in the field, and after prepping mentees for ML interviews, I want to ensure *you* don’t face the same pitfalls. This technical paper compiles the most actionable questions and answers around **Linear Regression**—a fan favorite with interviewers.

---

## Table of Contents

- [The Story & What You’ll Learn](#introduction)
- [Images & Sources](#images--sources)
- [Key Linear Regression Interview Questions & Answers](#key-questions--answers)
  - [High-Level Explanation](#explain-lr-high-level)
  - [Core Assumptions](#lr-assumptions)
  - [When is Linear Regression Appropriate?](#lr-when-appropriate)
  - [Interpreting Coefficients](#lr-coefficients)
  - [Objective Function](#lr-objective)
  - [Multicollinearity](#lr-multicollinearity)
  - [R² vs RMSE](#lr-r2-rmse)
  - [Validation vs. Overfitting](#lr-overfitting)
  - [L1 vs L2 Regularization](#lr-regularization)
  - [Outliers](#lr-outliers)
  - [Feature Scaling](#lr-feature-scaling)
  - [Feature Selection](#lr-feature-selection)  
  - [Underfitting](#lr-underfitting)
  - [Understanding Model Mistakes](#lr-mistakes)
  - [Handling Missing Data](#lr-missing)
  - [Explaining to Non-technical Stakeholders](#lr-explaining)
  - [Linear vs Nonlinear/Complex Models](#lr-complex)
  - [Training Data Amount Impact](#lr-data)
  - [Coefficient Instability](#lr-instability)
  - [Categorical Variables](#lr-categorical)
  - [Time-based Stability](#lr-stability)
  - [Comparing Two Similar Models](#lr-compare)
  - [Reducing Sensitivity](#lr-sensitivity)
  - [Diagnosing Model vs Data Problems](#lr-blame)
  - [Feature Transformations](#lr-transform)
- [References and Source Attribution](#references)

---

## Images & Sources

1. **Linear Regression Concept**
   ![Concept Illustration](https://substackcdn.com/image/fetch/$s_!HlFj!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb2479e6b-e71f-4506-ab89-ca1f1dcacaf7_700x525.gif)
   *[Source: Medium](https://medium.com/@novus_afk/linear-regression-an-overview-13d37a6bc4dd)*

2. **Mean Squared Error and Learning Curve**
   ![Learning Curve](https://substackcdn.com/image/fetch/$s_!2IQF!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1ae56a5b-7a40-422d-8d38-ae001bfa8b34_800x400.gif)
   *[Source: LinkedIn](https://www.linkedin.com/pulse/linear-regressionmostly-asked-questions-manralaitop30-manral-/)*

3. **Outliers**
   ![Outliers Example](https://substackcdn.com/image/fetch/$s_!V6al!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd832add0-c46e-4366-b20a-17733e5bb82e_1152x1152.png)
   *[Source: Kaggle](https://www.kaggle.com/discussions/questions-and-answers/480609)*

4. **Linear Regression vs Neural Networks**
   ![LR vs NN](https://substackcdn.com/image/fetch/$s_!iFXh!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7d9c3562-c94f-41d9-b5d8-057fe0c4cc9c_1001x471.webp)
   *[Source: GeeksForGeeks](https://www.geeksforgeeks.org/machine-learning/linear-regression-vs-neural-networks-understanding-key-differences/)*

---

## Key Questions & Answers

### <a name="explain-lr-high-level"></a>Can you explain how linear regression works at a high level?

Linear regression tries to predict numeric outcomes (like house prices) by assuming each factor (e.g., rooms, size) adds a fixed value to the prediction. The model learns weights for these factors by minimizing *how far off* its predictions are (usually using mean squared error).

### <a name="lr-assumptions"></a>What assumptions does linear regression make?

- **Linearity**: The outcome is a weighted sum of the inputs.
- **Independence of errors**: Residuals from different data points are not correlated.
- **Constant variance (homoscedasticity)**: Error spreads are similar across predictions.
- **Normality of errors:** Especially important for inference, not just prediction.
- **No strong multicollinearity:** Inputs are not near-duplicates.

*In practice, moderate violations of these assumptions don’t necessarily break the model—but they do impact interpretability and confidence bounds.*

### <a name="lr-when-appropriate"></a>How do you decide when to use linear regression?

- **Goal:** When you want interpretability or a baseline.
- **Relationship:** Trends look roughly linear, or simple feature interactions exist.
- **Size/quality of data:** Linear regression is robust to noise and works with moderate dataset sizes.
- **Empirical validation:** If validation performance is reasonable and residuals show no pattern, LR is appropriate.

### <a name="lr-coefficients"></a>How do you interpret linear regression coefficients?

Each coefficient tells you the *average* effect of increasing an input by 1 unit (holding others constant).
- Positive coefficient: Value increases output.
- Negative coefficient: Value decreases output.
- **Intercept:** Expected output when all inputs are zero (not always meaningful in practice).

> If features are correlated, direct interpretation gets messy.

### <a name="lr-objective"></a>What does the objective function of linear regression represent?

Usually, LR minimizes *mean squared error*—the average squared distance between predictions and actual values. This squared term emphasizes larger errors.

Optimization often uses:
- **Closed form** (if data is small/”nice”)
- **Gradient descent** otherwise

### <a name="lr-multicollinearity"></a>What happens if features are highly correlated?

- **Prediction can still be good**, but coefficients become unstable—small data changes lead to big swings in weights.
- **Interpretation** becomes unreliable.
- Fixes: remove redundant features, combine highly correlated inputs, or use regularization.

### <a name="lr-r2-rmse"></a>What's the difference between R² and RMSE?

- **R²:** How much variance in the target is explained by the model (unitless, 0–1).
- **RMSE:** Typical error size, in actual units of the prediction (e.g., dollars, kg).

### <a name="lr-overfitting"></a>Your LR model fits training well but does badly on validation. What first?

Check for:
- **Overfitting**: Too many (esp. redundant or weak) features.
- **Multicollinearity.**
- **Lack of regularization.**
- **Data leakage** or distribution differences in training/validation split.

### <a name="lr-regularization"></a>Difference between L1 and L2 regularization?

- **L2 (Ridge):** Shrinks all coefficients; keeps all features.
- **L1 (Lasso):** Pushes some coefficients to zero; automatic feature selection.
- Both can be combined (*Elastic Net*).

### <a name="lr-outliers"></a>How do outliers affect linear regression?

LR is sensitive to outliers due to squared error loss—outliers can “pull” the line, distorting fit for normal points.
- Fix: investigate, transform (log), or use robust regression variants.

### <a name="lr-feature-scaling"></a>Should you scale features for LR?

Scaling is critical if you use
- **Gradient descent optimizers**
- **Regularization terms**

It makes training faster, more stable, and coefficient interpretations easier.

### <a name="lr-feature-selection"></a>How do you decide which features to include?

- **Domain knowledge**
- **Data exploration (plots, stats)**
- **Empirical validation** on held-out data
- **Regularization (esp. L1)**
- **Stability checks** (do coefficients change wildly across splits?)

### <a name="lr-underfitting"></a>Diagnosing underfitting?

- LR fails on both training and validation with similar errors.
- Residual plots show clear patterns (unmodeled structure).
- More flexible model or better features improve performance.

### <a name="lr-mistakes"></a>How do you understand model mistakes?

**Residual plots**: If residuals vs. predictions or inputs show patterns, the model misses some structure. Outliers, heteroscedasticity, or bias can be identified.

### <a name="lr-missing"></a>How do missing values impact LR?

- LR can't handle missing values directly—most implementations drop those rows.
- With much missingness, this can bias results.
- Consider mean/median imputation, introducing a “missing” category for categoricals, or model-based imputation.

### <a name="lr-explaining"></a>How would you explain LR results to a nontechnical stakeholder?

- **Tell a story:** “This model looks at key factors and learns how each is associated with the outcome.”
- **Translate coefficients:** “X increases by Y units is associated with Z change in output, on average.”
- **Summarize performance:** Use intuitive, relatable terms—“On average, predictions are off by $X.”

### <a name="lr-complex"></a>When prefer LR over trees/neural nets?

- When interpretability is key.
- Problem is simple or data is limited.
- You want a strong, explainable baseline.

### <a name="lr-data"></a>How does the amount of training data affect LR performance?

- **More data:** More stable, reliable coefficients.
- **Too little data:** Coefficients move a lot; model is sensitive to noise.

### <a name="lr-instability"></a>If coefficients change a lot between retrains but metrics don't, why?

- Usually **feature correlation**—weights' sums matter more than individual values.
- Can also be due to small datasets or weak regularization.

### <a name="lr-categorical"></a>How do you handle categorical variables?

- **One-hot encoding:** Most common; don’t forget to drop one for reference.
- High-cardinality: Consider grouping or target encoding (with care to avoid leakage).

### <a name="lr-stability"></a>How do you assess model stability over time?

- Evaluate with time-based splits. If performance or coefficients degrade or shift, model/data process may have changed.

### <a name="lr-compare"></a>Comparing two LRs with same accuracy but different coefficients?

- Prefer the model with more stable, interpretable, or regularized coefficients; interpret with caution if features are correlated.

### <a name="lr-sensitivity"></a>How do you fix sensitivity to small changes in the data?

- Remove/reduce correlated features.
- Add regularization.
- Scale features.
- Check and transform outliers, or collect more data if possible.

### <a name="lr-blame"></a>How to tell if poor performance is model or data?

- If both train/val error are high: *Probably a data or feature problem*.
- If only val error is high: *Probably overfitting/model issue*.
- Inspect residual patterns and label quality.

### <a name="lr-transform"></a>What feature transformations to try before giving up on LR?

- Log/sqrt/reciprocal transforms
- Polynomial terms
- Pairwise interaction terms
- Binning
- Target transformation (e.g., log target)

---
