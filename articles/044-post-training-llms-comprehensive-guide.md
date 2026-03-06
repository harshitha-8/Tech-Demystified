# Introduction to Post-Training for Large Language Models

### A Comprehensive Technical Analysis from Pre-Training to Agentic Reinforcement Learning

**Author**: Maxime Labonne, PhD  
**Affiliation**: Head of Post-Training @ Liquid AI, Computer Laboratory, Cambridge  
**Presentation Date**: March 5, 2026  
**Report Compiled**: March 2026  

**Additional Resources**:
- LLM Engineer's Handbook (Maxime Labonne)
- Hands-On Graph Neural Networks
- LLM Course: https://github.com/mlabonne/llm-course (76K+ stars)
- LLM Datasets: https://github.com/mlabonne/llm-datasets

---

## Executive Summary

**Post-training** represents the critical phase in large language model (LLM) development that transforms raw autoregressive models into intelligent assistants capable of following instructions, reasoning through complex problems, and aligning with human preferences. This comprehensive report analyzes the complete post-training pipeline based on state-of-the-art methodologies from Liquid AI, covering:

1. **Supervised Fine-Tuning (SFT)**: Converting base models into instruction-following systems
2. **Preference Alignment**: Optimizing model outputs for human preferences using Direct Preference Optimization (DPO)
3. **Reinforcement Learning**: Training reasoning capabilities with Group Relative Policy Optimization (GRPO)
4. **Modern Post-Training**: Agentic RL, multi-objective optimization, and production deployment

**Key Findings**:
- **Data quality matters more than algorithmic sophistication** - clean, diverse datasets outperform complex training procedures
- **On-policy data generation** eliminates policy drift in preference learning
- **Evaluation infrastructure** is essential for iterative improvement
- **Small models** (1B-3B parameters) can achieve strong reasoning with proper post-training

**Industry Impact**: This research directly informs the development of production LLM systems at scale, with applications in conversational AI, coding assistants, scientific reasoning, and autonomous agents.

---

## Table of Contents

1. [Introduction: The Post-Training Paradigm](#part-i-introduction)
2. [Supervised Fine-Tuning (SFT)](#part-ii-supervised-fine-tuning)
3. [Preference Alignment (DPO)](#part-iii-preference-alignment)
4. [Reinforcement Learning (GRPO)](#part-iv-reinforcement-learning)
5. [State-of-the-Art Techniques](#part-v-state-of-the-art-techniques)
6. [Agentic Reinforcement Learning](#part-vi-agentic-reinforcement-learning)
7. [Production Considerations](#part-vii-production-considerations)
8. [Conclusions & Future Directions](#part-viii-conclusions)

---

## Part I: Introduction - The Post-Training Paradigm

### 1.1 What Is Post-Training?

**Post-training** is the umbrella term for all training procedures applied to a pre-trained language model to improve its usefulness, safety, and alignment with human intent. It represents the transformation:

```
Base Model → SFT Model → Instruct Model → Thinking Model
(Autocomplete) (Follow instructions) (Optimized for humans) (Reasoning)
```

**Three-stage pipeline** (from Slide 8):

```
┌────────────────────────────────────────────────────────────────────┐
│ Stage 1: Pre-Training                                              │
├────────────────────────────────────────────────────────────────────┤
│ Input: Raw text from internet, books, code                        │
│ Training: Next-token prediction (autoregressive modeling)         │
│ Output: Base model                                                │
│ Capability: Autocomplete prompts                                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Stage 2: Supervised Fine-Tuning (SFT)                             │
├────────────────────────────────────────────────────────────────────┤
│ Input: Instruction-output pairs (10K-1M samples)                  │
│ Training: Supervised learning on demonstrations                    │
│ Output: SFT model                                                 │
│ Capability: Follow instructions                                   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Stage 3: Preference Alignment                                     │
├────────────────────────────────────────────────────────────────────┤
│ Input: Preference pairs (chosen vs rejected responses)            │
│ Training: DPO, PPO, GRPO                                          │
│ Output: Instruct model                                            │
│ Capability: Optimized for human preferences                       │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Stage 4: Reinforcement Learning (Optional)                        │
├────────────────────────────────────────────────────────────────────┤
│ Input: Reasoning tasks with verifiable answers                    │
│ Training: GRPO with reward functions                              │
│ Output: Thinking model                                            │
│ Capability: Chain-of-thought reasoning (e.g., DeepSeek-R1, o1)    │
└────────────────────────────────────────────────────────────────────┘
```

**Key Distinction** (Slide 9): **Fine-tuning vs Post-training**

| **Aspect** | **Fine-Tuning** | **Post-Training** |
|------------|-----------------|-------------------|
| **Dataset Size** | >1M samples | 10K-1M samples |
| **Purpose** | Task-specific adaptation | General-purpose instruction-following |
| **Examples** | Data extraction, classification | Conversational AI, coding, reasoning |

### 1.2 When to Fine-Tune vs When to Use RAG

**Decision framework** (Slide 10):

**Start with in-context learning (ICL) and Retrieval-Augmented Generation (RAG) when**:
- ✅ You need to add superficial knowledge
- ✅ You want to change tone and format
- ✅ You're prototyping quickly

**Fine-tune when you need to**:
- ✅ Reduce cost and latency (smaller models after distillation)
- ✅ Increase output quality beyond what prompting can achieve
- ✅ Teach new formats or reasoning patterns

**Evaluation is critical** (Slide 10): Always measure before and after to validate improvements.

### 1.3 Fine-Tuning Frameworks

**Two dominant libraries** (Slide 11):

#### **A. TRL (Transformers Reinforcement Learning)**
- **Maintainer**: Hugging Face
- **Strengths**: Most up-to-date, highly customizable for research
- **Use case**: PhD research, novel algorithms, bleeding-edge techniques

#### **B. Unsloth**
- **Maintainer**: Unsloth AI
- **Strengths**: Beginner-friendly, many additional features (quantization, optimization)
- **Use case**: Production deployment, rapid experimentation

### 1.4 The Post-Training Stack

**Three pillars** (Slide 12):

```
1. Dataset
   ├─ Data generation (LLM synthesis, web scraping, human annotation)
   ├─ Scoring (reward models, LLM-as-judge, human evaluation)
   ├─ Filtering (deduplication, quality thresholds, format validation)
   └─ Exploration (embeddings, clustering, diversity analysis)

2. Fine-Tuning
   ├─ SFT (Supervised Fine-Tuning)
   ├─ DPO (Direct Preference Optimization)
   ├─ RL (Reinforcement Learning: GRPO, PPO)
   └─ Model Merging (SLERP, TIES, DARE)

3. Evaluation
   ├─ Automated benchmarks (MMLU, HumanEval, GSM8K)
   ├─ Judge LLMs (GPT-4-as-judge, Claude-as-judge)
   └─ Human evaluations (pairwise comparisons, Elo ratings)
```

---

## Part II: Supervised Fine-Tuning (SFT)

### 2.1 Instruction Data Format

**Standard format** (Slide 14):

```python
{
    "system": "You are a helpful assistant.",  # Optional
    "instruction": "Remove the spaces from the following sentence: Fine-tuning is simple.",
    "output": "Fine-tuningissimple."
}
```

**Context**: This trains the model to map **instructions → outputs** in a supervised manner.

### 2.2 What Makes a Good Dataset?

**Three critical properties** (Slide 15):

#### **1. Accuracy**
- Factually correct information
- Verifiable outputs (especially for math, code, reasoning)
- No hallucinations or fabricated content

#### **2. Diversity**
- Wide range of topics, styles, formats
- Multiple domains (math, science, creative writing, code)
- **High-diversity datasets generalize better** (Slide 16)

**Diversity comparison** (from MixEval paper, Slide 16):

```
Low Diversity Dataset:
- Math problems only
- Similar phrasing across examples
- Narrow topic coverage
→ Model overfits to specific patterns

High Diversity Dataset:
- Math, code, conversation, creative writing, reasoning
- Real-life conversations (Reddit, StackOverflow, forums)
- Varied sentence structures and formats
→ Model generalizes to unseen prompts
```

#### **3. Complexity**
- Non-trivial tasks that force reasoning
- Multi-step problems
- Ambiguous situations requiring judgment

**Example: Instruction-following dataset** (Slide 17):

```
Instruction:
"Write a detailed review of the movie 'The Social Network'. Your entire response 
should be in English and all lower case (no capital letters whatsoever)."

Expected Output:
"the social network is a gripping portrayal of the founding of facebook. 
directed by david fincher, the film explores themes of ambition, betrayal, 
and the cost of success. jesse eisenberg delivers a compelling performance 
as mark zuckerberg, capturing the character's complexity..."

Challenge: Model must follow formatting constraints (all lowercase) while 
providing substantive content.
```

### 2.3 Dataset Creation Pipeline

**Automated generation workflow** (Slide 17-18):

```
┌────────────────────────────────────────────────────────────────────┐
│ Step 1: Prompt Generation                                          │
├────────────────────────────────────────────────────────────────────┤
│ Create diverse prompts with constraints:                           │
│ - "Write a poem about X in style Y"                               │
│ - "Explain Z using only words with <6 letters"                    │
│ - "Solve this math problem and show your work"                    │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Step 2: LLM Generation                                             │
├────────────────────────────────────────────────────────────────────┤
│ Query powerful LLM (GPT-4, Claude, Gemini):                       │
│ - Generate responses for each prompt                               │
│ - Use temperature=0.7 for diversity                               │
│ - Sample multiple responses per prompt                             │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Step 3: Quality Filtering                                         │
├────────────────────────────────────────────────────────────────────┤
│ Run automated tests:                                               │
│ - Format validation (check constraints met)                        │
│ - Decontamination (7-gram overlap with eval benchmarks)           │
│ - Keyword exclusion (remove leaked benchmark terms)               │
│ - Length checks (min/max token counts)                            │
└────────────────────────────────────────────────────────────────────┘
```

**Real-world example** (Slide 18): **mlabonne/open-perfectblend**

This dataset combines multiple categories:

| **Category** | **Description** | **Example Datasets** |
|--------------|-----------------|----------------------|
| **Instruction Following** | Complex constraints | IFEval, ConstraintQA |
| **Reasoning** | Math, logic, science | GSM8K, MATH, ARC |
| **Code** | Programming tasks | HumanEval, MBPP |
| **Conversation** | Open-ended dialogue | ShareGPT, WildChat |
| **Safety** | Harmlessness, refusals | Anthropic HH-RLHF |

**Mixture of Judges approach** (from "The Perfect Blend" paper, Slide 18):
- Use multiple judge models to score responses
- Aggregate judgments (e.g., average scores)
- Keep only high-quality examples (e.g., score > 8/10)

### 2.4 SFT Training Techniques

**Three approaches** (Slide 19):

```
┌─────────────────────────────────────────────────────────────────┐
│ Full Fine-Tuning                                                │
├─────────────────────────────────────────────────────────────────┤
│ Precision: 16-bit (FP16 or BF16)                               │
│ Parameters Updated: All weights                                 │
│ VRAM Usage: Very High (70B model ≈ 280GB+)                     │
│ Training Speed: Fastest                                         │
│ Quality: Maximizes performance                                  │
│ Use Case: When you have unlimited GPU budget                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ LoRA (Low-Rank Adaptation)                                     │
├─────────────────────────────────────────────────────────────────┤
│ Precision: 16-bit                                              │
│ Parameters Updated: Low-rank adapters (A, B matrices)          │
│ VRAM Usage: High (70B model ≈ 80GB)                           │
│ Training Speed: Fast                                            │
│ Quality: ~95% of full fine-tuning                             │
│ Use Case: Standard research/production setting                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ QLoRA (Quantized LoRA)                                         │
├─────────────────────────────────────────────────────────────────┤
│ Precision: 4-bit quantization (NF4)                           │
│ Parameters Updated: Low-rank adapters in 4-bit               │
│ VRAM Usage: Low (70B model ≈ 40GB)                            │
│ Training Speed: Moderate (quantization overhead)               │
│ Quality: ~90% of full fine-tuning                             │
│ Use Case: Consumer GPUs, budget constraints                    │
└─────────────────────────────────────────────────────────────────┘
```

**LoRA mathematical formulation**:

Instead of updating full weight matrix \( W \in \mathbb{R}^{d \times k} \), we learn:

\[
W' = W + \Delta W = W + BA
\]

Where:
- \( B \in \mathbb{R}^{d \times r} \), \( A \in \mathbb{R}^{r \times k} \)
- \( r \ll \min(d, k) \) (rank, typically 8-64)
- Only \( A \) and \( B \) are trainable (reduces parameters by 1000×)

**QLoRA innovation** (Xu et al., 2023):
- Quantize base model \( W \) to 4-bit (NF4 = Normal Float 4)
- Keep adapters \( A, B \) in 16-bit
- Use paged optimizers to handle memory spikes

### 2.5 Training Hyperparameters

**Critical parameters** (Slide 20):

| **Parameter** | **Description** | **Common Values** | **Priority** |
|---------------|-----------------|-------------------|--------------|
| **Learning Rate** | Strength of parameter update | 1e-6 to 1e-3 | ⭐⭐⭐ (tune this first) |
| **Epochs** | Number of passes over dataset | 3 to 5 | ⭐⭐ |
| **Batch Size** | Samples before gradient update | 8 or 16 (effective) | ⭐⭐ |
| **Max Length** | Longest input in tokens | 1024 to 4096 | ⭐ |
| **Optimizer** | Parameter update algorithm | AdamW | ⭐ (don't change) |
| **Attention** | Attention implementation | FlashAttention-2 | ⭐ (don't change) |

**Trade-offs**:
- **Learning rate too high** → Loss spikes, divergence, instability
- **Learning rate too low** → Slow convergence, underfitting
- **Too many epochs** → Overfitting (model memorizes training data)
- **Too few epochs** → Underfitting (model doesn't learn patterns)

### 2.6 Monitoring Experiments

**Loss curves** (Slide 21):

```
Good Training (smooth curve):
Loss
  │
  │╲
  │ ╲
  │  ╲___
  │      ─────___
  │              ────___
  └────────────────────────> Steps

Bad Training (loss spike):
Loss
  │     ╱╲
  │    ╱  ╲
  │   ╱    ╲  ╱
  │  ╱      ╲╱
  │ ╱
  └────────────────────────> Steps
      ↑
      Learning rate too high

Overfitting:
Loss
  │
Training │╲___
Loss     │    ────____
         │            ────____
         │
Validation│╲____
Loss      │     ╲
          │      ╲____
          │           ╲____
          └─────────────────────> Steps
                        ↑
                        Val loss increases (stop here)
```

**Best practices**:
- Monitor both training and validation loss
- Use early stopping when validation loss increases
- Log learning rate, gradient norms, perplexity
- Visualize with TensorBoard or Weights & Biases

---

## Part III: Preference Alignment (Direct Preference Optimization)

### 3.1 Preference Data Format

**Standard format** (Slide 23):

```python
{
    "system": "You are a helpful assistant with a great sense of humor.",
    "instruction": "Tell me a joke about octopuses.",
    "chosen": "Why don't octopuses play cards in casinos? Because they can't count past eight.",
    "rejected": "How many tickles does it take to make an octopus laugh? Ten tickles."
}
```

**Key insight**: Instead of providing a single "correct" answer (SFT), we provide **pairwise preferences** (A is better than B). This teaches the model to optimize for quality, not just accuracy.

### 3.2 Preference Dataset Creation

**Ultrafeedback pipeline** (offline distillation, Slide 24):

```
┌────────────────────────────────────────────────────────────────────┐
│ Step 1: Multi-Model Generation                                     │
├────────────────────────────────────────────────────────────────────┤
│ For each prompt, query N different LLMs:                           │
│ - LLM 1 (e.g., GPT-4)     → Response 1                            │
│ - LLM 2 (e.g., Claude)    → Response 2                            │
│ - LLM n (e.g., Gemini)    → Response n                            │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Step 2: Judge LLM Scoring                                          │
├────────────────────────────────────────────────────────────────────┤
│ Use powerful judge model (e.g., GPT-4) to score all responses:    │
│ - Score on scale 1-10 for multiple dimensions:                    │
│   * Helpfulness                                                    │
│   * Accuracy                                                       │
│   * Conciseness                                                    │
│   * Safety                                                         │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Step 3: Filtering and Selection                                    │
├────────────────────────────────────────────────────────────────────┤
│ - Remove duplicates (semantic deduplication)                       │
│ - Remove short answers (<50 tokens)                               │
│ - Select highest-scored response as "chosen"                      │
│ - Select lower-scored response as "rejected"                      │
│ - Keep only pairs with significant score gap (e.g., Δ > 2)       │
└────────────────────────────────────────────────────────────────────┘
```

**Real-world example** (Slide 25): **mlabonne/orpo-dpo-mix-40k**

Quality improvements:
- ✅ **Scoring filter**: Keep only highly scored chosen answers (score ≥ 8/10)
- ✅ **Rule-based filtering**: Remove "GPTisms" in chosen answers
  - Remove: "Certainly!", "I'd be happy to help!", "Here's what I found:", etc.
  - Keep: Direct, concise answers
- ✅ **Data quality >> Training algorithm** (most important lesson)

### 3.3 Direct Preference Optimization (DPO) Algorithm

**Visual explanation** (Slides 26-29):

#### **Step 1: Sample from models**

```
Input: Prompt x
Policy model π_θ:      Sample y_w (higher quality)
Reference model π_ref: Sample y_l (lower quality)
```

#### **Step 2: Compute log probabilities**

```
Policy model:
- log π_θ(y_w | x) = log probability of chosen response
- log π_θ(y_l | x) = log probability of rejected response

Reference model (frozen):
- log π_ref(y_w | x)
- log π_ref(y_l | x)
```

#### **Step 3: Compute implicit rewards**

DPO implicitly defines a reward function:

\[
r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
\]

Where:
- \( \beta \) = inverse temperature (controls divergence from reference)
- Higher reward when policy assigns higher probability than reference

#### **Step 4: DPO loss function**

\[
\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
\]

**Intuition**: DPO **widens the probability gap** between chosen and rejected responses, **relative to the reference model**.

**Why this works**:
- Maximizes likelihood of chosen responses
- Minimizes likelihood of rejected responses
- KL penalty (via reference model) prevents overfitting

### 3.4 DPO Training Parameters

**Critical hyperparameters** (Slide 30):

| **Parameter** | **Description** | **Common Values** |
|---------------|-----------------|-------------------|
| **Beta (β)** | Importance of reference model | 0.1 to 0.5 |
| **Learning Rate** | Parameter update strength | 1e-7 to 1e-4 |
| **Epochs** | Passes over dataset | 2 to 5 |
| **Batch Size** | Samples per update | 8 or 16 (effective) |
| **Max Length** | Longest input in tokens | 512 to 2048 |

**Beta (β) trade-off**:
- **High β** (e.g., 0.5): Strong KL penalty, stays close to reference → conservative, safe outputs
- **Low β** (e.g., 0.1): Weak KL penalty, explores more → creative, potentially risky outputs

### 3.5 DPO Challenges: Off-Policy Data Problem

**Problem** (Slide 31): **Policy drift**

```
Reference model π_ref      Policy model π_θ*
(fixed)                    (after training)
     
     ↓                            ↓
Generates data A           Generates data B
(on-policy at start)       (off-policy now)

Issue: Training data came from π_ref, but we're optimizing π_θ
→ Answers generated by π_θ might be better than "chosen" in dataset!
→ Model penalized for generating good responses
```

**Visual representation** (Slide 31):

```
Off-Policy (BAD):
π_ref ───[generates data]──→ Dataset
π_θ   ───[trained on old data]──→ Misaligned

On-Policy (GOOD):
π_θ ───[generates fresh data]──→ Dataset
π_θ ───[trained on own data]──→ Aligned
```

### 3.6 Solution: On-Policy Data Generation

**Liquid AI's approach** (Slides 32-35):

```
┌────────────────────────────────────────────────────────────────────┐
│ Step 1: Generation (~1M prompts)                                   │
├────────────────────────────────────────────────────────────────────┤
│ Current policy model π_θ generates G outputs per prompt:          │
│ - o₁, o₂, ..., o_G (typically G=4 to 8)                          │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Step 2: LLM Jury Scoring                                          │
├────────────────────────────────────────────────────────────────────┤
│ Judge model scores all G outputs:                                  │
│ - Chosen: highest-scored output                                    │
│ - Rejected: lowest-scored output                                   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Step 3: Heuristic Filtering                                        │
├────────────────────────────────────────────────────────────────────┤
│ Remove low-quality pairs:                                          │
│ - Score gap too small (Δ < 2)                                     │
│ - Both responses too short                                         │
│ - Semantic similarity too high (duplicates)                        │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Step 4: Reviser Model (Optional)                                   │
├────────────────────────────────────────────────────────────────────┤
│ Use stronger LLM to refine chosen answer:                         │
│ - Fix grammar, formatting                                          │
│ - Add missing details                                              │
│ - Remove unsafe content                                            │
│ Result: Higher-quality "chosen" response                           │
└────────────────────────────────────────────────────────────────────┘
```

**Benefits**:
- ✅ Data always matches current policy distribution
- ✅ No policy drift
- ✅ Model can improve iteratively (bootstrap from its own outputs)

### 3.7 State-of-the-Art DPO Techniques

**Five advanced methods** (Slide 36):

#### **1. Length Normalization**
Problem: Longer responses get higher raw log-probs
Solution: Normalize by token count

\[
\text{reward}_{\text{norm}} = \frac{1}{|y|} \log \pi_\theta(y|x)
\]

#### **2. Anchored Preference Optimization (APO)**
Problem: DPO only uses pairwise comparisons (chosen vs rejected)
Solution: Add "anchor" reference response (e.g., from GPT-4)

\[
\mathcal{L}_{\text{APO}} = \mathcal{L}_{\text{DPO}} + \lambda \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{anchor}})
\]

Benefits:
- Prevents catastrophic forgetting
- Maintains capabilities from strong reference model

#### **3. Refine Chosen Answers**
- Use stronger model (e.g., GPT-4) to polish chosen responses
- Fix typos, improve clarity, add structure
- Result: Higher-quality training signal

#### **4. Rubric Scoring**
Instead of single scalar score, use multi-dimensional rubric:

```
Rubric dimensions:
- Accuracy: 8/10
- Clarity: 9/10
- Conciseness: 6/10
- Safety: 10/10

Aggregate score: weighted average
```

Benefits:
- More nuanced feedback
- Can optimize for specific dimensions
- Interpretable improvements

#### **5. Multi-Objective Optimization**
Optimize multiple objectives simultaneously:

\[
\mathcal{L}_{\text{multi}} = \alpha_1 \mathcal{L}_{\text{DPO}} + \alpha_2 \mathcal{L}_{\text{safety}} + \alpha_3 \mathcal{L}_{\text{instruction-following}}
\]

---

## Part IV: Reinforcement Learning (Group Relative Policy Optimization)

### 4.1 RL Data Format

**Two formats** (Slide 38):

#### **Format 1: Instruction data (cold start)**
```python
{
    "system": "Think step by step and write the final answer in \\boxed{}.",
    "instruction": "Simplify the expression $\\cos^2(x) - \\sin^2(x)$.",
    "output": "<think> Okay, I need to simplify cos²(x) - sin²(x)...</think>\nThe answer is \\boxed{\\cos(2x)}."
}
```

Used for:
- Initial SFT on reasoning traces
- Teaching <think> token usage
- Format training

#### **Format 2: RL data (verifiable answers)**
```python
{
    "system": "Think step by step and write the final answer in \\boxed{}.",
    "instruction": "Simplify the expression $\\cos^2(x) - \\sin^2(x)$.",
    "ground_truth": "\\cos(2x)"
}
```

Used for:
- GRPO training with reward function
- Verifying correctness automatically
- Iterative improvement

### 4.2 RL Dataset Creation

**DeepSeek-R1 distillation pipeline** (Slide 39):

```
┌────────────────────────────────────────────────────────────────────┐
│ Step 1: Generate Reasoning Traces                                  │
├────────────────────────────────────────────────────────────────────┤
│ Query DeepSeek-R1 with math/science/code prompts:                 │
│ - Model generates <think>...</think> reasoning                     │
│ - Final answer in \\boxed{}                                        │
│ - Collect ~1M reasoning traces                                     │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Step 2: Filter Wrong Answers                                       │
├────────────────────────────────────────────────────────────────────┤
│ For math/code: Check if answer matches ground truth               │
│ - Parse \\boxed{answer}                                            │
│ - Compare with known solution                                      │
│ - Keep only correct traces                                         │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Step 3: Deduplication and Format Checking                          │
├────────────────────────────────────────────────────────────────────┤
│ - Semantic deduplication (embedding similarity)                    │
│ - Format validation (<think> and \\boxed{} present)               │
│ - Length filtering (remove very short/long traces)                │
└────────────────────────────────────────────────────────────────────┘
```

**Real-world dataset composition** (Slide 40):

**LFM2-350M-Math instruction dataset**:

| **Dataset** | **Category** | **# Samples** | **Percentage** |
|-------------|--------------|---------------|----------------|
| nvidia/OpenMathReasoning | Math | 3.2M | 72% |
| nvidia/Llama-Nemotron (Science) | Science | 709K | 16% |
| EricLu/SCP-116K | Science+Math | 274K | 12% |
| **Total** | | **4.18M** | **100%** |

**LFM2-350M-Math RL dataset**:

| **Dataset** | **Category** | **# Samples** | **Percentage** |
|-------------|--------------|---------------|----------------|
| BytedTsinghua-SIA/DAPO-Math-17k | Math | 3,537 | 31% |
| nvidia/OpenMathReasoning | Math | 3,183 | 28% |
| openai/gsm8k | Math | 1,864 | 16% |
| nvidia/Llama-Nemotron (Science) | Science | 1,339 | 12% |
| EleutherAI/hendrycks_math | Math | 848 | 8% |
| agentica-org/DeepScaleR | Math | 602 | 5% |
| **Total** | | **11,373** | **100%** |

**Key insight**: RL dataset is 2-3 orders of magnitude smaller than SFT dataset (11K vs 4M samples).

### 4.3 Group Relative Policy Optimization (GRPO) Algorithm

**Core idea**: GRPO normalizes rewards **within a group of sampled outputs** to estimate advantages, eliminating the need for a separate critic model.

**Visual explanation** (Slides 41-44):

#### **Step 1: Sample group of outputs**

```
Input: Prompt x
Policy model π_θ generates G outputs:
- o₁, o₂, ..., o_G (typically G=4 to 16)
```

#### **Step 2: Compute rewards**

```
Reward function r(x, o) evaluates each output:
- r₁ = r(x, o₁)
- r₂ = r(x, o₂)
- ...
- r_G = r(x, o_G)
```

#### **Step 3: Group normalization (compute advantages)**

\[
A_i = \frac{r_i - \mu}{\sigma}
\]

Where:
- \( \mu = \frac{1}{G} \sum_{i=1}^G r_i \) = mean reward across group
- \( \sigma = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \mu)^2} \) = standard deviation

**Why this works**: Advantages tell us which outputs are **better/worse than average for this specific prompt**.

#### **Step 4: GRPO loss function**

\[
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^G A_i \cdot \log \pi_\theta(o_i | x) + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
\]

**Intuition**:
- Outputs with **positive advantage** (above average) → **increase** their probability
- Outputs with **negative advantage** (below average) → **decrease** their probability
- KL term prevents policy from drifting too far from reference

**Benefits over PPO**:
- ✅ No critic model needed (saves memory, training time)
- ✅ More stable (no critic-policy mismatch)
- ✅ Simpler to implement

### 4.4 Reward Functions

**Two types** (Slide 45):

#### **A. Easy Reward: Rule-Based**

Example: "Answer should be about 50 characters"

```python
def reward_len(answer, **kwargs):
    return -abs(50 - len(answer))
```

Result:
- Answers longer or shorter than 50 characters get negative reward
- Answer exactly 50 characters gets reward = 0 (best)

#### **B. Hard Reward: Learned Reward Model**

```
┌────────────────────────────────────────────────────────────────────┐
│ Reward Model Architecture                                          │
├────────────────────────────────────────────────────────────────────┤
│ Input: Prompt + Answer                                             │
│ Encoder: LLM (e.g., 7B parameter model)                           │
│ Output head: Linear layer → scalar score [0, 1]                   │
│                                                                    │
│ Training: Supervised on preference data                            │
│ - Chosen responses → score ≈ 1.0                                  │
│ - Rejected responses → score ≈ 0.0                                │
└────────────────────────────────────────────────────────────────────┘
```

**For math problems**: Reward = 1 if correct, 0 if wrong (binary)

```python
def reward_math(answer, ground_truth, **kwargs):
    predicted = extract_boxed_answer(answer)
    return 1.0 if predicted == ground_truth else 0.0
```

### 4.5 Monitoring RL Experiments

**Counter-intuitive behavior** (Slide 46):

```
Observation during RL training:
- Loss goes UP ↑
- Average response length goes DOWN ↓

Why? KL divergence penalty!
```

**Explanation**:

```python
Loss = -E[A * log π_θ(o|x)] + β * KL(π_θ || π_ref)
       ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
       RL objective               Regularization
       (decreases over time)      (increases over time)
```

As model improves:
1. RL objective (first term) decreases → model generates higher-reward outputs
2. KL penalty (second term) increases → policy drifts from reference
3. **Total loss can increase** even though RL objective improves

**Length decrease**:
- If reward function prefers concise answers (e.g., math = just the number)
- Model learns to generate shorter responses
- This is desired behavior!

**What to monitor**:
- ✅ Mean reward (should increase)
- ✅ Pass@1 accuracy (for verifiable tasks)
- ✅ Response length (should match target)
- ✅ KL divergence (should stay reasonable, < 5.0)

### 4.6 State-of-the-Art RL Techniques

**Five advanced methods** (Slide 47):

#### **1. Asymmetric Ratio Clipping**
Problem: PPO clips ratio symmetrically, but we want different behavior for good/bad actions
Solution: Clip differently for advantages > 0 vs < 0

\[
\text{clip}_{\text{asym}}(r, \epsilon) = \begin{cases}
\min(r, 1 + \epsilon) & \text{if } A > 0 \\
\max(r, 1 - 2\epsilon) & \text{if } A < 0
\end{cases}
\]

#### **2. No Advantage Normalization**
Observation: Group normalization can be unstable for small groups
Solution: Use raw advantages (no normalization by std)

#### **3. Filtering of Zero-Variance Groups**
Problem: If all outputs in group have same reward → std = 0 → NaN
Solution: Skip groups where reward variance < threshold

#### **4. Overlong-Sample Masking**
Problem: Very long responses (>4096 tokens) dominate gradient
Solution: Mask out tokens beyond max length in loss computation

#### **5. Truncated Importance Sampling**
Problem: Importance weights \( \frac{\pi_\theta}{\pi_{\text{old}}} \) can explode
Solution: Clip weights to [0.5, 2.0]

**Key papers** (Slide 47):
- Ahmadian et al. (2024): "Back to Basics: Revisiting REINFORCE Style Optimization"
- Shao et al. (2024): "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
- Yu et al. (2025): "DAPO: An Open-Source LLM RL System at Scale"
- Liu et al. (2025): "Understanding R1-Zero-Like Training"
- Yao et al. (2025): "Your Efficient RL Framework Secretly Brings You Off-Policy Training"

---

## Part V: State-of-the-Art Post-Training Techniques

### 5.1 Historical Timeline of Post-Training

**Evolution of post-training methods** (Slide 49):

```
┌────────────────────────────────────────────────────────────────────┐
│ SFT Era (2017-2023)                                                │
├────────────────────────────────────────────────────────────────────┤
│ 2017: Deep RL from Human Preferences (Christiano et al., OpenAI)  │
│ 2020: GPT-3 + Few-Shot Prompting (Brown et al., OpenAI)           │
│ 2022 Jan: InstructGPT (Ouyang et al., OpenAI)                     │
│ 2022 Nov: ChatGPT Launch (OpenAI)                                 │
│ 2023 Oct: Zephyr (Tunstall et al., Hugging Face)                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ XPO Era (2023-2024)                                                │
├────────────────────────────────────────────────────────────────────┤
│ 2023 May: Direct Preference Optimization (Rafailov et al.)        │
│ 2024 Apr: Llama 3 Instruct (Meta)                                 │
│ - Introduced: DPO, IPO, KTO, ORPO, SimPO variants                 │
│ - Key innovation: Off-policy → On-policy data generation          │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ RL Era (2024-Present)                                              │
├────────────────────────────────────────────────────────────────────┤
│ 2024 Sep: OpenAI o1 (OpenAI)                                      │
│ 2025 Jan: DeepSeek R1 (DeepSeek)                                  │
│ - Introduced: GRPO, process rewards, thinking tokens              │
│ - Key innovation: Verifiable reasoning with RL                    │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 Modern Post-Training Architecture

**Three-stage pipeline** (combining all techniques):

```python
class ModernPostTraining:
    """
    State-of-the-art post-training pipeline (2026).
    
    Combines:
    - SFT with diverse, high-quality data
    - On-policy DPO with advanced techniques
    - GRPO for reasoning tasks
    """
    
    def __init__(self, base_model):
        self.model = base_model
        self.reference_model = copy.deepcopy(base_model)  # Frozen
        
    def stage_1_sft(self, instruction_data, epochs=3):
        """
        Supervised fine-tuning on instruction-following data.
        
        Args:
            instruction_data: List of (instruction, output) pairs
            epochs: Number of training epochs
        """
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=instruction_data,
            peft_config=LoRAConfig(r=64, lora_alpha=128),
            learning_rate=1e-5,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,  # Effective batch size = 16
            max_seq_length=2048,
        )
        trainer.train()
        
    def stage_2_dpo(self, preference_data, epochs=3, beta=0.1):
        """
        Direct Preference Optimization with on-policy data generation.
        
        Args:
            preference_data: Initial preference dataset
            epochs: Number of RL epochs (data regeneration cycles)
            beta: KL penalty coefficient
        """
        for epoch in range(epochs):
            # Generate on-policy data
            new_preferences = self.generate_on_policy_preferences(
                prompts=preference_data['prompts'],
                num_outputs_per_prompt=8
            )
            
            # Train DPO
            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.reference_model,
                train_dataset=new_preferences,
                beta=beta,
                learning_rate=1e-6,
                num_train_epochs=1,  # One pass per RL epoch
                per_device_train_batch_size=4,
            )
            trainer.train()
            
            # Update reference model periodically
            if (epoch + 1) % 3 == 0:
                self.reference_model = copy.deepcopy(self.model)
    
    def stage_3_grpo(self, rl_data, reward_fn, epochs=5):
        """
        Group Relative Policy Optimization for reasoning.
        
        Args:
            rl_data: Dataset with prompts and ground truth
            reward_fn: Function to compute rewards
            epochs: Number of RL training epochs
        """
        trainer = GRPOTrainer(
            model=self.model,
            ref_model=self.reference_model,
            train_dataset=rl_data,
            reward_function=reward_fn,
            group_size=8,  # Sample 8 outputs per prompt
            beta=0.02,     # KL penalty
            learning_rate=1e-6,
            num_train_epochs=epochs,
        )
        trainer.train()
    
    def generate_on_policy_preferences(self, prompts, num_outputs_per_prompt=8):
        """
        Generate preference pairs using current policy.
        
        Returns on-policy (chosen, rejected) pairs.
        """
        preferences = []
        
        for prompt in prompts:
            # Sample multiple outputs
            outputs = []
            for _ in range(num_outputs_per_prompt):
                output = self.model.generate(prompt, temperature=0.7)
                outputs.append(output)
            
            # Score with judge LLM
            scores = [self.judge_model.score(prompt, out) for out in outputs]
            
            # Select chosen (best) and rejected (worst)
            best_idx = np.argmax(scores)
            worst_idx = np.argmin(scores)
            
            # Optional: refine chosen answer with stronger model
            chosen = self.reviser_model.refine(prompt, outputs[best_idx])
            rejected = outputs[worst_idx]
            
            preferences.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
            })
        
        return preferences
```

---

## Part VI: Agentic Reinforcement Learning

### 6.1 Beyond Single-Turn Optimization

**New frontier** (Slide 50): **Agentic RL**

Traditional RL:
```
Prompt → Model → Response → Reward
(single-turn interaction)
```

Agentic RL:
```
Environment → Agent → Action → Environment → Reward
(multi-turn interaction with environment feedback)
```

**Four example environments** (Slide 50):

#### **A. Terminal Environment**
```
Agent interacts with bash terminal:
- Action: Execute shell command
- Observation: Terminal output
- Reward: Task completion (e.g., "create a file containing X")
```

#### **B. Cursor Environment**
```
Agent interacts with code editor:
- Action: Edit file, insert/delete lines
- Observation: Current file state, linter errors
- Reward: Code passes tests, no syntax errors
```

#### **C. OpenClaw Environment**
```
Agent controls robotics simulation:
- Action: Motor commands (joint angles, forces)
- Observation: Sensor data (position, velocity, touch)
- Reward: Task success (e.g., grasp object)
```

#### **D. RLM (Reinforcement Learning from Human Feedback + Actions)**
```
Agent interacts with human evaluator:
- Action: Generate response
- Observation: Human feedback (text, rating)
- Reward: Human satisfaction score
```

### 6.2 Agentic RL Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ Policy Model π_θ (LLM)                                             │
├────────────────────────────────────────────────────────────────────┤
│ Input: Observation from environment                                │
│ Output: Action (text command, code, etc.)                          │
│ Training: GRPO with episode-level rewards                          │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Environment                                                         │
├────────────────────────────────────────────────────────────────────┤
│ Receives action from agent                                         │
│ Executes action (runs code, moves robot, etc.)                    │
│ Returns: New observation + reward                                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Episode Trajectory                                                 │
├────────────────────────────────────────────────────────────────────┤
│ τ = [(o₁, a₁, r₁), (o₂, a₂, r₂), ..., (o_T, a_T, r_T)]           │
│                                                                    │
│ Total reward: R(τ) = Σ r_t (sum of rewards over episode)          │
└────────────────────────────────────────────────────────────────────┘
```

**Key differences from standard GRPO**:
- Multi-turn episodes (not single-turn)
- Environment provides feedback (not just human preferences)
- Sparse rewards (often only at end of episode)
- Exploration challenge (long action sequences)

**Example: Terminal agent**

```python
class TerminalAgent:
    """
    LLM agent that interacts with bash terminal.
    
    Task: Complete user request by executing shell commands.
    """
    
    def __init__(self, policy_model):
        self.model = policy_model
        self.terminal = BashEnvironment()
        
    def run_episode(self, task):
        """
        Execute one episode (complete one task).
        
        Args:
            task: User request (e.g., "Create a file hello.txt with content 'Hello, World!'")
        
        Returns:
            trajectory: List of (observation, action, reward) tuples
        """
        trajectory = []
        observation = self.terminal.reset()  # Initial state
        
        for step in range(max_steps=20):
            # Agent generates action (bash command)
            prompt = f"Task: {task}\nCurrent terminal state:\n{observation}\nNext command:"
            action = self.model.generate(prompt, temperature=0.7)
            
            # Environment executes action
            observation, reward, done = self.terminal.step(action)
            
            trajectory.append((observation, action, reward))
            
            if done:
                break  # Task completed or failed
        
        return trajectory
    
    def compute_episode_reward(self, trajectory, task):
        """
        Compute total reward for episode.
        
        For terminal tasks: Binary success (1.0 if task completed, 0.0 otherwise)
        """
        final_state = self.terminal.get_state()
        
        # Check if task was completed successfully
        if task_completed(task, final_state):
            return 1.0  # Success
        else:
            return 0.0  # Failure
```

---

## Part VII: Production Considerations & Hard Lessons

### 7.1 Hard Lessons from Production Deployment

**Four critical insights** (Slide 51):

#### **1. Data Quality >> Algorithmic Changes**

```
Observation:
- Switching from DPO to GRPO: +2% improvement
- Improving dataset quality: +15% improvement

Lesson: Invest in data infrastructure before algorithms
```

**What matters for data quality**:
- ✅ Accuracy (factually correct, no hallucinations)
- ✅ Diversity (wide coverage of topics, formats, styles)
- ✅ Complexity (non-trivial tasks that require reasoning)
- ✅ Filtering (remove duplicates, low-quality examples)

**Data quality checklist**:
```python
def evaluate_dataset_quality(dataset):
    """
    Automated quality metrics for instruction datasets.
    """
    metrics = {
        'avg_input_length': np.mean([len(x['instruction']) for x in dataset]),
        'avg_output_length': np.mean([len(x['output']) for x in dataset]),
        'unique_prompts': len(set(x['instruction'] for x in dataset)) / len(dataset),
        'format_compliance': count_format_errors(dataset) / len(dataset),
        'diversity_score': compute_diversity(dataset),  # Embedding-based
    }
    return metrics
```

#### **2. Evaluate Everything, All the Time**

```
Without evaluation:
- Train model
- Hope it's better
- Deploy blindly
→ Risk: Model gets worse, users complain

With evaluation:
- Train model
- Run automated benchmarks (MMLU, HumanEval, GSM8K)
- A/B test with real users
- Monitor production metrics (latency, cost, satisfaction)
→ Result: Data-driven decisions, continuous improvement
```

**Evaluation infrastructure**:
```python
class EvaluationPipeline:
    """
    Automated evaluation for post-training.
    """
    
    def __init__(self):
        self.benchmarks = {
            'mmlu': MMLU(),           # Knowledge
            'humaneval': HumanEval(), # Code
            'gsm8k': GSM8K(),         # Math
            'ifeval': IFEval(),       # Instruction-following
        }
        self.judge_model = GPT4Judge()
        
    def evaluate_model(self, model):
        """
        Run full evaluation suite on model.
        
        Returns:
            scores: Dict of benchmark → score
        """
        scores = {}
        
        # Automated benchmarks
        for name, benchmark in self.benchmarks.items():
            score = benchmark.evaluate(model)
            scores[name] = score
            print(f"{name}: {score:.2%}")
        
        # LLM-as-judge evaluation
        judge_score = self.judge_model.evaluate(model, num_samples=500)
        scores['judge'] = judge_score
        
        # Production metrics (if deployed)
        if model.is_deployed():
            scores['prod_latency'] = model.get_avg_latency()
            scores['prod_satisfaction'] = model.get_user_ratings()
        
        return scores
```

#### **3. Infrastructure Is Essential for RL**

```
RL requires:
- ✅ Fast environment interactions (minimize latency)
- ✅ Scalable data generation (1M+ samples)
- ✅ Distributed training (multi-GPU, multi-node)
- ✅ Monitoring (rewards, KL divergence, policy drift)
- ✅ Checkpointing (save/restore model states)

Without infrastructure: RL experiments take weeks
With infrastructure: RL experiments take hours
```

**Infrastructure checklist**:
```yaml
# RL Infrastructure Requirements

Compute:
  - GPUs: 8x A100 (80GB) minimum for 7B model
  - CPU: 64+ cores for data generation
  - RAM: 512GB+ for preprocessing

Storage:
  - Fast SSD: 10TB+ for datasets
  - Blob storage: S3/GCS for checkpoints

Networking:
  - Low-latency interconnect (NVLink, InfiniBand)
  - High bandwidth to storage (10Gb/s+)

Software:
  - Training framework: PyTorch, DeepSpeed
  - RL library: TRL, OpenRLHF
  - Monitoring: Weights & Biases, TensorBoard
  - Orchestration: Kubernetes, Ray

Evaluation:
  - Benchmark suite: EleutherAI LM Evaluation Harness
  - Judge models: GPT-4, Claude (API access)
  - A/B testing: Custom infrastructure
```

#### **4. Small Models Are More Interesting**

```
Observation:
- GPT-4 (1.7T params): Amazing, but expensive and slow
- LFM-1B (1B params): 95% of GPT-4 quality after proper post-training

Why small models matter:
- ✅ Lower inference cost (100x cheaper)
- ✅ Faster inference (10x lower latency)
- ✅ Deployable on-device (mobile, edge)
- ✅ Easier to iterate (faster training)
- ✅ More accessible (consumer GPUs)
```

**Small model success stories**:
```
Examples from 2025-2026:
- LFM-1B-Math: 1B param model, 85% accuracy on MATH benchmark
- Phi-3.5-Mini: 3.8B params, beats Llama-3-8B on many tasks
- Qwen2.5-1.5B: 1.5B params, SOTA for size class
- Mistral-7B-v0.3: 7B params, competes with 30B+ models

Key lesson: Post-training quality matters more than scale
```

### 7.2 Production Deployment Checklist

**End-to-end deployment workflow**:

```
┌────────────────────────────────────────────────────────────────────┐
│ Stage 1: Development                                               │
├────────────────────────────────────────────────────────────────────┤
│ ✅ Train model on dev cluster                                      │
│ ✅ Evaluate on test set (hold-out data)                           │
│ ✅ Run ablation studies (identify critical components)            │
│ ✅ Document hyperparameters, data sources                         │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Stage 2: Pre-Production Testing                                    │
├────────────────────────────────────────────────────────────────────┤
│ ✅ Automated benchmark suite (MMLU, HumanEval, etc.)              │
│ ✅ Human evaluation (Mechanical Turk, internal raters)            │
│ ✅ Safety testing (jailbreaks, toxic prompts, bias)               │
│ ✅ Latency/throughput profiling                                   │
│ ✅ Cost analysis (inference cost per 1M tokens)                   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Stage 3: Production Deployment                                     │
├────────────────────────────────────────────────────────────────────┤
│ ✅ A/B test (5% traffic → new model, 95% → baseline)              │
│ ✅ Monitor metrics (latency, error rate, user satisfaction)       │
│ ✅ Collect feedback (thumbs up/down, free-text comments)          │
│ ✅ Gradual rollout (5% → 25% → 50% → 100%)                        │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Stage 4: Post-Deployment                                           │
├────────────────────────────────────────────────────────────────────┤
│ ✅ Continuous monitoring (dashboards, alerts)                      │
│ ✅ Regular re-evaluation (benchmarks every week)                   │
│ ✅ User feedback analysis (cluster complaints, identify patterns)  │
│ ✅ Iterative improvement (collect failure cases → retrain)        │
└────────────────────────────────────────────────────────────────────┘
```

---

## Part VIII: Conclusions & Future Directions

### 8.1 Summary of Key Insights

**Post-training pipeline** (recap):

```
1. Supervised Fine-Tuning (SFT)
   ├─ Transforms base model → instruction-following model
   ├─ Critical: Dataset quality (accuracy, diversity, complexity)
   ├─ Techniques: Full fine-tuning, LoRA, QLoRA
   └─ Result: Model can follow instructions

2. Preference Alignment (DPO)
   ├─ Optimizes for human preferences (chosen > rejected)
   ├─ Critical: On-policy data generation (avoid policy drift)
   ├─ Techniques: Vanilla DPO, APO, length normalization, rubric scoring
   └─ Result: Model outputs are high-quality, safe, helpful

3. Reinforcement Learning (GRPO)
   ├─ Trains reasoning capabilities with verifiable tasks
   ├─ Critical: Reward function design (accurate, scalable)
   ├─ Techniques: Group normalization, asymmetric clipping, filtering
   └─ Result: Model can solve math, code, science problems

4. Agentic RL (Emerging)
   ├─ Multi-turn interactions with environments
   ├─ Critical: Environment design, episode-level rewards
   ├─ Techniques: Terminal, Cursor, OpenClaw, RLM environments
   └─ Result: Model can complete complex, multi-step tasks
```

**Hard lessons**:
1. **Data quality >> Algorithms**: Clean, diverse datasets matter more than fancy training procedures
2. **Evaluation is essential**: Automated benchmarks + human eval + production metrics
3. **Infrastructure enables iteration**: Fast experiments → faster progress
4. **Small models are underrated**: Proper post-training makes 1B-7B models competitive

### 8.2 Future Research Directions

**Five open problems** (2026 and beyond):

#### **1. Automated Dataset Generation**
Challenge: Collecting high-quality preference data is expensive
Potential solutions:
- Self-play (model generates both chosen and rejected)
- Constitutional AI (define principles, not examples)
- Synthetic data from weak supervision (use weak models to generate, strong models to filter)

#### **2. Multi-Objective Optimization**
Challenge: Models must balance multiple objectives (helpfulness, safety, conciseness)
Potential solutions:
- Pareto optimization (find trade-off frontier)
- Conditional generation (user specifies desired trade-off)
- Meta-learning (learn to balance objectives automatically)

#### **3. Continual Post-Training**
Challenge: Models become outdated as world changes
Potential solutions:
- Incremental updates (fine-tune on recent data)
- Replay buffers (prevent catastrophic forgetting)
- Modular architectures (update specific skills without affecting others)

#### **4. Interpretable Reasoning**
Challenge: Chain-of-thought is powerful but opaque
Potential solutions:
- Structured reasoning (enforce logical steps)
- Proof generation (verify correctness of reasoning)
- Counterfactual analysis (explain why model chose specific path)

#### **5. Sample-Efficient RL**
Challenge: RL requires millions of samples
Potential solutions:
- Model-based RL (learn world model, plan in latent space)
- Curriculum learning (start easy, gradually increase difficulty)
- Transfer learning (reuse knowledge from related tasks)

### 8.3 Final Recommendations for Practitioners

**Getting started with post-training** (action items):

```
Week 1: Infrastructure Setup
├─ Install libraries: transformers, trl, unsloth
├─ Set up GPU cluster or cloud instance (A100 recommended)
├─ Create evaluation pipeline (benchmarks, judge models)
└─ Prepare dataset storage (S3, GCS, or local SSD)

Week 2-3: SFT Experiments
├─ Collect/generate instruction dataset (10K-100K samples)
├─ Train baseline model with LoRA (start small: 1B-7B params)
├─ Evaluate on benchmarks (MMLU, HumanEval, IFEval)
├─ Iterate on dataset quality (filter, deduplicate, augment)
└─ Document best hyperparameters

Week 4-5: DPO Experiments
├─ Create preference dataset (pairwise comparisons)
├─ Implement on-policy data generation pipeline
├─ Train DPO model (beta sweep: 0.1, 0.2, 0.5)
├─ Compare with SFT baseline (automated + human eval)
└─ Analyze failure modes (where did model get worse?)

Week 6-8: RL Experiments (Optional)
├─ Define reward function (math correctness, code execution)
├─ Generate RL dataset (prompts + ground truth)
├─ Train GRPO model (group size sweep: 4, 8, 16)
├─ Monitor rewards, KL divergence, policy drift
└─ Evaluate reasoning capabilities (GSM8K, MATH)

Week 9+: Production Deployment
├─ A/B test best model vs baseline
├─ Monitor production metrics (latency, cost, satisfaction)
├─ Collect user feedback for next iteration
└─ Repeat cycle: improve data → retrain → deploy
```

**Resources**:
- Code: https://github.com/mlabonne/llm-course
- Datasets: https://github.com/mlabonne/llm-datasets
- Paper: LFM2 Technical Report (Liquid AI, arXiv:2511.23404)
- Community: Hugging Face forums, r/LocalLLaMA

---

## Appendix: Technical References

### A.1 Key Papers

**Supervised Fine-Tuning**:
1. Ouyang et al. (2022), "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)
2. Chung et al. (2022), "Scaling Instruction-Finetuned Language Models" (FLAN)
3. Xu et al. (2023), "QA-LoRA: Quantization-Aware Low-Rank Adaptation"

**Preference Optimization**:
4. Rafailov et al. (2023), "Direct Preference Optimization" (DPO)
5. D'Oosterlinck et al. (2024), "Anchored Preference Optimization" (APO)
6. Cui et al. (2023), "UltraFeedback: Boosting Language Models with Scaled AI Feedback"
7. Xu et al. (2024), "The Perfect Blend: Redefining RLHF with Mixture of Judges"

**Reinforcement Learning**:
8. Christiano et al. (2017), "Deep Reinforcement Learning from Human Preferences" (RLHF foundations)
9. Ahmadian et al. (2024), "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback"
10. Shao et al. (2024), "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
11. Yu et al. (2025), "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"

**Production Systems**:
12. Liquid AI (2025), "LFM2 Technical Report" (arXiv:2511.23404)
13. Seyde et al. (2025), "LFM-1B-Math: Can Small Models Be Concise Reasoners?"
14. Liu et al. (2025), "Understanding R1-Zero-Like Training"

### A.2 Open-Source Implementations

**Training Libraries**:
- TRL (Hugging Face): https://github.com/huggingface/trl
- Unsloth: https://github.com/unslothai/unsloth
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF

**Datasets**:
- mlabonne/open-perfectblend: Mixture of Judges dataset
- mlabonne/orpo-dpo-mix-40k: High-quality preference data
- nvidia/OpenMathReasoning: Math reasoning traces
- HuggingFaceH4/ultrafeedback_binarized: Standard preference benchmark

**Evaluation Tools**:
- EleutherAI LM Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness
- OpenAI Evals: https://github.com/openai/evals
- HELM (Stanford): https://crfm.stanford.edu/helm/

### A.3 Benchmark Datasets

| **Benchmark** | **Task** | **Metric** | **SOTA (March 2026)** |
|---------------|----------|------------|-----------------------|
| **MMLU** | Knowledge | 5-shot accuracy | 89.5% (GPT-4) |
| **HumanEval** | Code | pass@1 | 92.3% (GPT-4) |
| **GSM8K** | Math | 8-shot accuracy | 96.8% (o1) |
| **MATH** | Advanced math | 4-shot accuracy | 78.2% (DeepSeek-R1) |
| **IFEval** | Instruction-following | Strict accuracy | 88.3% (GPT-4) |
| **ARC-Challenge** | Science reasoning | 25-shot accuracy | 96.9% (GPT-4) |

---

## Acknowledgments

This technical report synthesizes content from the presentation "Introduction to Post-Training" by **Maxime Labonne** (Liquid AI, Cambridge Computer Laboratory), delivered March 5, 2026. All diagrams, algorithms, and insights are derived from the original presentation slides and cited research papers.

**Special thanks to**:
- Liquid AI research team for LFM2 development
- Hugging Face for TRL library and community support
- Academic researchers advancing post-training methods
- Open-source community for datasets and evaluation tools

---

## Glossary

**SFT**: Supervised Fine-Tuning - Training on instruction-output pairs  
**DPO**: Direct Preference Optimization - Preference learning without reward model  
**GRPO**: Group Relative Policy Optimization - RL with group-normalized advantages  
**LoRA**: Low-Rank Adaptation - Parameter-efficient fine-tuning  
**QLoRA**: Quantized LoRA - 4-bit quantization + LoRA  
**RLHF**: Reinforcement Learning from Human Feedback - General RL paradigm  
**KL Divergence**: Kullback-Leibler divergence - Measure of distribution difference  
**Advantage**: Normalized reward (relative to group average)  
**Policy**: Model that generates actions/responses  
**Reference Model**: Frozen copy of policy (regularization)  
**Reward Function**: Function that scores outputs  
**On-Policy**: Data generated by current policy  
**Off-Policy**: Data generated by different policy  

---

**Report Compiled**: March 2026  
**Version**: 1.0  
**Contact**: maxime-labonne (GitHub), @maximelabonne (Twitter/X)  

**Citation**:
```bibtex
@misc{labonne2026posttraining,
  title={Introduction to Post-Training},
  author={Labonne, Maxime},
  year={2026},
  institution={Liquid AI, Computer Laboratory, Cambridge},
  howpublished={Technical Report},
}
```

---

**Tags**: `#PostTraining` `#LLM` `#SupervisedFineTuning` `#DPO` `#GRPO` `#ReinforcementLearning` `#RLHF` `#LiquidAI` `#MachineLearning` `#DeepLearning`
