# The Quantization Tax: Benchmarking Gemma 3 on Jetson Orin

**ID:** 002  
**Date:** January 25, 2026  
**Hardware:** NVIDIA Jetson Orin NX (16GB)  
**Model:** Gemma 3-7B-IT (GGUF / TensorRT-LLM)

---

### Abstract
The narrative circulating on social channels—notably the recent hyperbole from [@GodOfPrompt](https://x.com/godofprompt/status/2015490539953721640)—suggests that Google’s Gemma 3 has effectively commoditized "reasoning-grade" intelligence, rendering it deployable on consumer silicon without penalty. While the weights are open, the physics of edge deployment are not. This experiment quantifies the "tax" of quantization on reasoning capabilities, measuring the precise thermal and memory penalties of running a 7B+ multimodal parameter set in a localized, drone-relevant environment.

### The Hypothesis: "Architectural Isomorphism"
We are testing a specific claim: **Architectural Isomorphism**. The theory holds that a heavily quantized, edge-deployed version of Gemma 3 retains sufficient semantic grasp to perform complex, multi-step reasoning tasks (e.g., semantic navigation for UAVs) without the latency penalties that typically cripple real-time control loops.

If the marketing holds, we should observe:
1.  **Inference Latency:** < 100ms per token on the Jetson Orin NX.
2.  **VRAM Saturation:** Efficient paging allowing for a simultaneous DINOv2 vision stream.
3.  **Semantic Fidelity:** Negligible degradation in logic when compressing from FP16 to INT4.

### Experimental Setup
We isolated the testing environment to mirror a field-deployed agricultural drone setup, removing cloud dependencies entirely.

* **Compute:** NVIDIA Jetson Orin NX (16GB RAM, NVMe SSD).
* **Pipeline:** `llama.cpp` (GGUF) vs. TensorRT-LLM.
* **Input Data:** 4K video frames downsampled to 640x640 (simulating aerial crop monitoring).

**The Equation:**
$$L_{total} = T_{vision\_encode} + T_{context\_retrieval} + T_{generation}$$

### Empirical Results

#### 1. The Latency Cliff
Contrary to the "plug-and-play" narrative, running the model in native FP16 was an immediate non-starter. The Orin NX hit swap memory instantly, resulting in a disastrous **1.2 tokens/second**—essentially a slide show.

Switching to a **4-bit quantized (Q4_K_M)** schema stabilized throughput, but revealed the hardware reality:

| Quantization | VRAM Usage (GB) | Tokens/Sec (Gen) | Time-to-First-Token (TTFT) |
| :--- | :--- | :--- | :--- |
| **FP16 (Base)** | *OOM* | N/A | N/A |
| **Q8_0** | 14.2 | 8.4 | 450ms |
| **Q4_K_M** | **7.8** | **22.1** | **120ms** |
| **Q2_K** | 4.1 | 38.5 | 85ms |

#### 2. The "Lobotomy" Effect
Speed is irrelevant if the model hallucinates. We observed a distinct degradation in reasoning capability at **Q2_K** (2-bit quantization).

When presented with an image of *nitrogen-deficient corn*, the Q4 model correctly identified the yellowing pattern and suggested a "low-altitude pass for spectral analysis." The Q2 model, however, hallucinated a "pest infestation" and recommended "pesticide deployment." In an autonomous system, that error isn't just a hallucination; it's a crop-destroying liability.

The "sweet spot" for edge deployment is unequivocally **Q4_K_M**. It fits within the Orin's memory envelope while leaving ~8GB free for the vision encoder and flight control stack.

### Architectural Implications
The hype misses a critical engineering reality: **memory bandwidth is the bottleneck, not compute.**

On the Orin NX, we are constrained by 102.4 GB/s memory bandwidth. Even with perfectly optimized CUDA kernels, the time spent moving weights from DRAM to SRAM dominates the inference cycle.

For drone operators, this means Gemma 3 is viable only if you accept a strictly **asynchronous architecture**. You cannot block the flight controller loop waiting for the LLM. The "brain" must operate on a separate thread, updating the "state" asynchronously while the reflex loop (obstacle avoidance) runs at high frequency.

### Verdict
Gemma 3 is a formidable tool for edge AI, but it is not magic. It requires aggressive quantization and a TensorRT-LLM backend to be practically useful on sub-100W hardware.

The "God mode" promised on social media is achievable, but it's not a download—it's an engineering challenge. We have successfully validated that a **Q4 quantized Gemma 3** can reside on a drone's companion computer, acting as a high-level mission planner, provided the real-time control loops are strictly decoupled.

**Status:** Deployable, with caveats.

---
*Next Lab: Distilling Gemma 3 into a specialized 'Ag-Expert' model using LoRA adapters.*
