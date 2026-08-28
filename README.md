<h1 align="center">
  <b>RoboTwin</b> Bimanual Robotic Manipulation Platform<br>
</h1>
<h2 align="center">A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation</h2>

## 💡 Core Engineering Contributions

This repository contains the source code for the RoboTwin simulation platform and policy deployment. My core engineering and research implementations in this project include:

* **OpenVLA-OFT Architecture Optimization:** Reproduced and optimized the OpenVLA-OFT architecture specifically for the RoboTwin simulation environment.
* **End-to-End Dual-Arm Control:** Successfully achieved end-to-end continuous control for complex bimanual robotic manipulation tasks.
* **Distributed Fine-Tuning:** Implemented Fully Sharded Data Parallel (FSDP) and Low-Rank Adaptation (LoRA) for highly efficient distributed fine-tuning across 4 GPUs.
* **Data Pipeline Optimization:** Streamlined the entire data pipeline, including simulation setup, expert trajectory collection, RLDS format conversion, and action chunking optimizations for large-scale training.

---

## 📚 Overview

| Branch Name | Link |
|-------------|------|
| 2.0 Version Branch | `main` (latest) |
| IsaacLab-Arena Branch | `IsaacLab-Arena` |
| RLinf Branch | `RLinf_support` |
| 1.0 Version Branch | `RoboTwin-1.0` |

## 🛠️ Installation

1. Create a conda environment and install dependencies:
```bash
conda create -n robotwin python=3.10
conda activate robotwin
pip install -r requirements.txt

```

*(Please refer to the official documentation for detailed Isaac Sim and RL environment configurations. Installation typically takes about 20 minutes.)*

## 🤷‍♂️ Tasks Information

The platform supports over 50 bimanual manipulation tasks, featuring highly configurable parameters and strong domain randomization capabilities (e.g., varying lighting, textures, camera poses, and object physics).

## 🧑🏻‍💻 Usage

### Data Collection

We provide large-scale pre-collected trajectories. However, due to the high configurability and diversity of task and embodiment setups, it is highly recommended to perform data collection tailored to your specific policies.

**1. Task Running and Data Collection**
Running the following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.

```bash
bash collect_data.sh ${task_name} ${task_config}${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0

```

**2. Modify Task Config**
Users can customize task configurations in the `configs/` directory to adjust difficulty, randomization parameters, and dual-arm coordination constraints.

## 🚴‍♂️ Policy Baselines

The platform officially supports and evaluates multiple state-of-the-art policy baselines:

* **OpenVLA-oft** *(Core optimized implementation)*
* **DP** (Diffusion Policy)
* **ACT** (Action Chunking with Transformers)
* **DP3** (3D Diffusion Policy)
* **RDT** (Robotic Diffusion Transformer)
* **PI0**
* **TinyVLA** / **DexVLA**
* **LLaVA-VLA**
* **GO-1**

*⏰ TODO: G3Flow, HybridVLA, SmolVLA, AVR, UniVLA*

## 🏄‍♂️ Experiment & LeaderBoard

The RoboTwin Platform is designed to explore the following research topics:

1. Single-task fine-tuning capability
2. Visual robustness under extreme domain randomization
3. Language diversity robustness (language conditioning)
4. Multi-task capability and interference
5. Cross-embodiment zero-shot transfer performance

## 🏷️ License

This repository is released under the MIT license. See `LICENSE` for additional details.
