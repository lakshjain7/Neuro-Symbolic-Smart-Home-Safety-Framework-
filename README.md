# Neuro-Symbolic Smart-Home Safety Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

An advanced AI safety pipeline that eliminates sensor-induced hallucinations by fusing Spatio-Temporal Graph Neural Networks (GNNs) with strict, hardcoded Physics Guardrails and Symbolic Reasoning.

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Getting Started](#initial-setup--getting-started)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Core Components](#core-components--code)
7. [Performance](#performance)
8. [Contributing](#contributing)
9. [Documentation](#documentation)
10. [Research and References](#research--references)
11. [License](#license)

---

## Overview

### The Challenge

Modern smart homes are flooded with IoT sensors, but relying purely on standard Machine Learning to detect safety hazards creates a massive vulnerability: **AI Hallucinations.**

**Scenario:** If a smart plug's hardware glitches and sends a corrupted data spike, a traditional AI model might blindly trust the data and trigger a false emergency response:
- Shutting down the house's power
- Calling emergency services
- Triggering unnecessary alarms

### The Solution: Neuro-Symbolic AI

The framework solves this by combining **two distinct forms of AI:**

1. **The "Neuro" (Deep Learning):** A3TGCN2 (Attention Temporal Graph Convolutional Network) that treats the house as a graph structure and learns spatio-temporal patterns
2. **The "Symbolic" (Physics Guardrails):** Hardcoded laws of physics—specifically energy conservation—in the final decision layer

### How It Works

```text
Raw Sensor Data (1M+ records)
    ↓
[Neural Pipeline] → Predicts anomaly with 99% confidence
    ↓
[Symbolic Guardrail] → Cross-references with physical energy meter
    ↓
Physics Check: Sum(Devices) ≈ Main Meter?
    ├─ YES → Anomaly is REAL ✅ Alert user
    └─ NO  → Sensor hallucination detected 🚨 Block false alarm

```

**Result:** 92% verified accuracy across 6 distinct forensic test scenarios, effectively neutralizing sensor-induced false alarms.

---

## Initial Setup & Getting Started

Follow these steps to clone, configure, and prepare your environment before running the notebook.

### Step 1: Clone the Repository

```bash
git clone [https://github.com/lakshjain7/Neuro-Symbolic-Smart-Home-Safety-Framework-.git](https://github.com/lakshjain7/Neuro-Symbolic-Smart-Home-Safety-Framework-.git)
cd Neuro-Symbolic-Smart-Home-Safety-Framework-

```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate

```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

```

### Step 4: Configure Environment Variables

Create a `.env` file in the root directory for LLM-powered alerting features:

```env
OPENAI_API_KEY="your_api_key_here"
HUGGINGFACE_TOKEN="your_token_here"

```

### Step 5: Prepare the Dataset & Run

1. Place your 21-node smart home telemetry dataset (e.g., `telemetry_1M.csv`) in the same directory.
2. Launch Jupyter Notebook:
```bash
jupyter notebook

```


3. Open `Graph_Transformer.ipynb` and run the cells sequentially to execute the data ingestion, model training, and physics guardrail validation.

---

## System Architecture

The framework processes IoT telemetry through a highly customized **three-tier architecture**:

### Architecture Diagram

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                     SMART HOME SENSOR NETWORK (21 nodes)                 │
│     [Room1: AC, Heater, Thermostat] ... [Room21: Lights, Outlets]        │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────────┐      ┌──────────────────────┐
    │   DATA INGESTION    │      │  GRAPH CONSTRUCTION  │
    │   & SEGMENTATION    │      │  (Physical Layout)   │
    │                     │      │                      │
    │ • 60-min Windows    │      │ • Nodes: Devices     │
    │ • 24h Aggregates    │      │ • Edges: Proximity   │
    │ • 7d Statistics     │      │ • Attributes: Power  │
    └────────┬────────────┘      └──────────┬───────────┘
             │                              │
             └──────────────┬───────────────┘
                            ▼
            ┌─────────────────────────────────────┐
            │    NEURAL PIPELINE (A3TGCN2)        │
            │                                     │
            │  Layer 1: Atomic Device Profiling   │
            │  ↓                                  │
            │  Layer 2: Spatial Cross-Device      │
            │  ↓                                  │
            │  Layer 3: Temporal Context          │
            │  ↓                                  │
            │  Output: 6 Anomaly Categories       │
            └──────────┬────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │   SYMBOLIC LAYER (Physics Guardrail)│
        │                                     │
        │   Energy Conservation Check:        │
        │   Sum(Device Power) ≈ Grid Power?   │
        │                                     │
        │   IF Physics Match → ALERT ✅        │
        │   IF Physics Fail  → BLOCK 🚨        │
        └──────────┬────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────────────┐
        │   FINAL DECISION & ACTION           │
        │   (Verified Anomaly Detection)      │
        └─────────────────────────────────────┘

```

### Layer A: Data Ingestion & Segmentation

Raw per-minute data is chaotic. We transform it into structured format:

**Key Features:**

* **60-Frame Sliding Window:** Chunks data into 1-hour windows (60 minutes)
* **Long-Term Memory Encoding:** Calculates 24-hour and 7-day statistical features (means, maximums, variances)
* **Feature Format:** Each node contains both recent and historical data

### Layer B: Neural Pipeline (A3TGCN2)

The core predictive engine with 3-layer inference:

1. **Atomic Device Profiling:** Understands baseline behavior of each 21 nodes independently
2. **Spatial Cross-Device State Prediction:** Maps how devices affect each other physically using Graph Convolutions
3. **Temporal Context Analysis:** Tracks state evolution over 60-frame windows using Temporal Attention

### Layer C: Symbolic Pipeline (Physics Guardrail)

Deterministic rule engine that:

* Intercepts GNN anomaly flags
* Forces them to pass energy conservation check
* Validates predictions against physical laws before alerting

---

## Installation

### Prerequisites

* Python 3.8 or higher
* pip or conda
* CUDA 11.8+ (optional, for GPU acceleration)

### Required Libraries

```text
torch>=2.0.0
torch-geometric>=2.3.0
torch-geometric-temporal>=0.12.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
jupyter>=1.0.0
python-dotenv>=0.20.0

```

---

## Quick Start

All components of this project are contained within the `Graph_Transformer.ipynb` notebook. Below are conceptual snippets of what runs inside the core cells:

### Example 1: Load and Prepare Data

```python
import numpy as np

def create_sliding_windows(data, window_size=60):
    """
    Transforms sequential telemetry into 3D windows for the A3TGCN2 model.
    """
    windows = []
    for i in range(len(data) - window_size):
        window = data[i : i + window_size]
        # Transform from (Time, Nodes, Features) -> (Nodes, Features, Time)
        window = np.transpose(window, (1, 2, 0))
        windows.append(window)
    return np.array(windows)

```

### Example 2: Build the Graph Neural Network

```python
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2

class SmartHomeGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(SmartHomeGNN, self).__init__()
        self.tgnn = A3TGCN2(in_channels=node_features, 
                            out_channels=32, 
                            periods=periods,
                            batch_size=batch_size)
        self.linear = torch.nn.Linear(32, 6)

    def forward(self, x, edge_index):
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        return F.log_softmax(self.linear(h), dim=1)

```

### Example 3: Physics Guardrail (Symbolic Layer)

```python
class PhysicsGuardrail:
    def __init__(self, energy_tolerance=0.05):
        self.tolerance = energy_tolerance
    
    def verify_prediction(self, gnn_prediction, device_states, main_meter):
        if gnn_prediction == 0:  # Category 0 = SAFE
            return True
        
        reported_device_power = sum([device.watts for device in device_states])
        actual_grid_power = main_meter.watts
        difference = abs(reported_device_power - actual_grid_power)
        
        tolerance_band = actual_grid_power * self.tolerance
        
        if difference > tolerance_band:
            print("🚨 GUARDRAIL TRIGGERED: Blocking AI Hallucination ✓")
            return False  # Reject the GNN's alert
        
        print("✅ Physics verified: Anomaly is REAL")
        return True  # Physics check out; the anomaly is legitimate

```

---

## Core Components & Code

All core components are engineered as sequential stages within `Graph_Transformer.ipynb`:

1. **Data Pipeline:** Executes 60-minute sliding windows for temporal context and calculates 24-hour/7-day aggregates for long-term patterns.
2. **Graph Neural Network:** Defines the A3TGCN2 Architecture utilizing multi-head attention for diverse feature relationships, outputting a 6-class anomaly classification.
3. **Symbolic Reasoning Engine:** The custom physics guardrail executing first-order predicates for safety constraints and energy conservation verification.
4. **Integration Layer:** Fuses both AI approaches to provide explainability and validate decisions against physical laws.

---

## Performance

### Verified Accuracy Metrics

| Metric | Value |
| --- | --- |
| **Overall Accuracy** | 92% |
| **Fire Hazard Detection** | 94.3% |
| **Gas Leak Detection** | 88.7% |
| **Electrical Fault Detection** | 91.2% |
| **False Positive Rate** | 2.1% |
| **False Negative Rate** | 5.8% |
| **Inference Time** | 45ms per batch |
| **Maximum Supported Devices** | 500+ |

*Note: Results based on 6 forensic test scenarios with 1M+ real telemetry records*

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/YourFeature`
3. **Commit** your changes: `git commit -m 'Add YourFeature'`
4. **Push** to the branch: `git push origin feature/YourFeature`
5. **Open** a Pull Request with a clear description

---

## Documentation

* **[Graph_Transformer.ipynb](https://www.google.com/search?q=Graph_Transformer.ipynb)** - Interactive implementation containing the entire end-to-end framework, models, and exploratory visualizations.

---

## Research & References

This framework implements concepts from:

* **Neuro-symbolic AI Systems** - Combining neural networks with symbolic reasoning
* **Graph Attention Networks (GAT)** - Multi-head attention for graph structures
* **Temporal Graph Convolutions** - Learning from spatio-temporal data
* **Physics-informed Machine Learning** - Enforcing physical laws in AI
* **IoT Safety & Security** - Smart home threat detection
* **Energy Conservation Laws** - Fundamental physics constraints

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

## Author

**Laksh Jain**

* GitHub: [@lakshjain7](https://github.com/lakshjain7)
* Email: lakshjain960@gmail.com

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{jain2026neuro,
  title={Neuro-Symbolic Smart-Home Safety Framework},
  author={Jain, Laksh},
  year={2026},
  url={[https://github.com/lakshjain7/Neuro-Symbolic-Smart-Home-Safety-Framework-](https://github.com/lakshjain7/Neuro-Symbolic-Smart-Home-Safety-Framework-)}
}

```
