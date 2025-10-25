````markdown
# nonRecognition - Adversarial Fuzzer

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-research-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey.svg)]()

> A research-focused adversarial testing framework for evaluating facial recognition systems through systematic pattern generation and evolutionary optimization.

<p align="center">
  <img src="./images/nonrecognition_banner.png" alt="Adversarial Fabrics on Kickstarter" style="max-width:100%;height:auto;border-radius:8px;">
</p>

## Quick Links

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Pattern Library](#pattern-library)
- [Research Results](#research-results)

---

## Overview

![Research](https://img.shields.io/badge/type-research-blue.svg)
![AI Security](https://img.shields.io/badge/focus-AI%20security-red.svg)
![Privacy](https://img.shields.io/badge/goal-privacy-green.svg)

**nonRecognition** is a high-performance adversarial fuzzing framework designed to scientifically evaluate and document vulnerabilities in modern facial recognition systems. The project combines advanced pattern generation, genetic algorithms, and ensemble model testing to create reproducible, testable adversarial textiles.

### Mission

Join the first scientific effort to build reproducible, testable adversarial textiles and open source software that gives privacy back to people.

### Key Capabilities

- Hardware-agnostic HPC pattern generation (CUDA, Metal, CPU)
- Ensemble testing against multiple state-of-the-art models
- Genetic algorithm for evolved pattern optimization
- Landmark-aware surgical attacks
- Research-grade reporting and analytics

---

## Architecture Overview

![Architecture](https://img.shields.io/badge/architecture-modular-brightgreen.svg)
![Scalability](https://img.shields.io/badge/scalability-high-success.svg)

```mermaid
graph TB
    subgraph "Input Layer"
        A[Base Images]
        B[Pattern Library]
        C[Priority Recipes]
    end
    
    subgraph "Pattern Generation Engine"
        D[Hardware Detection]
        E[CUDA/cuPy]
        F[Metal/mlx]
        G[CPU/numba]
        H[numpy fallback]
        
        D -->|GPU Available| E
        D -->|Apple Silicon| F
        D -->|CPU Only| G
        D -->|Fallback| H
    end
    
    subgraph "Fuzzing Core"
        I[Pattern Compositor]
        J[Multiprocess Worker Pool]
        K[Image Overlay System]
    end
    
    subgraph "Model Ensemble"
        L[InsightFace buffalo_l]
        M[InsightFace buffalo_s]
        N[YOLOv8n]
        O[Anomaly Detector]
    end
    
    subgraph "Evolution Engine"
        P[Genetic Algorithm]
        Q[Mutation Engine]
        R[Crossover Engine]
        S[Priority Queue]
    end
    
    subgraph "Output Layer"
        T[Anomaly Logs]
        U[Recipe Database]
        V[Statistical Reports]
        W[Print-Ready Files]
    end
    
    A --> I
    B --> I
    C --> I
    
    E --> I
    F --> I
    G --> I
    H --> I
    
    I --> J
    J --> K
    K --> L
    K --> M
    K --> N
    
    L --> O
    M --> O
    N --> O
    
    O -->|Anomaly Found| P
    P --> Q
    P --> R
    Q --> S
    R --> S
    S --> C
    
    O --> T
    S --> U
    T --> V
    U --> W
```

---

## System Workflow

![Workflow](https://img.shields.io/badge/workflow-automated-blueviolet.svg)
![Pipeline](https://img.shields.io/badge/pipeline-continuous-informational.svg)

```mermaid
sequenceDiagram
    participant User
    participant Fuzzer
    participant PatternEngine
    participant ModelEnsemble
    participant Evolution
    participant Reports
    
    User->>Fuzzer: Start Fuzzing Campaign
    Fuzzer->>PatternEngine: Initialize Hardware Backend
    PatternEngine-->>Fuzzer: Backend Ready (CUDA/Metal/CPU)
    
    loop Each Epoch
        Fuzzer->>PatternEngine: Generate Pattern Batch
        PatternEngine->>PatternEngine: Apply Genetic Mutations
        PatternEngine-->>Fuzzer: Pattern Candidates
        
        par Parallel Testing
            Fuzzer->>ModelEnsemble: Test vs InsightFace-L
            Fuzzer->>ModelEnsemble: Test vs InsightFace-S
            Fuzzer->>ModelEnsemble: Test vs YOLOv8n
        end
        
        ModelEnsemble-->>Fuzzer: Detection Results
        
        alt Anomaly Detected
            Fuzzer->>Evolution: Save Recipe to Priority Queue
            Evolution->>Evolution: Mark for Next Epoch
            Fuzzer->>Reports: Log Anomaly Details
        else Normal Detection
            Fuzzer->>Reports: Log Standard Result
        end
    end
    
    User->>Fuzzer: Stop Campaign
    Fuzzer->>Reports: Generate Statistical Analysis
    Reports-->>User: Performance & Vulnerability Reports
```

---

## Fuzzing Process Flow

![Process](https://img.shields.io/badge/process-iterative-orange.svg)
![Stateful](https://img.shields.io/badge/state-resumable-success.svg)

```mermaid
stateDiagram-v2
    [*] --> Initialization
    
    Initialization --> LoadBaseImages
    LoadBaseImages --> LoadPriorityRecipes
    LoadPriorityRecipes --> DetectHardware
    DetectHardware --> InitializeModels
    
    InitializeModels --> EpochStart
    
    state EpochStart {
        [*] --> SelectPattern
        SelectPattern --> ApplyMutations: From Priority Queue
        SelectPattern --> RandomGeneration: New Pattern
        
        ApplyMutations --> GenerateCandidate
        RandomGeneration --> GenerateCandidate
        
        GenerateCandidate --> CompositeOverlay
    }
    
    EpochStart --> ParallelTesting
    
    state ParallelTesting {
        [*] --> WorkerPool
        WorkerPool --> TestInsightFaceL
        WorkerPool --> TestInsightFaceS
        WorkerPool --> TestYOLOv8n
        
        TestInsightFaceL --> AggregateResults
        TestInsightFaceS --> AggregateResults
        TestYOLOv8n --> AggregateResults
    }
    
    ParallelTesting --> EvaluateResults
    
    state EvaluateResults {
        [*] --> CheckAnomalies
        CheckAnomalies --> AnomalyDetected: Failure Found
        CheckAnomalies --> NormalDetection: All Pass
        
        AnomalyDetected --> SaveRecipe
        SaveRecipe --> AddToPriorityQueue
        AddToPriorityQueue --> LogAnomaly
        
        NormalDetection --> LogStandard
    }
    
    EvaluateResults --> CheckEpochComplete
    CheckEpochComplete --> EpochStart: Continue
    CheckEpochComplete --> GenerateReports: Complete
    
    GenerateReports --> [*]
```

---

## Pattern Generation Pipeline

![Patterns](https://img.shields.io/badge/patterns-40+-purple.svg)
![Generation](https://img.shields.io/badge/generation-GPU%20accelerated-brightgreen.svg)

```mermaid
flowchart LR
    subgraph "Pattern Selection"
        A[Pattern Library<br/>40+ Generators]
        B{Selection<br/>Strategy}
        C[Random<br/>Selection]
        D[Priority<br/>Queue]
    end
    
    subgraph "Genetic Operations"
        E[Parent Recipe 1]
        F[Parent Recipe 2]
        G{Operation}
        H[Mutation]
        I[Crossover]
        J[Layer Addition]
        K[Layer Removal]
    end
    
    subgraph "Pattern Composition"
        L[Base Layer]
        M[Layer Stack]
        N[Landmark Detection]
        O[Surgical Placement]
        P[Final Composite]
    end
    
    subgraph "Hardware Acceleration"
        Q{Backend}
        R[CUDA Processing]
        S[Metal Processing]
        T[CPU Processing]
    end
    
    A --> B
    B --> C
    B --> D
    
    C --> L
    D --> E
    D --> F
    
    E --> G
    F --> G
    G --> H
    G --> I
    H --> J
    H --> K
    I --> M
    
    L --> M
    M --> N
    N --> O
    O --> P
    
    P --> Q
    Q --> R
    Q --> S
    Q --> T
    
    R --> U[Pattern Output]
    S --> U
    T --> U
```

---

## Installation

![Installation](https://img.shields.io/badge/installation-automated-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-managed-success.svg)

### Prerequisites

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-optional-green.svg)
![Metal](https://img.shields.io/badge/Metal-optional-lightgrey.svg)

- Python 3.8 or higher
- CUDA-capable GPU (optional, for accelerated processing)
- Apple Silicon Mac (optional, for Metal acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/hevnsnt/norecognition.git
cd norecognition

# Run the installer
bash installer.sh
```

The installer will:
1. Create a virtual environment with the latest Python version
2. Install all dependencies from requirements.txt
3. Provide instructions for launching the fuzzer

### Manual Installation

```bash
# Create virtual environment
python3 -m venv norecognition_env
source norecognition_env/bin/activate  # On Windows: norecognition_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

![Usage](https://img.shields.io/badge/usage-CLI-blue.svg)
![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen.svg)

### Basic Usage

```bash
# Activate the virtual environment
source norecognition_env/bin/activate

# Start a fuzzing campaign with default settings
python fuzzer.py --epochs 10 --samples 1000

# Resume a previous campaign
python fuzzer.py --resume

# Generate reports from fuzzing data
python plot_reports.py
```

### Advanced Options

```bash
# Use specific GPU device
python fuzzer.py --device cuda:0 --epochs 20

# Specify custom base images directory
python fuzzer.py --images ./custom_images --epochs 5

# Set custom output directory
python fuzzer.py --output ./results --epochs 15
```

---

## How It Works

![Version](https://img.shields.io/badge/version-0.4-blue.svg)
![TUI](https://img.shields.io/badge/interface-TUI-green.svg)

### The Fuzzer in Action

<p align="center">
  <img src="./images/v04_fuzzer_working.gif" alt="Fuzzer Working" width="1024">
</p>

As of v0.4, the fuzzer features a full-screen Terminal User Interface (TUI) for real-time analysis with:

- Live-updating overall statistics
- Epoch progress tracking
- Detailed anomaly log
- In-terminal image preview of recent anomalies
- Mutation and evolution progress indicators

### Core Features

#### Hardware-Agnostic HPC

![CUDA](https://img.shields.io/badge/CUDA-supported-green.svg)
![Metal](https://img.shields.io/badge/Metal-supported-lightgrey.svg)
![CPU](https://img.shields.io/badge/CPU-supported-blue.svg)

The fuzzer's pattern engine auto-detects the best available compute backend:

- **NVIDIA CUDA** via cuPy
- **Apple Silicon Metal** via mlx
- **JIT-Compiled CPU** via numba
- **Standard CPU** via numpy

This enables massive parallel throughput across any modern machine.

#### Ensemble Model Testing

![Models](https://img.shields.io/badge/models-3-orange.svg)
![Testing](https://img.shields.io/badge/testing-ensemble-red.svg)

Every pattern is validated against multiple state-of-the-art systems:

- **InsightFace (buffalo_l)** - Large, high-accuracy face detector
- **InsightFace (buffalo_s)** - Smaller, faster face detector
- **YOLOv8n** - Modern, real-time object detector

An anomaly is registered when patterns fool models in significant ways.

#### Genetic Algorithm Evolution

![Algorithm](https://img.shields.io/badge/algorithm-genetic-purple.svg)
![Learning](https://img.shields.io/badge/learning-evolutionary-blueviolet.svg)

```mermaid
graph LR
    A[Initial Patterns] --> B[Test Against Models]
    B --> C{Anomaly<br/>Found?}
    C -->|Yes| D[Save to Priority Queue]
    C -->|No| E[Discard]
    D --> F[Next Epoch]
    F --> G[Select Parent Recipes]
    G --> H[Mutation]
    G --> I[Crossover]
    H --> J[New Generation]
    I --> J
    J --> B
```

The fuzzer learns by:

1. Saving successful pattern recipes to PRIORITY_TESTS
2. Using genetic algorithms in subsequent epochs:
   - **Mutation**: Randomly modify layers
   - **Crossover**: Combine two successful recipes
3. Evolving increasingly complex and effective patterns

#### Landmark-Aware Surgical Attacks

![Attacks](https://img.shields.io/badge/attacks-surgical-red.svg)
![Precision](https://img.shields.io/badge/precision-landmark%20based-orange.svg)

```mermaid
graph TB
    A[Detect Facial Landmarks] --> B{Attack Type}
    
    B --> C[adversarial_patch]
    B --> D[landmark_noise]
    B --> E[dazzle_camouflage]
    B --> F[swapped_landmarks]
    B --> G[saliency_eye_attack]
    
    C --> H[Place High-Contrast<br/>Sticker on Key Feature]
    D --> I[Apply Noise Only<br/>to Eyes/Nose/Mouth]
    E --> J[Draw Disruptive Lines<br/>Through Features]
    F --> K[Paste Mouth<br/>Over Eye Region]
    G --> L[Stamp Multiple Eyes<br/>to Confuse NMS]
    
    H --> M[Composite Pattern]
    I --> M
    J --> M
    K --> M
    L --> M
```

---

## Pattern Library

![Patterns](https://img.shields.io/badge/total%20patterns-40+-success.svg)
![Categories](https://img.shields.io/badge/categories-7-blue.svg)
![Extensible](https://img.shields.io/badge/extensible-yes-brightgreen.svg)

The fuzzer includes 40+ pattern generators organized by category:

### Geometric & Noise
![Type](https://img.shields.io/badge/type-geometric-yellow.svg)

- `simple_shapes`, `fractal_noise`, `perlin_noise`, `hf_noise`
- `checkerboard`, `gradient`, `op_art_chevrons`, `tiled_logo`, `fft_noise`

### Feature-Based & Saliency
![Type](https://img.shields.io/badge/type-feature%20based-orange.svg)

- `feature_collage`, `saliency_eye_attack`, `recursive_face_tile`
- `ascii_face`, `animal_print`, `trypophobia`, `pop_art_collage`

### Surgical & Landmark-Based
![Type](https://img.shields.io/badge/type-surgical-red.svg)

- `landmark_noise`, `swapped_landmarks`, `adversarial_patch`

### Camouflage & Texture
![Type](https://img.shields.io/badge/type-camouflage-green.svg)

- `camouflage`, `repeating_texture_object`, `warped_face`

### Structural & Dazzle
![Type](https://img.shields.io/badge/type-dazzle-blue.svg)

- `hyperface_like`, `dazzle_camouflage`, `interference_lines`, `3d_wireframe`

### Glitch & Sensor Attacks
![Type](https://img.shields.io/badge/type-glitch-purple.svg)

- `vortex`, `optical_flow`, `photonegative_patch`
- `colorspace_jitter`, `selective_blur`, `pixel_sort_glitch`

### Other
![Type](https://img.shields.io/badge/type-miscellaneous-lightgrey.svg)

- `random_text`, `qr_code`, `ir_led_attack`, `blackout_patches`

### Pattern Examples

<table align="center" style="border-collapse:collapse; border-spacing:0; padding:0; margin:0;">
  <tr>
    <td style="padding:0; margin:0;"><a href="./images/pattern_samples/45_shirt_gaiter_feature_collage_seed8374138_sample180500.jpg"><img src="./images/pattern_samples/45_shirt_gaiter_feature_collage_seed8374138_sample180500.jpg" width="200" style="display:block; margin:0; padding:0;"></a></td>
    <td style="padding:0; margin:0;"><a href="./images/pattern_samples/Man_Wearing_Gaiter_3d_wireframe+op_art_chevrons+repeating_texture_object_seed2984227_sample297500.jpg"><img src="./images/pattern_samples/Man_Wearing_Gaiter_3d_wireframe+op_art_chevrons+repeating_texture_object_seed2984227_sample297500.jpg" width="200" style="display:block; margin:0; padding:0;"></a></td>
    <td style="padding:0; margin:0;"><a href="./images/pattern_samples/facemask_1_qr_code+hyperface_like_seed948622_sample197000.jpg"><img src="./images/pattern_samples/facemask_1_qr_code+hyperface_like_seed948622_sample197000.jpg" width="200" style="display:block; margin:0; padding:0;"></a></td>
  </tr>
  <tr>
    <td style="padding:0; margin:0;"><a href="./images/pattern_samples/Woman_Wearing_Hoodie_feature_collage+3d_wireframe+vortex_seed8663387_sample303500.jpg"><img src="./images/pattern_samples/Woman_Wearing_Hoodie_feature_collage+3d_wireframe+vortex_seed8663387_sample303500.jpg" width="200" style="display:block; margin:0; padding:0;"></a></td>
    <td style="padding:0; margin:0;"><a href="./images/pattern_samples/full_body_dress_6_feature_collage_seed1358874_sample290000.jpg"><img src="./images/pattern_samples/full_body_dress_6_feature_collage_seed1358874_sample290000.jpg" width="200" style="display:block; margin:0; padding:0;"></a></td>
    <td style="padding:0; margin:0;"><a href="./images/pattern_samples/Woman_Wearing_Scarf_3d_wireframe+simple_shapes_seed4381582_sample280000.jpg"><img src="./images/pattern_samples/Woman_Wearing_Scarf_3d_wireframe+simple_shapes_seed4381582_sample280000.jpg" width="200" style="display:block; margin:0; padding:0;"></a></td>
  </tr>
  <tr>
    <td style="padding:0; margin:0;"><a href="./images/pattern_samples/full_body_shawl_6_perlin_noise+repeating_texture_object+3d_wireframe_seed3559762_sample370500.jpg"><img src="./images/pattern_samples/full_body_shawl_6_perlin_noise+repeating_texture_object+3d_wireframe_seed3559762_sample370500.jpg" width="200" style="display:block; margin:0; padding:0;"></a></td>
    <td style="padding:0; margin:0;"><a href="./images/pattern_samples/Woman_Wearing_Shawl_dazzle_camouflage+pop_art_collage+blackout_patches_seed9165740_sample173500.jpg"><img src="./images/pattern_samples/Woman_Wearing_Shawl_dazzle_camouflage+pop_art_collage+blackout_patches_seed9165740_sample173500.jpg" width="200" style="display:block; margin:0; padding:0;"></a></td>
    <td style="padding:0; margin:0;"><a href="./images/pattern_samples/Man_Hat_Hide_Face_hyperface_like+landmark_noise+checkerboard_seed9167792_sample240500.jpg"><img src="./images/pattern_samples/Man_Hat_Hide_Face_hyperface_like+landmark_noise+checkerboard_seed9167792_sample240500.jpg" width="200" style="display:block; margin:0; padding:0;"></a></td>
  </tr>
</table>

---

## Research Results

![Throughput](https://img.shields.io/badge/throughput-535%20tests%2Fmin-blue.svg)
![Data](https://img.shields.io/badge/data%20driven-yes-success.svg)
![Reproducible](https://img.shields.io/badge/reproducible-yes-brightgreen.svg)

### Performance Metrics

![Current Rate](https://img.shields.io/badge/current%20rate-535%20tests%2Fmin-orange.svg)
![Potential Rate](https://img.shields.io/badge/DGX%20Spark-24%2C000%20tests%2Fmin-green.svg)
![Speedup](https://img.shields.io/badge/speedup-45x-red.svg)

Current hardware achieves approximately 535 tests per minute. With dedicated infrastructure (NVIDIA DGX Spark with 4x A100 80GB GPUs), estimated throughput could reach 24,000 tests per minute (45x increase).

**Timeline Comparison:**
- Current rate (90 days): 69.3 million tests
- DGX Spark (2 days): 69.3 million tests

### Statistical Reporting

![Reports](https://img.shields.io/badge/reports-6%20types-blue.svg)
![Visualization](https://img.shields.io/badge/visualization-matplotlib-orange.svg)
![Analysis](https://img.shields.io/badge/analysis-scientific-purple.svg)

The fuzzer includes comprehensive reporting tools for scientific validation:

#### 1. Performance Report
![Report Type](https://img.shields.io/badge/report-performance-blue.svg)

Tracks raw throughput and testing velocity.

![Epoch 2 Performance Report](./images/reports/epoch_2_performance_report.png)

#### 2. Target Vulnerability Analysis
![Report Type](https://img.shields.io/badge/report-vulnerability-red.svg)

Identifies which test images are most vulnerable to adversarial patterns.

![Target Vulnerability Report](./images/reports/1_2_target_vulnerability_full_history.png)

#### 3. Pattern Success Rate Leaderboard
![Report Type](https://img.shields.io/badge/report-success%20rate-green.svg)

Calculates success rates for individual patterns.

![Pattern Success Rate Report](./images/reports/2_1_pattern_success_rate_full_history.png)

#### 4. Synergistic Pattern Combinations
![Report Type](https://img.shields.io/badge/report-synergy-purple.svg)

Identifies pattern combinations with enhanced effectiveness.

![Pattern Synergy Report](./images/reports/2_2_pattern_synergy_report_full_history.png)

#### 5. Top Vulnerabilities
![Report Type](https://img.shields.io/badge/report-top%20vulnerabilities-orange.svg)

Lists most repeatable failure cases for physical testing.

![Top 25 Vulnerabilities Report](./images/reports/1_3_top_vulnerabilities_by_image_and_recipe.png)

#### 6. Anomaly Type Distribution
![Report Type](https://img.shields.io/badge/report-anomaly%20distribution-yellow.svg)

Shows how patterns cause failures (confidence drop, person loss, etc.).

![Anomaly Type Distribution Report](./images/reports/1_1_pattern_anomaly_type_distribution.png)

---

## Project Background

![Mission](https://img.shields.io/badge/mission-privacy-blue.svg)
![Approach](https://img.shields.io/badge/approach-scientific-green.svg)
![Ethics](https://img.shields.io/badge/ethics-responsible-brightgreen.svg)

### The Mission

I'm a hacker who sees technology differently. For years, I've been fascinated by how machines interpret our world and us. I believe we can engineer fabrics to confuse these systems through reproducible, science-driven processes.

### Building on Giants

![Inspired By](https://img.shields.io/badge/inspired%20by-pioneers-purple.svg)

This project continues the groundbreaking work of:
- [capable.design](https://capable.design/)
- [adversarialfashion.com](https://adversarialfashion.com/)
- Adam Harvey's [adam.harvey.studio](https://adam.harvey.studio/)

While early artistic adversarial patterns were brilliant, many targeted older systems (like HAAR Cascade models) and were effective primarily under controlled conditions. Modern facial recognition has advanced significantly.

### Why This Matters

![Impact](https://img.shields.io/badge/impact-high-red.svg)
![Innovation](https://img.shields.io/badge/innovation-next%20gen-blue.svg)

**nonRecognition** brings those pioneering concepts into a new generation, building textiles for today's advanced detection systems through:

1. **Custom Fuzzer and Testing Suite** - High-performance tool for generating, testing, and analyzing patterns
2. **Adversarial Textiles** - Physically printed, sustainable materials optimized against modern recognition models

### About the Creator

![Experience](https://img.shields.io/badge/experience-decades-orange.svg)
![Community](https://img.shields.io/badge/community-SecKC-blue.svg)
![Speaker](https://img.shields.io/badge/speaker-BlackHat%20%7C%20DEF%20CON-red.svg)

[Bill Swearingen](https://about.me/billswearingen) - Security researcher with decades of experience:
- Co-founder of [SecKC](https://seckc.org)
- Speaker at [BlackHat](https://blackhat.com) and [DEF CON](https://defcon.org)
- Applying security mindset to privacy protection

---

## Roadmap

![Roadmap](https://img.shields.io/badge/roadmap-v0.5-blue.svg)
![Planning](https://img.shields.io/badge/status-planning-yellow.svg)

### v0.5 Proposals

#### Facial Recognition Matching Pipeline
![Feature](https://img.shields.io/badge/feature-1%3AN%20matching-orange.svg)

- Implement 1:N matching pipeline using ArcFace
- Test for misidentification (wrong person matching)

#### Feature Enhancements
![Enhancement](https://img.shields.io/badge/type-enhancement-blue.svg)

- Improve state detection without --resume flag
- Refactor Fuzz class for better maintainability
- Improve log parsing robustness
- Enhance Anomaly Preview resolution

#### Expanded Evaluation Harness
![Expansion](https://img.shields.io/badge/type-expansion-green.svg)

- Add support for MTCNN, RetinaFace detectors
- Build harness for cloud API testing (Azure, AWS Rekognition)

---

## Ethics and Intent

![Ethics](https://img.shields.io/badge/ethics-responsible%20disclosure-green.svg)
![Intent](https://img.shields.io/badge/intent-research-blue.svg)
![Goal](https://img.shields.io/badge/goal-privacy%20protection-brightgreen.svg)

This project is a research tool for auditing computer vision systems. The goal is to discover and document vulnerabilities in detection models to help developers build more robust, fair, and secure systems.

The adversarial patterns are offered to promote awareness and discussion about privacy in an age of ubiquitous surveillance. The mission is to build a wardrobe that protects privacy.

---

## Contributing

![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)
![Community](https://img.shields.io/badge/community-open-blue.svg)

Contributions are welcome. Please see CONTRIBUTING.md for guidelines.

---

## License

![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This project is licensed under the MIT License. See LICENSE file for details.

---

## Contact

![Contact](https://img.shields.io/badge/contact-open-blue.svg)
![Response](https://img.shields.io/badge/response-active-success.svg)

- **Author**: Bill Swearingen
- **Email**: bill@seckc.org
- **GitHub**: [@hevnsnt](https://github.com/hevnsnt)
- **Website**: [about.me/billswearingen](https://about.me/billswearingen)

---

## Acknowledgments

![Thanks](https://img.shields.io/badge/thanks-community-red.svg)
![Recognition](https://img.shields.io/badge/recognition-pioneers-purple.svg)

This research stands on the shoulders of giants in adversarial fashion and computer vision security. Special thanks to the pioneering work in adversarial patterns and privacy-preserving design.
