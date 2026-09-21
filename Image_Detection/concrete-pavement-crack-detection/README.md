# Concrete Infrastructure Crack Detection & Classification
## Automated Visual Inspection for Structural Health Monitoring (SHM)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end deep learning computer vision system for concrete crack detection, classification, explainability, and severity estimation. Tailored for civil infrastructure monitoring—including reinforced concrete bridge decks, highway pavements, and retaining walls—connecting directly to structural health monitoring (SHM) assessments.

---

## 1. Engineering Motivation & SHM Context

Visual crack detection is the foundational diagnostic tool for structural maintenance:
- **Corrosion Catalyst**: Cracks are the primary pathway allowing water, atmospheric carbon dioxide (carbonation), and deicing salts (chloride attack) to reach embedded steel rebar. Once depassivated, expanding rust causes spalling and loss of tensile steel section.
- **Safety-Critical Asymmetry**: In civil engineering, a **False Negative (missed crack)** can lead to undetected structural degradation and sudden brittle failure under live traffic loads. Conversely, a **False Positive (false alarm)** merely requires routine manual verification. Our models incorporate **cost-sensitive weighted loss functions** to maximize **Recall** and safety margins.
- **Explainability**: Structural engineers require verifiable proof that deep learning models focus on actual fracture lines rather than ambient surface artifacts (formwork seams, broom tining, aggregate exposure, or guardrail shadows). We integrate **Grad-CAM** attention mapping to validate model spatial reasoning.

---

## 2. Dataset: SDNET2018 Benchmark

The system uses the **SDNET2018** benchmark dataset from Utah State University, comprising over **56,000 annotated images** ($256 \times 256$ pixels) reflecting real-world infrastructure inspections:

| Structural Element | Code | Subfolders | Cracked Count | Uncracked Count | Total Images |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bridge Decks** | `D` | `CD` (Cracked), `UD` (Uncracked) | 2,025 | 11,595 | 13,620 |
| **Pavements** | `P` | `CP` (Cracked), `UP` (Uncracked) | 2,608 | 21,726 | 24,334 |
| **Walls** | `W` | `CW` (Cracked), `UW` (Uncracked) | 3,851 | 14,287 | 18,138 |
| **Grand Total** | — | — | **8,484 (15.1%)** | **47,608 (84.9%)** | **56,092** |

> **Natural Class Imbalance**: With 15.1% cracked vs. 84.9% sound concrete, naive accuracy is misleading. Our training pipeline calculates a positive class weight ($w_{\text{pos}} \approx 5.61$) to enforce high sensitivity to structural cracks.

---

## 3. System Structure & Methodology

```
concrete-pavement-crack-detection/
├── data/                                 # Dataset directory / symlinks
├── src/
│   ├── __init__.py                       # Package declaration
│   ├── dataset.py                        # SDNET loader, stratified splits (70/15/15), augmentations
│   ├── models.py                         # ResNet-18 & MobileNetV2 transfer learning wrappers
│   ├── train.py                          # Mixed-precision two-phase training loop with EarlyStopping
│   ├── evaluate.py                       # Test evaluation, confusion matrix, ROC/PR curves, failure analysis
│   ├── explainability.py                 # Grad-CAM attention heatmap visualization
│   └── severity.py                       # ACI 224R crack width and severity grading module
├── Crack-detection-nb.ipynb              # Comprehensive, storytelling Jupyter Notebook (Colab-ready)
├── requirements.txt                      # Python dependencies
├── Pipfile                               # Pipenv virtual environment configuration
└── README.md                             # Project documentation
```

### Model Architectures
- **ResNet-18**: Deep residual connections ($F(x) + x$) retain high-frequency spatial edge details, ideal for thin fracture propagation paths.
- **MobileNetV2**: Inverted residual blocks with linear bottlenecks and depthwise separable convolutions. Delivers an 80% reduction in parameters ($2.2\text{M}$ vs $11.2\text{M}$), enabling real-time edge deployment on drone-mounted inspection cameras ($>30$ FPS).

### Two-Phase Transfer Learning
1. **Phase 1 (Warmup)**: Freeze backbone convolutional feature extractors; train the binary classification head with AdamW ($\text{LR} = 10^{-3}$).
2. **Phase 2 (Fine-Tuning)**: Unfreeze top convolutional blocks at a reduced learning rate ($\text{LR} = 10^{-4}$), adapting filters to concrete aggregate textures and micro-fissures.

---

## 4. Advanced SHM Extensions

### 1. Grad-CAM Visual Explainability
Computes the gradients of the crack class score with respect to the final convolutional feature maps:
$$\alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y_{\text{crack}}}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_{k} \alpha_k A^k\right)$$
Overlays high-resolution attention heatmaps onto the concrete surface to visually verify fracture localization.

### 2. Crack Severity & Width Estimation (ACI 224R Conformance)
Uses morphological thinning and distance transform algorithms masked by Grad-CAM attention to calculate the maximum and mean crack width in millimeters, bucketing detections into American Concrete Institute standards:
- **Low / Hairline ($<0.30$ mm)**: Tolerable shrinkage micro-cracking; periodic SHM logging.
- **Moderate ($0.30 - 1.00$ mm)**: Exceeds durability limits for exposed bridges; requires epoxy sealing or water-repellent silane treatment.
- **Severe ($>1.00$ mm)**: Critical structural warning indicating possible flexural/shear distress, settlement, or rebar corrosion; demands immediate engineering inspection.

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support (e.g. GTX 1050 Ti or higher) or Google Colab

### Using Pipenv (Recommended for Local PC)
```powershell
# Navigate to project directory
cd "d:\HafeezBello\HHB\JUPYTER\CompVis_Projects\concrete-pavement-crack-detection"

# Activate environment and install dependencies
pipenv install
pipenv run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Using Standard Pip / Virtualenv
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 6. Usage Guide

### Interactive Jupyter Notebook
Launch the complete, storytelling notebook:
```powershell
pipenv run jupyter notebook Crack-detection-nb.ipynb
```
The notebook is structured according to `ml-best-practices` with alternating code execution and civil engineering markdown analyses.

### Running via Python Scripts
```python
# 1. Load Data
from src.dataset import get_dataloaders
train_loader, val_loader, test_loader, info = get_dataloaders(structure_type="deck", batch_size=32)

# 2. Build & Train Model
from src.models import build_model
from src.train import train_crack_model

model = build_model("resnet18", pretrained=True)
history = train_crack_model(model, train_loader, val_loader, epochs=5, pos_weight=info["pos_weight"])

# 3. Evaluate on Unseen Test Set
from src.evaluate import evaluate_model_on_test_set, plot_confusion_matrix
results = evaluate_model_on_test_set(model, test_loader)
plot_confusion_matrix(results["confusion_matrix"])

# 4. Generate Grad-CAM Explainability Heatmap
from src.explainability import explain_crack_prediction
fig, prob, heatmap = explain_crack_prediction(model, "path/to/concrete_sample.jpg")

# 5. Estimate Severity
from src.severity import estimate_crack_severity
severity = estimate_crack_severity("path/to/concrete_sample.jpg", gradcam_heatmap=heatmap, visualize=True)
print("Severity Tier:", severity["severity_level"])
```

---

## 7. Hardware Compatibility: Local PC & Google Colab

| Feature | Local PC (NVIDIA GTX 1050 Ti, 4GB) | Google Colab (Tesla T4, 15GB) |
| :--- | :--- | :--- |
| **CUDA Acceleration** | Yes (Native CUDA 12.1 via PyTorch) | Yes (Default GPU runtime) |
| **Mixed Precision** | `torch.amp.autocast('cuda')` (~1.2 GB VRAM) | Supported (Fast throughput) |
| **Batch Size** | 32 (Safe headroom, no OOM) | 64 – 128 |
| **Execution Path** | Direct local access to `D:\...\SDNET2018` | 1-Click download / Google Drive mount |

---

## 8. References

1. **SDNET2018 Dataset**: Dorafshan, S., Thomas, R. J., & Maguire, M. (2018). *SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks*. Data in Brief, 21, 1664–1668.
2. **ACI Committee 224**: *Control of Cracking in Concrete Structures (ACI 224R-01)*. American Concrete Institute, Farmington Hills, MI.
3. **Grad-CAM**: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. ICCV 2017.
4. **Structural Health Monitoring**: Aktan, A. E., et al. (2000). *Issues in infrastructure health monitoring for management*. Journal of Engineering Mechanics, 126(7), 711-724.
