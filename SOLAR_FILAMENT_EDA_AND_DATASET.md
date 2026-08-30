# 🔬 Solar Filament Segmentation: Comprehensive EDA & Dataset Audit

---

## 1. 📁 Directory Structure & File Inventory

```text
/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/                                 # 707 Grayscale Images (2048 × 2048 JPEG)
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json # COCO JSON (1,154 image records, 8,199 annotations)
└── test/
    └── test_images/                                  # 180 Grayscale Images (2048 × 2048 JPEG)
```

* **Image Dimensions**: Exactly $2048 \times 2048$ pixels.
* **Color Mode**: Single-channel grayscale (`mode: L`, 8-bit $[0, 255]$).
* **Image Integrity**: 100% valid; zero corruptions or decode errors across all train and test files.

---

## 2. 🌍 GONG Solar Observatory Stations

Filenames follow the pattern `YYYYMMDDHHMMSS<Station>h.jpeg`. The single-letter code denotes the Global Oscillation Network Group (GONG) observatory site:

| Code | Observatory Site | Train Images | Test Images | Solar Disk Mean Intensity | Image Std |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **`Lh`** | Learmonth (Australia) | 127 (18.0%) | 34 (18.9%) | $127.8 \pm 1.5$ | $56.4 \pm 0.9$ |
| **`Ch`** | Cerro Tololo (Chile) | 125 (17.7%) | 33 (18.3%) | $133.1 \pm 2.1$ | $58.2 \pm 2.0$ |
| **`Mh`** | Mauna Loa (Hawaii, USA) | 125 (17.7%) | 26 (14.4%) | $127.3 \pm 1.6$ | $57.6 \pm 2.9$ |
| **`Bh`** | Big Bear (California, USA) | 118 (16.7%) | 33 (18.3%) | $129.6 \pm 2.2$ | $58.0 \pm 2.4$ |
| **`Th`** | Teide (Tenerife, Spain) | 110 (15.6%) | 27 (15.0%) | $127.6 \pm 1.7$ | $54.2 \pm 1.8$ |
| **`Uh`** | Udaipur (India) | 102 (14.4%) | 27 (15.0%) | $129.1 \pm 2.0$ | $56.9 \pm 1.1$ |

* **Observation Range**: 2011-01-09 to 2022-08-03 (Solar Cycle 24 and early Cycle 25).
* **Station Parity**: Train and test sets have matching station balances and exposure characteristics.

---

## 3. 🏷️ Category & Chirality Distribution

The JSON schema specifies 4 category IDs:

* **Category 1 (Left-bearing)**: 2,535 instances (**30.92%**)
* **Category 2 (Right-bearing)**: 2,590 instances (**31.59%**)
* **Category 3 (Unidentifiable)**: 3,074 instances (**37.49%**)
* **Category 4 (Ambiguous)**: 0 instances (**0.00%**)

> [!NOTE]
> Chirality (Left vs. Right) is evenly balanced. Category 4 has 0 instances and can be omitted.

---

## 4. 👥 Multi-Annotator Dynamics & Human Agreement Ceiling

A critical attribute of MAGFiLO 1.0 is that **296 of the 707 training images (41.9%)** were independently labeled by 2 or 3 solar physicists:

* **1 Annotator**: 411 images (58.1%)
* **2 Annotators**: 145 images (20.5%)
* **3 Annotators**: 151 images (21.4%)

### Human-to-Human Inter-Rater Agreement:
* **Semantic Mask Dice**: **Mean = 0.5946**, **Median = 0.6151** (Min: 0.18, Max: 0.85)
* **Semantic Mask IoU**: **Mean = 0.4372**, **Median = 0.4441**
* **Instance Count Discrepancy**: **$2.0 \pm 1.8$ filaments** difference per image (max difference of 10 instances).

> [!IMPORTANT]
> Because expert-to-expert agreement averages **Dice ~0.60–0.69**, realistic high-performing models operate in the 0.65–0.72 Dice range. Scores substantially higher (>0.85) generally reflect overfitting to specific annotator conventions or public dataset leakage.

---

## 5. 📐 Morphological & Geometric Statistics

### Instance Area (COCO Scales):
* **Small** ($< 1,024\text{ px}^2$ / $< 32\times 32$): **3,473 (42.36%)**
* **Medium** ($1,024 - 9,216\text{ px}^2$): **4,492 (54.79%)**
* **Large** ($\ge 9,216\text{ px}^2$): **234 (2.85%)**

### Key Quantiles:
| Metric | Min | 25% | Median | Mean | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Area ($\text{px}^2$)** | 9.0 | 670.0 | **1,228.0** | 2,119.8 | 2,438.0 | 37,739.0 |
| **BBox Aspect Ratio ($W/H$)** | 0.12 | 0.74 | **1.11** | 1.38 | 1.76 | 12.78 |
| **Spine Length ($\text{px}$)** | 12.1 | 74.5 | **111.3** | 147.4 | 185.0 | 1,105.0 |
| **Spine Points** | 2 | 7 | **10** | 12.5 | 15 | 122 |
| **Instances per Image** | 1 | 4 | **7.0** | 7.1 | 9 | 26 |

---

## 6. ☀️ Spatial Distribution on the Solar Disk

* **Solar Disk Center**: $(1024, 1024)$
* **Solar Radius ($R_{\odot}$)**: $\approx 850 - 950\text{ px}$
* **Filament Centroids**: Mean $(1030.5, 1040.5)$, Std $(\approx 412, 349)$
* **Limb Boundary**: **100.00%** of all filaments reside strictly within radial distance $R < 950\text{ px}$. The outer dark space contains zero filaments.
