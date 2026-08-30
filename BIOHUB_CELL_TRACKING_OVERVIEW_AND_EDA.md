# 🔬 Biohub: Cell Tracking During Development (Zebrafish 3D+Time)

---

## 1. 📌 Competition Synopsis

* **Official Title**: Biohub - Cell Tracking During Development
* **Host**: **Chan Zuckerberg Biohub (CZ Biohub)** (Royer Lab)
* **Goal**: Detect and track developing zebrafish embryonic cells through **3D space and time ($3\text{D} + \text{time}$)**, including mitosis (division) events and cell lineage graph reconstruction.
* **Prize Pool**: **$60,000 USD**
* **Deadline**: **September 29, 2026**
* **Submission Mode**: **Code Competition (Kernels-Only)**, max 5 daily submissions.
* **Evaluation Metric**: **Graph Jaccard / Edge F1 Matching** with a $7.0\ \mu\text{m}$ Euclidean matching radius in physical coordinates.

---

## 2. 📁 Dataset Architecture & Dimensions

Stored as OME-Zarr v0.5 / Zarr V3 4D imagery and GEFF graph annotations:

```text
/kaggle/input/competitions/biohub-cell-tracking-during-development/
├── sample_submission.csv        # Example submission CSV format
├── train/                       # 199 Volumes
│   ├── <dataset_id>.zarr/       # 4D Spatiotemporal microscopy volume (T, Z, Y, X)
│   └── <dataset_id>.geff/       # Paired ground-truth cell lineage graph (Nodes + Edges)
└── test/                        # 4 Volumes
    └── <dataset_id>.zarr/       # 4D Spatiotemporal microscopy volume (T, Z, Y, X)
```

### Dataset Specifications:
| Property | Specification |
| :--- | :--- |
| **Volume Shape $(T, Z, Y, X)$** | **$(100, 64, 256, 256)$** (100 timepoints, 64 depth slices, $256 \times 256$ XY frame) |
| **Total Frames per Volume** | **6,400 2D slices** |
| **Voxel Data Type** | `uint16` ($[0, 65535]$, typical foreground signal $[75, 2145]$) |
| **Chunk Shape** | `(1, 64, 256, 256)` (one full 3D spatial volume per timepoint chunk) |
| **Physical Voxel Spacing** | **$[T=1.0\text{ s}, Z=1.625\ \mu\text{m}, Y=0.40625\ \mu\text{m}, X=0.40625\ \mu\text{m}]$** |
| **Anisotropic Aspect** | The $Z$-axis spacing is $4\times$ thicker than $X/Y$ ($1.625 / 0.40625 = 4.0$). |

---

## 3. 🧬 Embryo Cohorts & Validation Setup

* **Embryo `44b6`**: 71 train volumes (35.7%), 2 test volumes (50.0%)
* **Embryo `6bba`**: 128 train volumes (64.3%), 2 test volumes (50.0%)
* **Validation Strategy**: Use **GroupKFold** grouped on embryo ID or non-overlapping spatiotemporal clusters.

---

## 4. 📊 Ground Truth Graph Annotations (`.geff`)

Analysis of all 199 training `.geff` files:
* **Total Nodes**: **133,318** cell centroids $(t, z, y, x)$.
* **Total Edges**: **128,883** temporal tracking links $(u \to v)$.
* **Total Division Events**: **151** cell divisions across the entire dataset.
* **Nodes per Volume**: Mean **$669.9$** (Median: $659.0$, Range: $50 - 1950$).
* **Cell Density**: Mean **$7.04$ annotated cells per frame**.
* **Sparse Annotation Rule**: The ground-truth tracks are a **sparse subset**. Your model must track **all cells** in the volume; the evaluator matches against the sparse subset.

---

## 5. 🏃 Cell Kinematics & Physical Displacement

* **Frame-to-Frame Physical Speed**:
  * Mean: **$2.13\ \mu\text{m}$ / frame** ($\approx 3.98\text{ px}$)
  * Median: **$1.82\ \mu\text{m}$ / frame** ($\approx 3.16\text{ px}$)
  * 95th Percentile: **$5.34\ \mu\text{m}$ / frame** ($\approx 10.49\text{ px}$)
  * Max: **$60.76\ \mu\text{m}$ / frame**
* **Temporal Continuity**: $100.00\%$ of annotated transitions connect adjacent frames ($\Delta t = 1$).

---

## 6. 📤 Unified Submission Format

Submissions are formatted as a single CSV containing both `node` and `edge` records:

```csv
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6_0113de3b,node,1,0,32,128,128,-1,-1
1,44b6_0113de3b,node,2,1,32,128,128,-1,-1
2,44b6_0113de3b,edge,-1,-1,-1,-1,-1,1,2
```

1. **`node` row**: `row_type = 'node'`, `node_id = <int>`, `t`, `z`, `y`, `x` (voxel coords), `source_id = -1`, `target_id = -1`.
2. **`edge` row**: `row_type = 'edge'`, `node_id = -1`, `t=-1, z=-1, y=-1, x=-1`, `source_id = <parent_node_id>`, `target_id = <child_node_id>`.

---

## 7. 💡 Proven Winning Strategies

```mermaid
flowchart LR
    A["3D Volume (100, 64, 256, 256)"] --> B["Step 1: Multi-Scale 3D DoG / StarDist"]
    B --> C["Candidate Centroids (t, z, y, x)"]
    C --> D["Step 2: Hungarian / LAPTrack Linking (Dist <= 7.0 µm)"]
    D --> E["Lineage Graph (Nodes + Edges)"]
    E --> F["submission.csv"]
```

1. **Detection is the Dominant Lever (>80% Score Contribution)**:
   * Multi-Scale 3D Difference of Gaussians (DoG) with anisotropic kernel scaling ($(\sigma_z, \sigma_y, \sigma_x) = (s / 4.0, s, s)$) and scale-space maximum suppression reaches **LB > 0.826** without neural network training.
2. **Tracking & Association**:
   * Pairwise Euclidean distance in physical coordinates:
     $$\text{dist} = \sqrt{(1.625 \Delta z)^2 + (0.40625 \Delta y)^2 + (0.40625 \Delta x)^2}$$
   * Hungarian algorithm / Linear Sum Assignment with a distance gate $\le 7.0 - 8.0\ \mu\text{m}$.
   * Mitosis splitting: Allow 1-to-2 links when two child candidates appear near a division site.
