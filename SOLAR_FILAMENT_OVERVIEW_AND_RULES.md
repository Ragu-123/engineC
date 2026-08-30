# ☀️ Solar Filament Segmentation Challenge 2026: Overview, Rules & Prize Requirements

---

## 1. 📌 Competition Synopsis

* **Official Title**: Solar Filament Segmentation Challenge 2026
* **Host**: **Earth-Space AI Research (ESAIR) Lab** (University of Missouri–St. Louis / Georgia State University & National Solar Observatory)
* **Goal**: Automatic instance segmentation of solar filaments from full-disk H-Alpha ($2048 \times 2048$) solar observations collected by the GONG telescope network.
* **Prize Pool**: **$3,000 USD**
* **Deadline**: **November 15, 2026 (06:00 UTC)**
* **Winner Announcement**: **IEEE BigData 2026 Conference** (December 14–17, 2026, Phoenix, Arizona, USA)

---

## 2. ⚖️ Evaluation Rubric (70% Quantitative / 30% Qualitative)

The organizers have formally stated that **public leaderboard ranking is only one factor** and not the sole determinant of winners. The final evaluation uses a hybrid rubric:

### A. Quantitative Performance (70%)
1. **Mean Dice Score**: Computed via `torchmetrics.segmentation.DiceScore` against ground-truth instances.
2. **Dice & IoU Distribution**: Uniformity and stability of segmentation quality across all test observations.
3. **Fragmentation & Over-Merging Penalties**:
   * **One-to-Many Penalty**: A single biological filament incorrectly split into multiple fragmented predicted masks.
   * **Many-to-One Penalty**: Distinct adjacent filaments erroneously merged into a single mask blob.
4. **Computational Efficiency**: End-to-end inference runtime and hardware resource utilization.

### B. Qualitative & Scientific Rigor (30%)
1. **Morphological Fidelity**: Visual inspection of predicted segmentations against H-Alpha images to ensure physically accurate boundaries, spine continuity, and absence of spurious background artifacts.
2. **Methodology Documentation**: A comprehensive technical paper/report documenting the preprocessing, architecture, hyperparameters, and post-processing.
3. **Code Modularity & Quality**: Clean, modular, well-commented code that conforms to the competition's open-access policy.

---

## 3. 📤 Submission Contract & Constraints

### CSV Schema
Submissions must be written to `/kaggle/working/submission.csv`:
```csv
filament_id,segmentation_rle
20110120105534Ch_1,PPP8...
20110120105534Ch_2,PPP8...
```

* **`filament_id`**: `<observation_stem>_<instance_index>` (e.g. `20110120105534Ch_1`).
* **`segmentation_rle`**: Compressed COCO Run-Length Encoding string (`pycocotools.mask.encode(np.asfortranarray(mask))['counts'].decode('ascii')`) targeting native $2048 \times 2048$ resolution.

### ⚠️ Mandatory Non-Overlap Rule (Panoptic Constraint)
Under the official Panoptic Quality scorer, **no two predicted instances in the same image may share overlapping positive pixels**. If masks overlap, the submission status will immediately fail with `SubmissionStatus.ERROR`. Detections must be processed through **Score-Ordered Panoptic Painting** before RLE encoding.

---

## 4. 🏆 Mandatory Prize Eligibility Criteria

To receive prize money ($3,000 USD split across Top 3 teams):
1. **Code Deliverables**: Winning teams must provide reproducible source code (training pipelines, inference scripts, dependency versions, and model checkpoints).
2. **Methodology Report**: Submit a technical document detailing the pipeline and design decisions.
3. **Team Size**: Maximum 5 participants per team.
4. **Vendor Onboarding**: Complete vendor onboarding (including tax verification like W-8BEN for non-US participants and international SWIFT banking details).
