# Fetal MRI Ventriculomegaly Detection

A unified pipeline for automatic ventricular segmentation and ventriculomegaly diagnosis in fetal brain MRI. The system combines a classical 5-stage processing chain with an Attention U-Net trained via transfer learning from ImageNet.

**Team:** Ana Sofia Lopez Romero, Andrea Nathaly Ochoa Torres, Viviana Alejandre Paniagua, Laisha Nicole Cortes Montes
**Institution:** Tecnologico de Monterrey, Biomedical Engineering, Team 3, June 2026

---

## Clinical Background

Ventriculomegaly (VM) is one of the most common abnormalities found in fetal brain MRI, affecting approximately 1 in 100 pregnancies. It is diagnosed by measuring the width of the lateral ventricles at the atrial level in the axial transventricular plane. The accepted reference standard is the Cardoza criterion (1988):

| Classification | Atrial Diameter |
|----------------|-----------------|
| Normal | < 10 mm |
| Mild VM | 10 to 12 mm |
| Moderate VM | 12 to 15 mm |
| Severe VM | > 15 mm |

Cerebrospinal fluid (CSF) appears hyperintense in T2-weighted sequences because of its high water content and long T2 relaxation time. This physical property makes intensity-based thresholding a viable strategy for ventricular segmentation without needing labeled training data.

---

## Dataset

The notebook loads 10 fetal brain MRI volumes from two sources with different acquisition protocols:

| Source | Plane | Indexing |
|--------|-------|----------|
| Hospital dataset | Sagittal | `data[x, :, :]` |
| JC dataset | Axial | `data[:, :, z]` |

Volumes are provided in NIfTI format (`.nii` / `.nii.gz`). The notebook handles the orientation difference automatically and applies plane-aware slice selection throughout the pipeline.

Two volumes receive specialized analysis: `t2-t25` (JC axial, high image quality) and `05-badrecon` (hospital, poor reconstruction quality with low SNR). These represent the best and worst case scenarios in the dataset.

---

## Pipeline Overview

The notebook is organized into labeled sections that map to each deliverable of the SMART objective:

| Section | Stage | What it does |
|---------|-------|-------------|
| A | Data inventory | Loads all volumes, prints metadata (shape, resolution, FOV, spacing), generates orientation-aware mosaic |
| B | Interactive explorer | Volume selector with slice slider; updates range automatically when switching volumes |
| C | Stage I: Digital analysis | Quantitative analysis per slice: matrix size, resolution, bits, Nyquist criterion, six visual transforms (equalization, gamma, log) |
| D | DSP functions | Manual implementations of convolution, median filter, erosion, dilation, connected component labeling, and small object removal with explicit loops |
| E | Stage II: Filtering | Noise reduction applied to all volumes; compares three filters (Gaussian, median, bilateral) on 128x128 ROI crops |
| F | Stage III: Segmentation | Five strategies evaluated against the v3 reference; Dice scores compared |
| G | Stages IV-V | Atrial diameter measurement and clinical classification |
| H | CNN | Attention U-Net training, inference, and comparison against the classical pipeline |
| I | Metrics | Full table per volume: Dice, IoU, Sensitivity, Specificity, Accuracy, diameters, classifications |
| J | Batch processing | Eight-panel mini-report for every volume plus a summary comparison table |
| K | Conclusions | Formatted final report with methodology summary, per-volume results, and limitations |
| L | Priority analysis | Specialized processing for t2-t25 and 05-badrecon with optimized parameters |
| M | Presentation summaries | One summary image per stage, formatted for slides |

---

## Stage II: Noise Model and Filter Selection

MRI magnitude images follow a **Rician distribution**, not a Gaussian one. At high SNR (inside tissue), Rician noise approximates a positively skewed Gaussian. At low SNR (background), it converges to a Rayleigh distribution.

**Why Gaussian and not median:** The median filter is optimal for impulsive noise (salt-and-pepper), which is not the dominant noise type in MRI. Applying it to Rician noise rounds the edges of the ventricle boundary and biases the atrial diameter measurement. The Gaussian kernel smooths the noise while preserving the gradients that the thresholding step in Stage III relies on.

The classical v3 pipeline adds a CLAHE step (contrast-limited adaptive histogram equalization) followed by bilateral filtering to handle field inhomogeneity, which appears as a slow intensity gradient across the image.

---

## Stage III: Segmentation Strategies

Five methods are applied and compared against the v3 segmentation (reference):

| Method | Description |
|--------|-------------|
| 1. Global threshold | Otsu-style threshold at mu + 1.5 * sigma, followed by professor morphology |
| 2. Adaptive threshold | Local Gaussian mean threshold; handles regional intensity variation |
| 3. K-means (k=2) | Groups pixels into two clusters; the cluster with higher mean intensity is the CSF mask |
| 4. Canny + fill | Detects ventricle boundaries, fills enclosed regions, intersects with CSF mask |
| 5. v3 (reference) | Detects the interhemispheric fissure, seeds ventricle regions, segments at the 90th intensity percentile |

Dice score is computed for each method against method 5. The v3 pipeline consistently produces the most complete and anatomically plausible masks across both dataset sources.

---

## Stages IV-V: Quantification and Classification

The atrial diameter is computed from the segmented left and right ventricle masks using `cv2.minAreaRect`, which fits a minimum-area rotated rectangle to the mask contour. The minor axis of that rectangle corresponds to the clinical measurement used in the Cardoza criterion.

Additional geometric features extracted per ventricle:

| Feature | Unit |
|---------|------|
| Atrial diameter (minor axis of min. rect.) | mm |
| Major axis | mm |
| Area | mm^2 |
| Eccentricity (fitted ellipse) | dimensionless |
| Compactness = 4pi * Area / Perimeter^2 | dimensionless |

**Multiplanar validation:** The measurement is repeated on the three slices above and below the best axial slice. If the coefficient of variation across those slices is below 0.25, the mean is reported. If not, only the best slice is used and a warning is issued.

---

## Stage H: CNN with Transfer Learning

The deep learning component is an **Attention U-Net** (Oktay et al., MIDL 2018) with an **EfficientNet-B0 encoder** pretrained on ImageNet (Tan and Le, ICML 2019), implemented via `segmentation_models_pytorch`.

Because no pixel-level annotations are available, **pseudo-labels** generated by the classical v3 pipeline are used as training targets. This approach is justified by Raghu et al. (NeurIPS 2019), who showed that ImageNet features transfer effectively to medical imaging tasks even with small datasets.

Training proceeds in two phases:

| Phase | What is updated | Rationale |
|-------|----------------|-----------|
| 1. Decoder only | All decoder + attention gates | Prevents encoder weights from diverging before the decoder learns the task |
| 2. Full fine-tuning | All layers | Allows the encoder to adapt its low-level features to grayscale MRI texture |

Loss function: Dice loss (Milletari et al., 3DV 2016), chosen for its robustness under class imbalance between the small ventricular region and the rest of the brain.

---

## Metrics

The evaluation table (Section I) reports the following per volume and per method:

| Metric | Description |
|--------|-------------|
| Dice | 2 * TP / (2 * TP + FP + FN) |
| IoU | TP / (TP + FP + FN) |
| Sensitivity | TP / (TP + FN) — fraction of ventricle pixels captured |
| Specificity | TN / (TN + FP) — fraction of non-ventricle pixels correctly rejected |
| Accuracy | (TP + TN) / total pixels |
| Atrial diameter | mm, per ventricle |
| Classification | Normal / Mild VM / Moderate VM / Severe VM |

CNN metrics use the v3 pseudo-label as the reference mask, not a manual annotation. This means CNN Dice scores reflect agreement with the classical pipeline, not ground truth.

---

## How to Run

The notebook is designed for **Google Colab** with Google Drive mounted.

1. Upload `RETO_VENTRICULOMEGALIA.ipynb` to Google Colab
2. Mount your Google Drive when prompted (Section A)
3. Place your NIfTI volumes in a folder called `Ejemplos mri` in your Drive root
4. Run all cells in order from top to bottom

**Runtime estimate:** Sections A-G run in under 10 minutes on a free Colab CPU. Section H (CNN training) requires a GPU runtime (T4 or better) and takes approximately 15 to 30 minutes depending on the number of volumes.

**Dependencies** (all available in Colab by default or installed in Section 1):
- PyTorch + torchvision
- segmentation-models-pytorch (`smp`)
- nibabel or SimpleITK (for NIfTI loading)
- OpenCV (`cv2`)
- NumPy, Pandas, Matplotlib
- ipywidgets (for Section B interactive explorer)
- scikit-image

---

## Files

```
.
+-- RETO_VENTRICULOMEGALIA.ipynb    # Complete unified pipeline notebook
+-- docs/
    +-- IEEE_PROCESAMIENTO.pdf      # Full IEEE-format report
```

---

## Future Work

**Ground truth annotations:** All CNN metrics in this project use pseudo-labels from the classical pipeline as reference. Recruiting a radiologist to annotate even 20 to 30 slices would allow computing true Dice and IoU against expert segmentation, making the results comparable to published benchmarks like FeTA 2021 (Payette et al., 2021).

**3D volumetric segmentation:** The current pipeline processes one 2D slice at a time. Replacing the 2D Attention U-Net with a 3D U-Net would leverage the spatial continuity between slices and reduce the variance across the multiplanar validation step.

**Multi-class segmentation:** The pipeline currently produces a binary ventricle mask. Extending it to segment cortex, white matter, and subcortical structures simultaneously would allow computing a full brain parcellation, which is the clinical standard for fetal MRI reporting.

**Automated slice selection:** The best axial slice is currently selected by finding the slice with the largest ventricular area. A dedicated classifier trained on anatomical landmarks (the foramen of Monro and the choroid plexus) would make slice selection more robust on volumes with poor reconstruction quality.

**Severity trend analysis:** Applying the pipeline to longitudinal scans of the same patient at different gestational ages would allow tracking whether ventriculomegaly progresses, stabilizes, or resolves, which is relevant for prognosis in cases of mild isolated VM.

---

## References

Cardoza J. et al., "Detection of fetal intracranial abnormalities with ultrasound," *Obstetrics and Gynecology*, 1988.

Prayer D. et al., "MRI of fetal brain abnormalities," *Neuroradiology*, 2017.

Glenn O.A., "MR imaging of the fetal brain," *Pediatric Radiology*, 2010.

Oktay O. et al., "Attention U-Net: Learning Where to Look for the Pancreas," *MIDL 2018.* arXiv:1804.03999.

Tan M. and Le Q.V., "EfficientNet: Rethinking Model Scaling for CNNs," *ICML 2019.* arXiv:1905.11946.

Raghu M. et al., "Transfusion: Understanding Transfer Learning for Medical Imaging," *NeurIPS 2019.* arXiv:1902.07208.

Milletari F. et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation," *3DV 2016.* arXiv:1606.04797.

Payette K. et al., "An automatic multi-tissue human fetal brain segmentation benchmark using the Fetal Tissue Annotation Dataset (FeTA)," *Scientific Data* 8, 330 (2021).
