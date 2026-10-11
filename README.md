
# 3D cbct-sct

3D CBCT to synthetic CT. This repository is built upon [nnUNet_translation](https://github.com/phyrise/nnUNet_translation) 

## Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd 3d-cbct-sct
uv python install 3.11
uv sync --python 3.11
uv sync --project totalseg --python 3.11
```

TotalSegmentator is a second environment because it loads its own `nnunetv2`; one shared environment would import this translation package. Pseudo labels and the 7-class teacher use `totalseg/.venv`. Translation training uses `uv sync`. 

Elastix 5.2.0 (Linux x86-64) is not in git. Parameters are in `core/synthrad2025_configs`.

```bash
bash scripts/fetch_elastix.sh
```

## Data


```bash
source scripts/nnunet_env.sh
```

Paired NRRD: `data/nrrd/CBCT` and `data/nrrd/CT`, same file name (for example `P001_1.nrrd`).

```bash
uv run python core/nrrd_to_nifti.py --src data/nrrd --dst data/work/rigid
uv run python core/make_fov_masks.py --cbct-dir data/work/rigid/CBCT --out-dir data/work/rigid/fovMasks
uv run python core/run_elastix_deformable.py --src data/work/rigid --dst data/work/deformable --region HN --fov-mask-dir data/work/rigid/fovMasks --workers 4 --threads-per-job 4
uv run python core/prepare_training_data.py --config config.yaml
```

FOV figures: `data/work/rigid/fovMasks/vis`.
Registration figures: `data/work/deformable/vis`.
Fixed image is CBCT.

`test_cases: [P001_1, P002_1]` writes those cases to `imagesTs`. Empty keeps every case in `imagesTr`.

## MAE training

```bash
uv run nnUNetv2_train 93 3d_fullres 0 -tr nnUNetTrainerMRCT_mae_scratch300 -p nnResUNetPlans
```

## MAE + AFP training

```bash
# 7-class pseudo labels. --hn adds TotalSegmentator head and neck muscles to the muscles class (use it for head CT).
uv run python core/extract_pseudo_labels.py --hn
# Training the 7-class AFP encoder on pseudo labels.
uv run --project totalseg python core/train_teacher.py --fold 0
# MAE + AFP training
uv run nnUNetv2_train 93 3d_fullres 0 -tr nnUNetTrainerMRCT_AFP_scratch300 -p nnResUNetPlans
```
Dropout variants (`p=0.1` on encoder stages E3–E5) append `_drop`: `nnUNetTrainerMRCT_mae_scratch300_drop`, `nnUNetTrainerMRCT_AFP_scratch300_drop`.

### Schematic diagram of the `nnUNetTrainerSegScratch` Trainer
 <img width="4209" height="1560" alt="AFP_figure" src="https://github.com/user-attachments/assets/999dd5ac-9256-4fa2-8bfa-596c96b39639" />
 
 This figure was created using imaging data from the [SynthRAD2025 dataset](https://doi.org/10.5281/zenodo.14918089)
(Thummerer et al., 2025), licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Image processing and visualization were performed by the authors of this repository.


## Inference and evaluation

One command per trained model. It predicts the test cases (`imagesTs`), converts to HU, scores, and plots.

```bash
uv run python core/evaluate_sct.py --trainer nnUNetTrainerMRCT_mae_scratch300
uv run python core/evaluate_sct.py --trainer nnUNetTrainerMRCT_AFP_scratch300
```

Output: `results/eval/<trainer>/fold_0/test/`. `metrics_geom.json` holds MAE, PSNR, and NCC inside the geometric FOV (the reported score).
## References

1. Longuefosse et al. "[Adapted nnU-Net: A Robust Baseline for Cross-Modality Synthesis and Medical Image Inpainting](https://doi.org/10.1007/978-3-031-73281-2_3)". *SASHIMI*, 2024.

2. Isensee et al. "[nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://doi.org/10.1038/s41592-020-01008-z)". *Nature Methods*, 2021.

3. Longuefosse et al. "[Anatomical feature-prioritized loss for enhanced MR to CT translation](https://doi.org/10.1088/1361-6560/adea07)". *Physics in Medicine & Biology*, 2025.

4. Sequeiro Gonzalez et al. "[Deep Learning-Based Cross-Anatomy CT Synthesis Using Adapted nnResU-Net with Anatomical Feature Prioritized Loss](https://arxiv.org/abs/2509.22394)". *arXiv*, 2025.

5. https://github.com/SynthRAD2025/preprocessing

6. Thummerer et al. "[SynthRAD2025 Grand Challenge dataset: Generating synthetic CTs for radiotherapy from head to abdomen](https://doi.org/10.1002/mp.17981)". *Medical Physics*, 2025.
