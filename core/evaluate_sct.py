"""Predict, convert to HU, score, and plot one trained sCT model.

    uv run python core/evaluate_sct.py --trainer nnUNetTrainerMRCT_AFP_scratch300

Output goes under nnunet_results/eval/<trainer>/fold_<fold>/<split>/:
    pred/               nnUNetv2_predict output (normalized)
    hu/                 HU volumes
    metrics_geom.json   MAE, PSNR, NCC inside the geometric FOV (the reported score)
    viz/                figures

Wraps nnUNetv2_predict, denorm_predictions_to_hu.py, eval_mae_body_mask.py, and
viz_sct_results.py with the paths from config.yaml. --split test reads imagesTs,
--split train reads imagesTr.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

CORE = Path(__file__).resolve().parent
REPO = CORE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nnunetv2.training.loss.AFP_from_plans import apply_dir_env, load_config, resolve_cfg  # noqa: E402


def run(cmd: list[str], env: dict[str, str]) -> None:
    logger.info("+ {}", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainer", required=True, help="e.g. nnUNetTrainerMRCT_mae_scratch300, nnUNetTrainerMRCT_AFP_scratch300")
    ap.add_argument("--plans", default="nnResUNetPlans")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--split", choices=["test", "train"], default="test", help="imagesTs or imagesTr")
    ap.add_argument("--config", type=Path, default=REPO / "config.yaml")
    ap.add_argument("--data-range", type=float, default=2000.0, help="PSNR peak span in HU")
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Redo prediction even if pred/ exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_dir_env(cfg)
    raw = resolve_cfg(cfg, "nnunet_raw")
    preprocessed = resolve_cfg(cfg, "nnunet_preprocessed")
    results = resolve_cfg(cfg, "nnunet_results")
    work = resolve_cfg(cfg, "work_prefix")
    name = str(cfg["dataset_name"])
    cbct_id = int(cfg["cbct_dataset_id"])
    ct_id = cbct_id + 1
    images = "imagesTs" if args.split == "test" else "imagesTr"

    cbct_dir = raw / f"Dataset{cbct_id:03d}_{name}_CBCT" / images
    gt_dir = raw / f"Dataset{ct_id:03d}_{name}_CT" / images
    fingerprint = preprocessed / f"Dataset{ct_id:03d}_{name}_CT" / "dataset_fingerprint.json"
    mask_dir = work / "rigid" / "fovMasks"
    model_dir = results / f"Dataset{cbct_id:03d}_{name}_CBCT" / f"{args.trainer}__{args.plans}__{args.configuration}"
    ckpt = model_dir / f"fold_{args.fold}" / args.checkpoint
    for p, what in ((cbct_dir, "CBCT images"), (gt_dir, "CT images"), (fingerprint, "CT fingerprint"),
                    (mask_dir, "FOV masks"), (ckpt, "checkpoint")):
        if not p.exists():
            raise SystemExit(f"missing {what}: {p}")

    out = results / "eval" / args.trainer / f"fold_{args.fold}" / args.split
    pred = out / "pred"
    hu = out / "hu"
    metrics = out / "metrics_geom.json"
    viz = out / "viz"
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    py = sys.executable
    predict_bin = Path(py).parent / "nnUNetv2_predict"
    if not predict_bin.is_file():
        raise SystemExit(f"missing {predict_bin}. Run with uv run from the repository root.")

    if args.overwrite and pred.exists():
        shutil.rmtree(pred)
    if pred.exists() and list(pred.glob("*.nii.gz")):
        logger.info("prediction exists, skipping: {} (use --overwrite to redo)", pred)
    else:
        run([str(predict_bin), "-i", str(cbct_dir), "-o", str(pred), "-d", str(cbct_id), "-c", args.configuration,
             "-p", args.plans, "-tr", args.trainer, "-f", str(args.fold), "--disable_tta", "--rec", "mean",
             "-chk", args.checkpoint], env)

    run([py, str(CORE / "denorm_predictions_to_hu.py"), "-i", str(pred), "-g", str(gt_dir), "-o", str(hu),
         "--fingerprint", str(fingerprint)], env)
    run([py, str(CORE / "eval_mae_body_mask.py"), "--pred-dir", str(hu), "--gt-dir", str(gt_dir),
         "--cbct-dir", str(cbct_dir), "--mask-dir", str(mask_dir), "--out", str(metrics),
         "--data-range", str(args.data_range)], env)
    if not args.skip_viz:
        run([py, str(CORE / "viz_sct_results.py"), "--pred-dir", str(hu), "--cbct-dir", str(cbct_dir),
             "--gt-dir", str(gt_dir), "--mask-dir", str(mask_dir), "--metrics", str(metrics),
             "--out-dir", str(viz), "--title", f"{args.trainer} fold {args.fold} {args.split}"], env)

    m = json.loads(metrics.read_text())
    logger.info("{} fold {} {} n={}: MAE {:.2f} PSNR {:.2f} NCC {:.4f} (CBCT MAE {:.2f})",
                args.trainer, args.fold, args.split, m["n"], m["pred_mae_mean"], m["pred_psnr_mean"],
                m["pred_ncc_mean"], m["cbct_mae_mean"])
    logger.info("outputs: {}", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
