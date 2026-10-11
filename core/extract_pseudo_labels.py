"""Extract 7-class pseudo labels and build the distillation teacher dataset.

TotalSegmentator runs unless --no-totalseg is set. That mode reads labels from
--pseudo-label-dir, or from work_prefix/totalseg/labels7_flat.
--hn adds TotalSegmentator head_muscles and headneck_muscles to class 3 (muscles);
without it class 3 holds only the trunk muscles of `total`, which are nearly empty
on head and neck CT. After planning, intensity mean/std are replaced with the
translation CT fingerprint (dataset cbct_id+1, geometric FOV). AFP applies the
teacher to student sCT and CT, which live in that z-score, so the teacher must
be trained in the same space. Then writes the teacher raw dataset and
preprocesses it for core/train_teacher.py.
"""
import argparse
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

HN_MUSCLE_TASKS = ["head_muscles", "headneck_muscles"]


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    import json

    path.write_text(json.dumps(payload, indent=2) + "\n")


def pin_teacher_intensity(teacher_pp: Path, src_fp: Path, plans_name: str = "nnUNetPlans") -> None:
    """Overwrite teacher mean/std with the student translation fingerprint.

    plan_and_preprocess would otherwise compute stats on the 7-class labels.
    AFP sees student CT / sCT, so the teacher must use that same z-score.
    """
    if not src_fp.is_file():
        raise SystemExit(f"missing student fingerprint {src_fp}. Run core/prepare_training_data.py first.")
    props = _load_json(src_fp)["foreground_intensity_properties_per_channel"]
    dst_fp = teacher_pp / "dataset_fingerprint.json"
    plans_path = teacher_pp / f"{plans_name}.json"
    for path in (dst_fp, plans_path):
        if not path.is_file():
            raise SystemExit(f"missing {path} after plan_and_preprocess")
        data = _load_json(path)
        data["foreground_intensity_properties_per_channel"] = props
        _write_json(path, data)
    ch = props["0"]
    logger.info(
        "teacher intensity pinned from {}  mean={:.3f} std={:.3f}",
        src_fp, float(ch["mean"]), float(ch["std"]),
    )


def nnunet_env(cfg: dict) -> dict[str, str]:
    apply_dir_env(cfg)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    logger.info("+ {}", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def totalseg_python() -> Path:
    py = REPO / "totalseg" / ".venv" / "bin" / "python"
    if not py.is_file():
        raise SystemExit("TotalSegmentator environment is missing. From the repository root: uv sync --project totalseg")
    return py


def totalseg_env() -> dict[str, str]:
    env = os.environ.copy()
    # Keep this fork's nnunetv2 off the path. TotalSegmentator needs the stock package.
    env.pop("PYTHONPATH", None)
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        env.pop(key, None)
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO / "config.yaml")
    ap.add_argument("--no-totalseg", action="store_true", help="Do not run TotalSegmentator")
    ap.add_argument("--pseudo-label-dir", type=Path, help="7-class labels when TotalSegmentator is off")
    ap.add_argument("--hn", action="store_true",
                    help=f"Add TotalSegmentator {' and '.join(HN_MUSCLE_TASKS)} to the muscles class")
    ap.add_argument(
        "--fingerprint-from",
        type=int,
        default=None,
        help="Dataset id whose FOV fingerprint the teacher must use. "
        "Default: translation CT (cbct_dataset_id+1). AFP sees that z-score, not dataset 93 (CBCT).",
    )
    ap.add_argument("--device", default="gpu:0", help="TotalSegmentator device: gpu:0 or cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    use_totalseg = not args.no_totalseg
    work = resolve_cfg(cfg, "work_prefix")
    raw = resolve_cfg(cfg, "nnunet_raw")
    name = str(cfg["dataset_name"])
    cbct_id = int(cfg["cbct_dataset_id"])
    ct_id = cbct_id + 1
    teacher_id = int(cfg["teacher_dataset_id"])
    if teacher_id in (cbct_id, ct_id):
        raise SystemExit("teacher_dataset_id must differ from the CBCT id and the CBCT id + 1")
    ct_images = raw / f"Dataset{ct_id:03d}_{name}_CT" / "imagesTr"
    if not ct_images.is_dir():
        raise SystemExit(f"missing {ct_images}. Run core/prepare_training_data.py through pair first.")
    labels_out = work / "totalseg" / "labels7_flat"
    supplied = args.pseudo_label_dir.resolve() if args.pseudo_label_dir else None
    py = sys.executable
    env = nnunet_env(cfg)
    workers = str(args.workers)

    if use_totalseg:
        if shutil.which("nvidia-smi") is None:
            raise SystemExit("nvidia-smi not found. Stopped before TotalSegmentator.")
        tasks = list(HN_MUSCLE_TASKS) if args.hn else []
        if tasks:
            logger.info("muscles class = total trunk muscles + {}", " + ".join(tasks))
        else:
            logger.info("head/neck muscle labels off (pass --hn to add them)")
        run(
            [
                str(totalseg_python()), str(CORE / "infer_7labels.py"),
                "--input", str(ct_images),
                "--work", str(work / "totalseg"),
                "--labels-out", str(labels_out),
                "--device", args.device,
                "--muscle-tasks", ",".join(tasks),
            ],
            totalseg_env(),
            cwd=REPO / "totalseg",
        )
        seg_dir = labels_out
    else:
        seg_dir = supplied or labels_out
        logger.info("TotalSegmentator off. Labels from {}", seg_dir)
        if not seg_dir.is_dir() or not any(seg_dir.glob("*.nii.gz")):
            raise SystemExit(f"missing pseudo labels in {seg_dir}. Run TotalSegmentator or pass --pseudo-label-dir.")

    teacher_raw = raw / f"Dataset{teacher_id:03d}_{name}_CT_7labels"
    cmd = [
        py, str(CORE / "build_teacher_raw.py"),
        "--ct-dir", str(ct_images),
        "--seg-dir", str(seg_dir),
        "--out", str(teacher_raw),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run(cmd)
    run(
        [
            "nnUNetv2_plan_and_preprocess", "-d", str(teacher_id),
            "-c", "3d_fullres", "--clean", "-np", workers, "-npfp", workers,
        ],
        env,
    )
    teacher_pp = resolve_cfg(cfg, "nnunet_preprocessed") / f"Dataset{teacher_id:03d}_{name}_CT_7labels"
    fp_id = int(args.fingerprint_from) if args.fingerprint_from is not None else ct_id
    if fp_id == cbct_id:
        src_fp = resolve_cfg(cfg, "nnunet_preprocessed") / f"Dataset{cbct_id:03d}_{name}_CBCT" / "dataset_fingerprint.json"
    elif fp_id == ct_id:
        src_fp = resolve_cfg(cfg, "nnunet_preprocessed") / f"Dataset{ct_id:03d}_{name}_CT" / "dataset_fingerprint.json"
    else:
        raise SystemExit(f"--fingerprint-from {fp_id} must be {cbct_id} (CBCT) or {ct_id} (CT)")
    pin_teacher_intensity(teacher_pp, src_fp)
    run(
        [
            "nnUNetv2_preprocess", "-d", str(teacher_id),
            "-c", "3d_fullres", "-np", workers,
        ],
        env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
