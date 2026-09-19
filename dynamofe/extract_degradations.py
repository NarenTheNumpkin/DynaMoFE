"""Extract and cache video degradation signatures across benchmark datasets and codec environments."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

from cods.codecs import prepare_face_image, video_environment
from cted.data import read_video_list
from dynamofe.degradation import DegradationSignatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS = (
    "canonical",
    "jpeg40",
    "webp50",
    "h264_crf35",
    "h265_crf32",
    "resize075_then_h264_crf30",
    "h264_crf30_then_resize075",
)


def extract_ffpp_test_degradations(
    output_path: Path,
    environments: Sequence[str] = ENVIRONMENTS,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"FF++ test degradations already exist: {output_path}")
        return

    config_path = PROJECT_ROOT / "outputs/external_codec_robustness_benchmark_v1/benchmark_configuration.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing benchmark config: {config_path}")

    schedule_cache = PROJECT_ROOT / "outputs/cods_v0_decision_stability_seed42/cache/ffpp_test_sealed_confirmation.pt"
    cache = torch.load(schedule_cache, map_location="cpu", weights_only=False, mmap=True)
    frame_indices = np.asarray(cache["frame_indices"])  # (700, 8, 4)
    paths = list(cache["paths"])
    labels = list(cache["labels"])

    data_root = PROJECT_ROOT / "data/faces/ffpp_c23"
    extractor = DegradationSignatureExtractor(size=128)

    results: dict[str, Any] = {
        "paths": paths,
        "labels": labels,
        "environments": list(environments),
        "feature_names": extractor.FEATURE_NAMES,
        "degradations": {},  # env -> (700, 16) array
    }

    print(f"Extracting degradation features for {len(paths)} FF++ test videos across {len(environments)} environments...")
    start_time = time.perf_counter()

    for env in environments:
        env_feats = []
        t0 = time.perf_counter()
        for idx, rel_path in enumerate(paths):
            vid_frames = []
            for f_idx in frame_indices[idx].reshape(-1):
                img_p = data_root / rel_path / f"{int(f_idx):05d}.jpg"
                with Image.open(img_p) as img:
                    vid_frames.append(prepare_face_image(img, 224))
            canonical = np.stack(vid_frames)
            transformed = video_environment(canonical, env, ffmpeg_binary="/usr/bin/ffmpeg", timeout_seconds=120)
            feat = extractor.extract_from_numpy(transformed)
            env_feats.append(feat)

        env_arr = np.stack(env_feats)  # (700, 16)
        results["degradations"][env] = env_arr
        print(f"  [{env}] done in {time.perf_counter() - t0:.2f}s, mean HF ratio: {env_arr[:, 0].mean():.4f}, mean blockiness: {env_arr[:, 5].mean():.4f}")

    total_time = time.perf_counter() - start_time
    print(f"Extraction completed in {total_time:.2f}s. Saving to {output_path}...")
    torch.save(results, output_path)
    print("Saved successfully.")


def extract_celebdf_degradations(
    output_path: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Celeb-DF degradations already exist: {output_path}")
        return

    list_file = PROJECT_ROOT / "data/lists/celebdf_test.txt"
    data_root = PROJECT_ROOT / "data/faces/celebdf_v2_test"
    extractor = DegradationSignatureExtractor(size=128)

    with open(list_file, "r") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    paths = [parts[0] for parts in lines]
    labels = [int(parts[3]) for parts in lines]

    print(f"Extracting degradation features for {len(paths)} Celeb-DF videos...")
    feats = []
    start_time = time.perf_counter()

    for idx, rel_path in enumerate(paths):
        frame_dir = data_root / rel_path
        frame_files = sorted(frame_dir.glob("*.jpg"))
        if not frame_files:
            # Fallback to zeros if empty
            feats.append(np.zeros(16, dtype=np.float32))
            continue
        # Pick 32 evenly spaced frames
        num_frames = len(frame_files)
        indices = np.linspace(0, num_frames - 1, min(32, num_frames), dtype=int)
        vid_frames = []
        for fi in indices:
            with Image.open(frame_files[fi]) as img:
                vid_frames.append(prepare_face_image(img, 224))
        canonical = np.stack(vid_frames)
        feat = extractor.extract_from_numpy(canonical)
        feats.append(feat)

    feats_arr = np.stack(feats)
    res = {
        "paths": paths,
        "labels": labels,
        "degradations": feats_arr,
        "feature_names": extractor.FEATURE_NAMES,
    }
    torch.save(res, output_path)
    print(f"Celeb-DF extraction done in {time.perf_counter() - start_time:.2f}s. Saved to {output_path}.")


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "outputs/dynamofe/degradations"
    extract_ffpp_test_degradations(out_dir / "ffpp_test_degradations.pt")
    extract_celebdf_degradations(out_dir / "celebdf_degradations.pt")
