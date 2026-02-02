#!/usr/bin/env python3
"""
Standalone validation script for InsightFace ArcFace Torch .bin targets.

- ONLY runs validation on the provided --val-targets list
- Loads a backbone from insightface/recognition/arcface_torch/backbones
- Computes embeddings for each target's {name}.bin under --data-dir
- Reports best accuracy over thresholds (simple protocol, as in the provided code)

Example:
  python validate_val_targets_only.py \
    --repo-root /path/to/insightface/recognition/arcface_torch \
    --weights /path/to/model.pt \
    --data-dir /path/to/faces_emore \
    --network r100 \
    --val-targets lfw cfp_fp agedb_30

Notes:
- Expects validation bins at: <data-dir>/<target>.bin
- Requires: torch, numpy, scikit-learn
- Decoder: tries mxnet first; if unavailable falls back to opencv-python (cv2).
"""

import argparse
import os
import sys
import pickle
import logging

import numpy as np
import torch

# NumPy compatibility shims (matches your training script intent)
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.str = np.str_

import sklearn
import sklearn.preprocessing


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _decode_image(bin_bytes):
    """
    Decode raw encoded image bytes into RGB uint8 HWC.
    Tries mxnet first (often used by InsightFace), else OpenCV.
    """
    try:
        import mxnet as mx  # type: ignore
        img = mx.image.imdecode(bin_bytes).asnumpy()
        # mxnet returns RGB already typically; keep as-is.
        return img
    except Exception:
        import cv2  # type: ignore
        img = cv2.imdecode(np.frombuffer(bin_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed (got None). Is the bin data valid?")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def load_bin(path, image_size=(112, 112)):
    """
    Load InsightFace .bin verification dataset.
    Returns:
      data_list: [no_flip_tensor, flip_tensor], each shaped (2*N, 3, H, W)
      issame_list: list[bool] length N
    """
    with open(path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")

    n = len(issame_list)
    # There are 2*n images in bins (pairwise)
    data_list = []
    for _flip in [0, 1]:
        data_list.append(torch.empty((n * 2, 3, image_size[0], image_size[1]), dtype=torch.float32))

    for idx in range(n * 2):
        img = _decode_image(bins[idx])  # HWC RGB uint8
        if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
            # Most insightface bins are already 112x112, but be defensive.
            # Resize only if needed.
            try:
                import cv2  # type: ignore
                img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                raise RuntimeError(
                    f"Image size is {img.shape[:2]} but expected {image_size}, and cv2 resize failed: {e}"
                )

        # Convert to CHW float and normalize to [-1, 1]
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img_t = (img_t / 255.0 - 0.5) / 0.5

        # no flip
        data_list[0][idx] = img_t
        # flip horizontally (width dimension in CHW is dim=2)
        data_list[1][idx] = torch.flip(img_t, dims=[2])

    return data_list, issame_list


def evaluate_accuracy(embeddings, issame_list):
    """
    Simple best-threshold accuracy over squared L2 distances.
    Mirrors your provided code (not full 10-fold protocol).
    """
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), axis=1)

    thresholds = np.arange(0, 4, 0.01)
    issame = np.array(issame_list, dtype=np.bool_)

    accuracy_list = []
    for thr in thresholds:
        predict_issame = np.less(dist, thr)
        accuracy_list.append(np.mean(predict_issame == issame))

    best_idx = int(np.argmax(accuracy_list))
    return float(accuracy_list[best_idx]), float(thresholds[best_idx])


@torch.no_grad()
def infer_embeddings(backbone, data_tensor, batch_size, device):
    """
    data_tensor: (N, 3, H, W) on CPU
    returns: (N, embedding_dim) numpy float32
    """
    backbone.eval()
    embs = []
    for i in range(0, len(data_tensor), batch_size):
        batch = data_tensor[i : i + batch_size].to(device, non_blocking=True)
        out = backbone(batch)
        out = out.detach().float().cpu().numpy()
        embs.append(out)
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def validate(backbone, data_dir, val_targets, batch_size=64, image_size=(112, 112), device="cuda"):
    """
    Validate ONLY on the given val_targets.
    Looks for <data_dir>/<target>.bin
    """
    results = {}
    for name in val_targets:
        bin_path = os.path.join(data_dir, f"{name}.bin")
        if not os.path.exists(bin_path):
            logging.warning(f"[Val] {name}: NOT FOUND at {bin_path}")
            continue

        try:
            data_list, issame_list = load_bin(bin_path, image_size=image_size)

            # two passes: no-flip and flip; then sum as in your training code
            emb0 = infer_embeddings(backbone, data_list[0], batch_size, device=device)
            emb1 = infer_embeddings(backbone, data_list[1], batch_size, device=device)

            embeddings = emb0 + emb1
            embeddings = sklearn.preprocessing.normalize(embeddings)

            acc, thr = evaluate_accuracy(embeddings, issame_list)
            results[name] = {"acc": acc, "threshold": thr}
            logging.info(f"[Val] {name}: {acc*100:.2f}% (best_thr={thr:.2f})")
        except Exception as e:
            logging.exception(f"[Val] {name}: ERROR: {e}")

    return results


def load_backbone(repo_root, network, embedding_size, fp16, weights_path, device):
    """
    Loads insightface arcface_torch backbone via backbones.get_model.
    repo_root should be: .../insightface/recognition/arcface_torch
    """
    sys.path.insert(0, repo_root)

    try:
        from backbones import get_model  # from arcface_torch
    except Exception as e:
        raise RuntimeError(
            f"Failed to import backbones.get_model from repo_root={repo_root}. "
            f"Make sure repo_root points to recognition/arcface_torch. Original error: {e}"
        )

    backbone = get_model(network, dropout=0.0, fp16=fp16, num_features=embedding_size).to(device)
    backbone.eval()

    if weights_path is None:
        raise ValueError("--weights is required (path to model .pt state_dict).")

    state = torch.load(weights_path, map_location="cpu")
    # Accept either raw state_dict or a checkpoint dict containing state_dict_backbone
    if isinstance(state, dict) and "state_dict_backbone" in state:
        state_dict = state["state_dict_backbone"]
    else:
        state_dict = state

    # If someone saved a DDP-wrapped state dict with 'module.' prefix, strip it.
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"Missing keys when loading weights (showing up to 20): {missing[:20]}")
    if unexpected:
        logging.warning(f"Unexpected keys when loading weights (showing up to 20): {unexpected[:20]}")

    return backbone


def parse_args():
    p = argparse.ArgumentParser(description="Standalone validator (val_targets only) for arcface_torch bins")
    p.add_argument("--repo-root", type=str, required=True,
                   help="Path to insightface/recognition/arcface_torch (so imports like backbones work).")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to backbone weights (.pt). Can be raw state_dict or a checkpoint with state_dict_backbone.")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory containing <target>.bin files (e.g., faces_emore root used by training).")
    p.add_argument("--val-targets", nargs="+", required=True,
                   help="Validation targets to run, e.g. lfw cfp_fp agedb_30 calfw cfp_ff cplfw")
    p.add_argument("--network", type=str, default="r100", help="Backbone name (e.g., r100, r50, etc.).")
    p.add_argument("--embedding-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--fp16", action="store_true", help="Construct backbone in fp16 mode (as in arcface_torch).")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--image-size", type=int, nargs=2, default=[112, 112], help="H W, usually 112 112")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    backbone = load_backbone(
        repo_root=args.repo_root,
        network=args.network,
        embedding_size=args.embedding_size,
        fp16=args.fp16,
        weights_path=args.weights,
        device=args.device,
    )

    results = validate(
        backbone=backbone,
        data_dir=args.data_dir,
        val_targets=args.val_targets,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        device=args.device,
    )

    # Print a compact summary (still only val_targets)
    if results:
        logging.info("=== Summary (val_targets only) ===")
        for k in args.val_targets:
            if k in results:
                logging.info(f"{k}: {results[k]['acc']*100:.2f}% (thr={results[k]['threshold']:.2f})")


main()