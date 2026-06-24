#!/usr/bin/env python3
"""
generate_centroid_fixtures.py — Python-reference fixtures for
Gemma4AssistantMaskedEmbedder (use_ordered_embeddings=true path).

Requires:
  - mlx, mlx-vlm @ d49d428 (same pin as generate_mtp_fixtures.py)
  - E2B assistant checkpoint in HF cache:
      mlx-community/gemma-4-E2B-it-assistant-bf16

Output:
  tools/fixtures/centroid_masked_embedder/case_01.safetensors
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import mlx.core as mx
import numpy as np

SEED = 42
E2B_DRAFTER_ID = "mlx-community/gemma-4-E2B-it-assistant-bf16"
PINNED_MLX_VLM_SHA = "d49d428e9f570dc0387b9598b3b7e0ea391590d2"


def _stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def save_fixture(path: Path, tensors: dict[str, mx.array], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), tensors, metadata=metadata)


def load_e2b_drafter():
    try:
        from mlx_vlm.speculative.drafters import load_drafter
        from mlx_vlm.utils import load_config
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}")

    model_dir = snapshot_download(E2B_DRAFTER_ID)
    config = load_config(model_dir)
    if not config.get("use_ordered_embeddings", False):
        print(
            "WARNING: checkpoint has use_ordered_embeddings=false; "
            "fixtures may not exercise centroid path.",
            file=sys.stderr,
        )
    drafter, _ = load_drafter(E2B_DRAFTER_ID, kind="mtp")
    mx.eval(drafter.parameters())
    return drafter, config, model_dir


def generate_centroid_fixtures(out_root: Path) -> list[dict]:
    model, config, model_dir = load_e2b_drafter()
    masked = model.masked_embedding
    if masked is None:
        sys.exit("E2B drafter has no masked_embedding module — wrong checkpoint?")

    text_cfg = config["text_config"]
    hidden_size = text_cfg["hidden_size"]
    vocab_size = text_cfg["vocab_size"]

    rng = np.random.default_rng(SEED + _stable_hash("centroid_case_01"))
    batch, seq_len = 1, 2
    hidden_states = mx.array(
        rng.standard_normal((batch, seq_len, hidden_size)).astype(np.float32)
    ).astype(mx.bfloat16)

    # lm_head weight comes from tied embed_tokens for E-series assistants.
    lm_head_weight = model.model.embed_tokens.weight

    logits = masked(hidden_states, lm_head_weight)
    mx.eval(logits)

    out_dir = out_root / "centroid_masked_embedder"
    path = out_dir / "case_01.safetensors"
    tensors = {
        "inputs/hidden_states": hidden_states,
        "outputs/logits": logits,
    }
    metadata = {
        "drafterModelId": E2B_DRAFTER_ID,
        "useOrderedEmbeddings": str(config.get("use_ordered_embeddings", False)),
        "hiddenSize": str(hidden_size),
        "vocabSize": str(vocab_size),
        "mlxVlmSha": PINNED_MLX_VLM_SHA,
    }
    save_fixture(path, tensors, metadata)
    print(f"Wrote {path}")
    return [{"path": str(path.relative_to(out_root)), **metadata}]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_root = repo_root / "tools" / "fixtures"
    manifest = generate_centroid_fixtures(out_root)
    manifest_path = out_root / "centroid-fixture-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
