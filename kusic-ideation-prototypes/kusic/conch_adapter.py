"""Optional offline CONCH image adapter for K02. Author: Kevin-C.

The caller must be authorized to process every supplied image as a research
cohort. K02 eligibility metadata controls retrieval, not authorization to embed
the supplied cohort. This adapter imports no optional packages until called.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import re
import threading
from importlib import metadata as package_metadata
from pathlib import Path

from . import retrieval

_MAX_IMAGES = 16
_MAX_SIDE = 4096
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_LOCK = threading.Lock()
_CONFIG = "conch_ViT-B-16"


def _path_text(value: object, field: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{field} must be a local file path")
    text = os.fspath(value)
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError(f"{field} must be a nonempty, unpadded local file path")
    remainder = text[2:] if re.match(r"^[A-Za-z]:[\\/]", text) else text
    if text.startswith(("\\\\", "//")) or ":" in remainder or any(ord(char) < 32 for char in text):
        raise ValueError(
            f"{field} must be a local file path; URLs and network paths are unsupported"
        )
    return text


def _screen_links(path: Path, field: str, depth: int = 0) -> None:
    if depth > 16:
        raise ValueError(f"{field} has too many symbolic link or junction targets")
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            target = Path(_path_text(os.readlink(cursor), field))
            if not target.is_absolute():
                target = cursor.parent / target
            _screen_links(target, field, depth + 1)


def _local_path(value: object, field: str, base_dir: Path | None = None) -> Path:
    text = _path_text(value, field)
    try:
        path = Path(text)
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        _screen_links(path, field)
        path = path.resolve(strict=True)
        _path_text(str(path), field)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{field} does not identify an accessible local path") from exc
    return path


def _local_file(value: object, field: str, base_dir: Path | None = None) -> Path:
    path = _local_path(value, field, base_dir)
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{field} must identify a nonempty regular file")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{field} does not identify an accessible local file") from exc
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare(payload: dict, base_dir: Path | None = None) -> tuple[dict, list[Path]]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    if payload.keys() - {"query", "records", "top_k", "purpose"}:
        raise ValueError("payload has unknown fields")
    if not isinstance(payload.get("query"), dict) or not isinstance(payload.get("records"), list):
        raise ValueError("payload requires a query object and records array")
    if len(payload["records"]) + 1 > _MAX_IMAGES:
        raise ValueError(f"At most {_MAX_IMAGES} total query and record images are supported")
    request = copy.deepcopy(payload)
    items = [request["query"], *request["records"]]
    paths = []
    for index, item in enumerate(items):
        field = "query" if index == 0 else f"records[{index - 1}]"
        if not isinstance(item, dict):
            raise ValueError(f"{field} must be an object")
        allowed = (
            {"patient_id", "institution", "as_of", "authorized_patient_ids", "image_path"}
            if index == 0
            else {
                "record_id",
                "patient_id",
                "institution",
                "observed_at",
                "source_ref",
                "permitted_use",
                "consent",
                "image_path",
            }
        )
        if "embedding" in item:
            raise ValueError(f"{field} must supply image_path instead of embedding")
        if item.keys() - allowed:
            raise ValueError(f"{field} has unknown fields")
        paths.append(_local_file(item.pop("image_path", None), f"{field}.image_path", base_dir))
        item["embedding"] = [1.0]
    # This validates dates, identity ambiguities, consent types and all metadata
    # before importing a model runtime or reading checkpoint contents.
    retrieval._validate(request)
    return request, paths


def _image_module():
    try:
        return importlib.import_module("PIL.Image")
    except (ImportError, OSError) as exc:
        raise ValueError("Image embedding requires Pillow in the selected Python runtime") from exc


def _inspect_image(path: Path, image_module, index: int) -> dict:
    if path.stat().st_size > _MAX_IMAGE_BYTES:
        raise ValueError(f"Image {index + 1} exceeds the 64 MiB compressed-file limit")
    try:
        with image_module.open(path) as image:
            width, height = image.size
            if not 1 <= width <= _MAX_SIDE or not 1 <= height <= _MAX_SIDE:
                raise ValueError(f"Image {index + 1} exceeds the 4096 by 4096 decoded-image limit")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Image {index + 1} must contain exactly one frame")
            # Decode before model construction. The image is closed immediately.
            image.load()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Image {index + 1} is not a supported decodable image") from exc
    return {"image": index + 1, "sha256": _sha256(path), "width": width, "height": height}


def _safe_state(torch, weights: Path) -> tuple[dict, bool]:
    kwargs = {"map_location": "cpu", "weights_only": True, "mmap": True}
    mmap_used = True
    try:
        checkpoint = torch.load(str(weights), **kwargs)
    except (TypeError, RuntimeError) as exc:
        message = str(exc).lower()
        if "mmap" not in message or "weights_only" in message:
            raise ValueError(
                "Checkpoint requires a compatible PyTorch weights_only loader"
            ) from exc
        # Older PyTorch/legacy archives may not support mmap. Never relax
        # weights_only=True, including on the retry.
        kwargs.pop("mmap")
        mmap_used = False
        try:
            checkpoint = torch.load(str(weights), **kwargs)
        except Exception as retry:
            raise ValueError("Could not load this checkpoint with weights_only=True") from retry
    except Exception as exc:
        raise ValueError("Could not load this checkpoint with weights_only=True") from exc
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint must contain a nonempty tensor state dictionary")
    visual = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            raise ValueError("Checkpoint state keys must be strings")
        key = key.removeprefix("module.")
        if key.startswith("visual."):
            key = key.removeprefix("visual.")
            if key in visual or not isinstance(value, torch.Tensor):
                raise ValueError("Checkpoint visual state contains duplicate keys or non-tensors")
            visual[key] = value
    if not visual:
        raise ValueError("Checkpoint has no CONCH visual.* tensor weights")
    return visual, mmap_used


def _load_visual(weights: Path):
    try:
        torch = importlib.import_module("torch")
        coca = importlib.import_module("conch.open_clip_custom.coca_model")
        transforms = importlib.import_module("conch.open_clip_custom.transform")
        constants = importlib.import_module("conch.open_clip_custom.constants")
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        raise ValueError(
            "Optional CONCH runtime is missing or incompatible. Select a Python runtime "
            "with PyTorch, torchvision, timm, Pillow and the reviewed official CONCH source."
        ) from exc
    config_path = Path(coca.__file__).parent / "model_configs" / f"{_CONFIG}.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["embed_dim"] != 512 or config["vision_cfg"]["image_size"] != 448:
            raise ValueError("CONCH runtime has an unsupported model configuration")
        # Construct only the official visual tower: no random text/decoder model,
        # factory checkpoint loader, pretrained initialization or hub call.
        with torch.device("cpu"):
            model = coca._build_vision_tower(
                config["embed_dim"], config["vision_cfg"], config["embed_dim_caption"]
            )
        state, mmap_used = _safe_state(torch, weights)
        model.load_state_dict(state, strict=True)
        del state
        model.to(device="cpu")
        model.eval()
        preprocess = transforms.image_transform(
            config["vision_cfg"]["image_size"],
            mean=constants.OPENAI_DATASET_MEAN,
            std=constants.OPENAI_DATASET_STD,
        )
    except (OSError, KeyError, TypeError, RuntimeError) as exc:
        raise ValueError(
            "CONCH visual weights/runtime do not match the supported configuration"
        ) from exc
    metadata = {
        "torch_version": str(torch.__version__),
        "conch_source": "https://github.com/mahmoodlab/CONCH",
        "model_config": _CONFIG,
        "config_sha256": _sha256(config_path),
        "mmap": mmap_used,
        "visual_state_load": "strict",
        "projection": "official contrastive head, L2 normalized",
        "feature_path": "projected_pool_normalized",
    }
    for name in ("timm", "torchvision"):
        try:
            metadata[f"{name}_version"] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            metadata[f"{name}_version"] = None
    metadata["source_sha256"] = {}
    for name, module in (("coca_model", coca), ("transform", transforms), ("constants", constants)):
        try:
            metadata["source_sha256"][name] = _sha256(Path(module.__file__))
        except (AttributeError, OSError):
            metadata["source_sha256"][name] = None
    try:
        distribution = package_metadata.distribution("conch")
        metadata["conch_version"] = distribution.version
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        revision = direct_url.get("vcs_info", {}).get("commit_id")
        if revision is None:
            match = re.search(r"/([0-9a-f]{40})\.zip(?:$|[?#])", direct_url.get("url", ""))
            revision = match.group(1) if match else None
        metadata["source_revision"] = revision
    except (package_metadata.PackageNotFoundError, ValueError, OSError):
        metadata["conch_version"] = None
        metadata["source_revision"] = None
    return torch, model, preprocess, metadata


def _encode(path: Path, image_module, torch, model, preprocess) -> list[float]:
    with image_module.open(path) as image:
        if (
            any(not 1 <= side <= _MAX_SIDE for side in image.size)
            or getattr(image, "n_frames", 1) != 1
        ):
            raise ValueError("An image changed to an unsupported size or frame count")
        # Preserve the publisher's resize/crop/convert order for non-RGB modes.
        tensor = preprocess(image).unsqueeze(0).to(device="cpu")
    latent, _ = model(tensor)
    latent = torch.nn.functional.normalize(latent, dim=-1)
    if tuple(latent.shape) != (1, 512):
        raise ValueError("CONCH returned an unexpected image embedding dimension")
    values = latent[0].detach().cpu().tolist()
    if any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in values):
        raise ValueError("CONCH returned a nonfinite image embedding")
    return retrieval._vector(values, "CONCH image embedding", 512)


def embed_priors(
    payload: dict, weights_path: str | os.PathLike, base_dir: str | os.PathLike | None = None
) -> dict:
    """Embed an explicitly authorized local cohort and return a K02-ready request.

    Use the usual K02 metadata, replacing each query/record ``embedding`` with
    ``image_path``. This processes every supplied image, including records that
    K02 later excludes. It never downloads resources, alters images, trains, or
    uses a GPU. Provenance hashes are identifiers, not anonymization.
    """
    if base_dir is not None:
        try:
            base_dir = _local_path(base_dir, "base_dir")
            if not base_dir.is_dir():
                raise ValueError("base_dir must identify an existing local directory")
        except (OSError, TypeError, RuntimeError) as exc:
            raise ValueError("base_dir must identify an existing local directory") from exc
    request, paths = _prepare(payload, base_dir)
    weights = _local_file(weights_path, "weights_path")
    image_module = _image_module()
    images = [_inspect_image(path, image_module, i) for i, path in enumerate(paths)]
    weights_hash = _sha256(weights)
    with _LOCK:
        # Keep the process-wide CPU-thread change scoped to this operation.
        try:
            torch = importlib.import_module("torch")
        except (ImportError, OSError) as exc:
            raise ValueError(
                "Image embedding requires PyTorch in the selected Python runtime"
            ) from exc
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            torch, model, preprocess, runtime = _load_visual(weights)
            items = [request["query"], *request["records"]]
            with torch.inference_mode():
                for item, path, image in zip(items, paths, images, strict=True):
                    if _sha256(path) != image["sha256"]:
                        raise ValueError(
                            "An image changed after validation; retry with stable local files"
                        )
                    item["embedding"] = _encode(path, image_module, torch, model, preprocess)
            retrieval._validate(request)
        finally:
            torch.set_num_threads(previous_threads)
    return {
        "request": request,
        "provenance": {
            "author": "Kevin-C",
            "weights_sha256": weights_hash,
            "weights_only": True,
            "device": "cpu",
            "threads": 2,
            "batch_size": 1,
            "image_count": len(images),
            "embedding_dimensions": 512,
            "runtime": runtime,
            "images": images,
            "scope": "Caller-authorized offline research cohort; no clinical effectiveness claim",
        },
    }
