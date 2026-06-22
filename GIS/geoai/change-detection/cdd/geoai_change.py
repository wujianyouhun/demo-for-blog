"""GeoAI ChangeStar integration for building change detection."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


DEFAULT_GEOAI_MODEL = "s1_s1c1_vitb"


def list_geoai_changestar_models() -> Dict[str, str]:
    """Return available GeoAI ChangeStar model aliases."""
    try:
        import geoai

        return geoai.list_changestar_models()
    except Exception:
        return {
            DEFAULT_GEOAI_MODEL: "GeoAI ChangeStar default model",
        }


def run_geoai_change_detection(
    image_a: str,
    image_b: str,
    output_dir: str | Path,
    model_name: str = DEFAULT_GEOAI_MODEL,
    tile_size: int = 1024,
    overlap: int = 64,
    threshold: float = 0.5,
    device: Optional[str] = None,
    visualize: bool = True,
) -> Dict[str, object]:
    """Run GeoAI ChangeStar and write raster/vector/preview outputs.

    GeoAI ChangeStar is the default demo path because it can run with a
    pretrained model and does not require the user to train before inference.
    """
    import geoai

    from cdd.visualize import ChangeVisualizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = f"{Path(image_a).stem}_vs_{Path(image_b).stem}_geoai"
    change_path = output_dir / f"{name}_change.tif"
    vector_path = output_dir / f"{name}_change.gpkg"
    t1_path = output_dir / f"{name}_t1_semantic.tif"
    t2_path = output_dir / f"{name}_t2_semantic.tif"

    result = geoai.changestar_detect(
        image1_path=str(image_a),
        image2_path=str(image_b),
        model_name=model_name,
        output_change=str(change_path),
        output_t1_semantic=str(t1_path),
        output_t2_semantic=str(t2_path),
        output_vector=str(vector_path),
        tile_size=tile_size,
        overlap=overlap,
        threshold=threshold,
        device=device,
    )

    preview_paths = {}
    if visualize:
        overlay = output_dir / f"{name}_overlay.png"
        side = output_dir / f"{name}_side_by_side.png"
        ChangeVisualizer.create_change_overlay(
            image_b,
            change_path,
            output_path=overlay,
            color=(255, 0, 0),
            opacity=0.55,
        )
        ChangeVisualizer.create_comparison_image(
            image_a,
            image_b,
            change_map_path=change_path,
            output_path=side,
            mode="side_by_side",
        )
        preview_paths = {"overlay": str(overlay), "side_by_side": str(side)}

    change_map = result.get("change_map")
    changed_pixels = int(np.sum(change_map > 0)) if change_map is not None else None

    return {
        "engine": "geoai",
        "model_name": model_name,
        "mask": str(change_path),
        "vectors": str(vector_path),
        "t1_semantic": str(t1_path),
        "t2_semantic": str(t2_path),
        "previews": preview_paths,
        "changed_pixels": changed_pixels,
    }
