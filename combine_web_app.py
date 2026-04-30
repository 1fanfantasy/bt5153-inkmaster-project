from __future__ import annotations

import importlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "calli-kaggle" / "data" / "data"
SUMMARY_PATH = ROOT / "calli-kaggle" / "Summary.csv"
MODEL_DIR = ROOT / "vit_calligraphy_best"
TEMP_ROOT = ROOT / "temp_pic"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class RuntimeDeps:
    cv2: Any
    torch: Any
    skeletonize: Any
    distance_transform_edt: Any
    transformers: Any


@dataclass(frozen=True)
class CamDeps:
    GradCAMPlusPlus: Any
    show_cam_on_image: Any
    ClassifierOutputTarget: Any


def normalize_path(path_like: str | Path) -> Path:
    return Path(str(path_like).replace("\\", "/"))


def resolve_workspace_path(path_str: str) -> Path:
    path = normalize_path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


@lru_cache(maxsize=1)
def get_runtime_deps() -> RuntimeDeps:
    missing: list[str] = []

    def _load(module_name: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
            return None

    cv2 = _load("cv2")
    torch = _load("torch")
    morphology = _load("skimage.morphology")
    ndimage = _load("scipy.ndimage")
    transformers = _load("transformers")

    if missing:
        pkg_hint = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            "Missing runtime dependencies: "
            f"{pkg_hint}. Activate the same environment used by combine.ipynb before running this app."
        )

    return RuntimeDeps(
        cv2=cv2,
        torch=torch,
        skeletonize=morphology.skeletonize,
        distance_transform_edt=ndimage.distance_transform_edt,
        transformers=transformers,
    )


@lru_cache(maxsize=1)
def get_cam_deps() -> CamDeps:
    try:
        cam_module = importlib.import_module("pytorch_grad_cam")
        image_module = importlib.import_module("pytorch_grad_cam.utils.image")
        target_module = importlib.import_module("pytorch_grad_cam.utils.model_targets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pytorch-grad-cam is not available in the current environment."
        ) from exc

    return CamDeps(
        GradCAMPlusPlus=cam_module.GradCAMPlusPlus,
        show_cam_on_image=image_module.show_cam_on_image,
        ClassifierOutputTarget=target_module.ClassifierOutputTarget,
    )


@st.cache_data(show_spinner=False)
def load_summary() -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    summary_df = pd.read_csv(SUMMARY_PATH)
    summary_df = summary_df[["Label", "Calligrapher Name"]].dropna()
    label_to_master = dict(zip(summary_df["Label"], summary_df["Calligrapher Name"]))
    master_to_label = dict(zip(summary_df["Calligrapher Name"], summary_df["Label"]))
    return summary_df, label_to_master, master_to_label


@st.cache_data(show_spinner=False)
def load_train_classes() -> list[str]:
    train_dir = DATA_ROOT / "train"
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])


def resolve_embedding_file(root_dir: Path, style_name: str, suffix: str) -> Path | None:
    candidates = [
        root_dir / f"{style_name}{suffix}",
        root_dir / style_name / f"{style_name}{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = list(root_dir.rglob(f"{style_name}{suffix}"))
    return found[0] if found else None


@st.cache_data(show_spinner=False)
def scan_style_assets() -> pd.DataFrame:
    summary_df, _, _ = load_summary()
    rows = []
    for _, row in summary_df.iterrows():
        label = row["Label"]
        pseudo_path = ROOT / f"{label}_pseudo_labels.csv"
        train_pt = resolve_embedding_file(ROOT / "embedding" / "train", label, ".pt")
        test_pt = resolve_embedding_file(ROOT / "embedding" / "test", label, ".pt")
        train_json = resolve_embedding_file(ROOT / "embedding" / "train", label, "_paths.json")
        test_json = resolve_embedding_file(ROOT / "embedding" / "test", label, "_paths.json")
        rows.append(
            {
                "label": label,
                "calligrapher": row["Calligrapher Name"],
                "pseudo_csv": pseudo_path.exists(),
                "embedding_ready": all([train_pt, test_pt, train_json, test_json]),
            }
        )
    return pd.DataFrame(rows)


def style_option_map() -> dict[str, str]:
    asset_df = scan_style_assets()
    mapping: dict[str, str] = {}
    for _, row in asset_df.iterrows():
        if row["embedding_ready"]:
            status = "full pipeline"
        elif row["pseudo_csv"]:
            status = "match only"
        else:
            status = "classification only"
        label = row["label"]
        calligrapher = row["calligrapher"]
        mapping[f"{label} | {calligrapher} | {status}"] = label
    return mapping


def infer_device(torch_module: Any) -> str:
    has_mps = hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
    if has_mps:
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource(show_spinner=False)
def load_vit_resources() -> tuple[Any, Any, str]:
    deps = get_runtime_deps()
    processor = deps.transformers.ViTImageProcessor.from_pretrained(MODEL_DIR)
    model = deps.transformers.ViTForImageClassification.from_pretrained(MODEL_DIR)
    device = infer_device(deps.torch)
    model.to(device)
    model.eval()
    return processor, model, device


@st.cache_resource(show_spinner=False)
def load_qwen_ocr_resources() -> tuple[Any, Any, str]:
    deps = get_runtime_deps()
    processor = deps.transformers.AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = deps.transformers.AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    device = infer_device(deps.torch)
    model.to(device)
    model.eval()
    return processor, model, device


@st.cache_data(show_spinner=False)
def load_pseudo_df(style_label: str) -> pd.DataFrame:
    pseudo_path = ROOT / f"{style_label}_pseudo_labels.csv"
    if not pseudo_path.exists():
        raise FileNotFoundError(f"Pseudo label file not found: {pseudo_path.name}")
    return pd.read_csv(pseudo_path)


@st.cache_resource(show_spinner=False)
def load_embedding_results(style_label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    deps = get_runtime_deps()
    train_root = ROOT / "embedding" / "train"
    test_root = ROOT / "embedding" / "test"

    train_pt = resolve_embedding_file(train_root, style_label, ".pt")
    test_pt = resolve_embedding_file(test_root, style_label, ".pt")
    train_json = resolve_embedding_file(train_root, style_label, "_paths.json")
    test_json = resolve_embedding_file(test_root, style_label, "_paths.json")

    missing = [
        name
        for name, path in [
            ("train_pt", train_pt),
            ("test_pt", test_pt),
            ("train_paths_json", train_json),
            ("test_paths_json", test_json),
        ]
        if path is None
    ]
    if missing:
        raise FileNotFoundError(
            f"Embedding assets missing for style={style_label}: {', '.join(missing)}"
        )

    train_emb = deps.torch.load(train_pt, map_location="cpu")
    test_emb = deps.torch.load(test_pt, map_location="cpu")

    with open(train_json, "r", encoding="utf-8") as file_obj:
        train_paths = json.load(file_obj)
    with open(test_json, "r", encoding="utf-8") as file_obj:
        test_paths = json.load(file_obj)

    return (
        {"image_paths": train_paths, "embeddings": train_emb},
        {"image_paths": test_paths, "embeddings": test_emb},
    )


def build_prediction_figure(labels: list[str], probs: list[float]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    plt.rcParams["font.sans-serif"] = ["Songti SC", "SimHei", "Arial Unicode MS", "DejaVu Sans","SimSong"]
    plt.rcParams["axes.unicode_minus"] = False
    axis.pie(probs, labels=labels, autopct="%1.1f%%", startangle=140)
    axis.set_title("Top style probabilities")
    return figure


def predict_style_distribution(image_path: Path, top_n: int = 6) -> dict[str, Any]:
    deps = get_runtime_deps()
    processor, model, device = load_vit_resources()
    _, label_to_master, _ = load_summary()
    train_classes = load_train_classes()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with deps.torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = deps.torch.softmax(logits, dim=1).squeeze(0).cpu().numpy() * 100

    pred_idx = int(np.argmax(probs))
    pred_label = train_classes[pred_idx]

    top_indices = probs.argsort()[-top_n:][::-1]
    top_labels = [label_to_master.get(train_classes[idx], train_classes[idx]) for idx in top_indices]
    top_probs = [float(probs[idx]) for idx in top_indices]

    if sum(top_probs) < 100:
        top_labels = top_labels + ["Others"]
        top_probs = top_probs + [max(0.0, 100.0 - sum(top_probs))]

    top_three = []
    for idx in top_indices[:3]:
        label = train_classes[idx]
        top_three.append(
            {
                "style_label": label,
                "calligrapher": label_to_master.get(label, label),
                "probability": float(probs[idx]),
            }
        )

    return {
        "predicted_label": pred_label,
        "predicted_calligrapher": label_to_master.get(pred_label, pred_label),
        "top_three": top_three,
        "figure": build_prediction_figure(top_labels, top_probs),
    }


def reshape_transform(tensor: Any) -> Any:
    tokens = tensor[:, 1:, :]
    token_count = tokens.shape[1]
    side = int(np.sqrt(token_count))
    if side * side != token_count:
        raise ValueError(f"Unexpected token count: {token_count}")
    return tokens.reshape(tensor.size(0), side, side, tensor.size(2)).permute(0, 3, 1, 2)


def resolve_vit_layer(model: Any, layer_idx: int, norm_pos: str = "after") -> Any:
    layer = model.vit.encoder.layer[layer_idx]
    if norm_pos == "after":
        return layer.layernorm_after
    if norm_pos == "before":
        return layer.layernorm_before
    raise ValueError("norm_pos must be 'after' or 'before'")


def generate_attention_overlay(image_path: Path) -> np.ndarray:
    deps = get_runtime_deps()
    cam_deps = get_cam_deps()
    processor, model, device = load_vit_resources()

    class ViTLogitsWrapper(deps.torch.nn.Module):
        def __init__(self, hf_model: Any):
            super().__init__()
            self.hf_model = hf_model

        def forward(self, x: Any) -> Any:
            return self.hf_model(pixel_values=x).logits

    image = Image.open(image_path).convert("RGB")
    size_cfg = processor.size
    if isinstance(size_cfg, dict):
        height = size_cfg.get("height", 224)
        width = size_cfg.get("width", height)
    else:
        height = width = int(size_cfg)

    vis_rgb = np.array(image.resize((width, height))).astype(np.float32) / 255.0
    input_tensor = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    wrapped_model = ViTLogitsWrapper(model).to(device).eval()

    with deps.torch.no_grad():
        logits = wrapped_model(input_tensor)
        pred_idx = int(deps.torch.argmax(logits, dim=1).item())

    target_layers = [
        resolve_vit_layer(model, -2, norm_pos="after"),
        resolve_vit_layer(model, -3, norm_pos="after"),
        resolve_vit_layer(model, -4, norm_pos="after"),
    ]
    target_weights = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    target_weights = target_weights / target_weights.sum()

    grayscale_sum = np.zeros((height, width), dtype=np.float32)
    targets = [cam_deps.ClassifierOutputTarget(pred_idx)]
    for layer, weight in zip(target_layers, target_weights):
        with cam_deps.GradCAMPlusPlus(
            model=wrapped_model,
            target_layers=[layer],
            reshape_transform=reshape_transform,
        ) as cam:
            grayscale = cam(input_tensor=input_tensor, targets=targets)[0]
            grayscale_sum += grayscale.astype(np.float32) * float(weight)

    grayscale_sum = np.clip(grayscale_sum, 0.0, 1.0)
    return cam_deps.show_cam_on_image(vis_rgb, grayscale_sum, use_rgb=True)


def save_uploaded_image(uploaded_file: Any, processed_dir: Path) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    output_path = processed_dir / f"upload_input{suffix}"
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def make_processed_dir() -> Path:
    timestamp = time.strftime("%y%m%d_%H%M%S")
    processed_dir = TEMP_ROOT / f"web_processed_{timestamp}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def read_binary(image_path: str | Path) -> np.ndarray:
    deps = get_runtime_deps()
    image = deps.cv2.imread(str(image_path), deps.cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    _, binary = deps.cv2.threshold(image, 0, 255, deps.cv2.THRESH_BINARY + deps.cv2.THRESH_OTSU)
    if binary[0, 0] == 255:
        binary = deps.cv2.bitwise_not(binary)
    return binary


def adaptive_canvas_size(
    binary_images: list[np.ndarray],
    padding_ratio: float = 0.15,
    min_size: int = 256,
    max_size: int = 768,
    align_to: int = 16,
) -> int:
    deps = get_runtime_deps()
    longest = 0
    for binary in binary_images:
        points = deps.cv2.findNonZero(binary)
        if points is None:
            continue
        _, _, width, height = deps.cv2.boundingRect(points)
        longest = max(longest, width, height)

    if longest == 0:
        size = min_size
    else:
        size = int(np.ceil(longest * (1 + 2 * padding_ratio)))
        size = max(min_size, min(max_size, size))
    return int(np.ceil(size / align_to) * align_to)


def get_centroid(binary_img: np.ndarray) -> tuple[int, int]:
    deps = get_runtime_deps()
    moments = deps.cv2.moments(binary_img)
    if moments["m00"] == 0:
        height, width = binary_img.shape
        return width // 2, height // 2
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def shift_image(binary_img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    deps = get_runtime_deps()
    height, width = binary_img.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return deps.cv2.warpAffine(binary_img, matrix, (width, height), flags=deps.cv2.INTER_NEAREST, borderValue=0)


def centroid_align(reference_binary: np.ndarray, target_binary: np.ndarray) -> tuple[np.ndarray, int, int]:
    ref_cx, ref_cy = get_centroid(reference_binary)
    tar_cx, tar_cy = get_centroid(target_binary)
    dx = ref_cx - tar_cx
    dy = ref_cy - tar_cy
    return shift_image(target_binary, dx, dy), dx, dy


def overlap_score(ref_skel: np.ndarray, user_skel: np.ndarray, tolerance: int = 10) -> float:
    deps = get_runtime_deps()
    ref_bool = ref_skel > 0
    user_bool = user_skel > 0
    if ref_bool.sum() == 0 or user_bool.sum() == 0:
        return 0.0

    dist_to_ref = deps.distance_transform_edt(~ref_bool)
    dist_to_user = deps.distance_transform_edt(~user_bool)
    user_distances = dist_to_ref[user_bool]
    ref_distances = dist_to_user[ref_bool]
    user_soft = np.exp(-(user_distances**2) / (2 * (tolerance**2)))
    ref_soft = np.exp(-(ref_distances**2) / (2 * (tolerance**2)))
    return float(0.5 * user_soft.mean() + 0.5 * ref_soft.mean())


def refine_alignment_by_search(
    ref_skel: np.ndarray,
    user_skel: np.ndarray,
    search_radius: int = 12,
    tolerance: int = 10,
) -> tuple[np.ndarray, int, int, float]:
    best_score = -1.0
    best_dx = 0
    best_dy = 0
    best_aligned = user_skel.copy()

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            shifted = shift_image(user_skel, dx, dy)
            score = overlap_score(ref_skel, shifted, tolerance=tolerance)
            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy
                best_aligned = shifted

    return best_aligned, best_dx, best_dy, best_score


def remove_small_noise_and_watermark(binary: np.ndarray, min_area_ratio: float = 0.002) -> np.ndarray:
    deps = get_runtime_deps()
    height, width = binary.shape
    num_labels, labels, stats, _ = deps.cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    total_area = height * width

    for index in range(1, num_labels):
        _, y, _, box_height, area = stats[index]
        area_ratio = area / total_area
        if area_ratio < min_area_ratio:
            continue
        if y > height * 0.75 and box_height < height * 0.15:
            continue
        cleaned[labels == index] = 255

    return cleaned


def preprocess_and_align_clean(
    image_path: str | Path,
    canvas_size: int = 256,
    padding_ratio: float = 0.15,
) -> np.ndarray:
    deps = get_runtime_deps()
    binary = read_binary(image_path)
    binary = remove_small_noise_and_watermark(binary)
    points = deps.cv2.findNonZero(binary)
    if points is None:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x, y, width, height = deps.cv2.boundingRect(points)
    subject = binary[y : y + height, x : x + width]
    target_side = int(canvas_size * (1 - 2 * padding_ratio))
    scale = min(target_side / max(width, 1), target_side / max(height, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = deps.cv2.resize(subject, (new_width, new_height), interpolation=deps.cv2.INTER_NEAREST)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    start_x = (canvas_size - new_width) // 2
    start_y = (canvas_size - new_height) // 2
    canvas[start_y : start_y + new_height, start_x : start_x + new_width] = resized
    return canvas


def generate_normalized_evaluation(
    std_path: str | Path,
    stu_path: str | Path,
    canvas_size: int | str = "auto",
    padding_ratio: float = 0.15,
    tolerance_ratio: float = 0.02,
    search_radius_ratio: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, int, float]:
    deps = get_runtime_deps()

    std_bin = read_binary(std_path)
    stu_bin = read_binary(stu_path)

    if canvas_size == "auto":
        canvas_size = adaptive_canvas_size([std_bin, stu_bin], padding_ratio=padding_ratio)

    std_aligned = preprocess_and_align_clean(std_path, canvas_size=canvas_size, padding_ratio=padding_ratio)
    stu_aligned = preprocess_and_align_clean(stu_path, canvas_size=canvas_size, padding_ratio=padding_ratio)

    std_skel = (deps.skeletonize(std_aligned > 0) * 255).astype(np.uint8)
    stu_skel = (deps.skeletonize(stu_aligned > 0) * 255).astype(np.uint8)

    stu_skel_centroid, coarse_dx, coarse_dy = centroid_align(std_skel, stu_skel)
    search_radius = max(4, int(round(canvas_size * search_radius_ratio)))
    soft_tolerance = max(3, int(round(canvas_size * tolerance_ratio)))

    _, fine_dx, fine_dy, align_score = refine_alignment_by_search(
        std_skel,
        stu_skel_centroid,
        search_radius=search_radius,
        tolerance=soft_tolerance,
    )

    total_dx = coarse_dx + fine_dx
    total_dy = coarse_dy + fine_dy
    stu_aligned_refined = shift_image(stu_aligned, total_dx, total_dy)

    tolerance_pixels = max(3, int(round(canvas_size * tolerance_ratio)))
    kernel = np.ones((tolerance_pixels, tolerance_pixels), np.uint8)
    std_safe_zone = deps.cv2.dilate(std_aligned, kernel, iterations=1)
    errors = deps.cv2.bitwise_and(stu_aligned_refined, deps.cv2.bitwise_not(std_safe_zone))

    total_student_pixels = np.count_nonzero(stu_aligned_refined)
    error_pixels = np.count_nonzero(errors)
    score = max(0.0, 100 - (error_pixels / total_student_pixels) * 200) if total_student_pixels > 0 else 0.0

    final_canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    final_canvas[std_aligned > 0] = [220, 220, 220]
    final_canvas[stu_aligned_refined > 0] = [30, 30, 30]
    final_canvas[errors > 0] = [255, 0, 0]

    return (
        std_aligned,
        stu_aligned_refined,
        final_canvas,
        float(score),
        int(canvas_size),
        int(total_dx),
        int(total_dy),
        float(align_score),
    )


def compare_skeleton_and_save(user_upload_path: Path, match_img_path: Path, process_dir: Path) -> dict[str, Any]:
    deps = get_runtime_deps()

    def _make_skeleton_vis(binary_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        skel = (deps.skeletonize(binary_img > 0) * 255).astype(np.uint8)
        vis = np.ones((*skel.shape, 3), dtype=np.uint8) * 255
        vis[skel > 0] = [0, 0, 0]
        return skel, vis

    std_aligned, stu_aligned, _, score, canvas_size, total_dx, total_dy, align_score = generate_normalized_evaluation(
        std_path=match_img_path,
        stu_path=user_upload_path,
    )

    std_vis_skel, std_vis = _make_skeleton_vis(std_aligned)
    stu_vis_skel, stu_vis = _make_skeleton_vis(stu_aligned)
    overlap = np.logical_and(std_vis_skel > 0, stu_vis_skel > 0)

    overlap_vis = np.ones((*std_vis_skel.shape, 3), dtype=np.uint8) * 255
    overlap_vis[std_vis_skel > 0] = [180, 180, 180]
    overlap_vis[stu_vis_skel > 0] = [0, 0, 0]
    overlap_vis[overlap] = [255, 0, 0]

    overlap_img_path = process_dir / "overlap_skeleton.png"
    deps.cv2.imwrite(str(overlap_img_path), overlap_vis)

    return {
        "score": score,
        "align_score": align_score,
        "canvas_size": canvas_size,
        "total_dx": total_dx,
        "total_dy": total_dy,
        "overlap_img_path": overlap_img_path,
        "master_vis": std_vis,
        "user_vis": stu_vis,
        "overlap_vis": overlap_vis,
    }


def compare_stroke_and_save(user_upload_path: Path, match_img_path: Path, process_dir: Path) -> dict[str, Any]:
    deps = get_runtime_deps()

    def _make_stroke_vis(binary_img: np.ndarray) -> np.ndarray:
        vis = np.ones((*binary_img.shape, 3), dtype=np.uint8) * 255
        vis[binary_img > 0] = [0, 0, 0]
        return vis

    std_aligned, stu_aligned, _, score, canvas_size, total_dx, total_dy, align_score = generate_normalized_evaluation(
        std_path=match_img_path,
        stu_path=user_upload_path,
    )

    std_bin = (std_aligned > 0).astype(np.uint8)
    stu_bin = (stu_aligned > 0).astype(np.uint8)
    overlap = np.logical_and(std_bin > 0, stu_bin > 0)

    std_vis = _make_stroke_vis(std_bin)
    stu_vis = _make_stroke_vis(stu_bin)
    overlap_vis = np.ones((*std_bin.shape, 3), dtype=np.uint8) * 255
    overlap_vis[std_bin > 0] = [180, 180, 180]
    overlap_vis[stu_bin > 0] = [0, 0, 0]
    overlap_vis[overlap] = [255, 0, 0]

    overlap_img_path = process_dir / "stroke_overlap.png"
    user_extract_path = process_dir / "user_extract.png"
    master_extract_path = process_dir / "master_extract.png"

    deps.cv2.imwrite(str(overlap_img_path), overlap_vis)
    deps.cv2.imwrite(str(user_extract_path), stu_vis)
    deps.cv2.imwrite(str(master_extract_path), std_vis)

    return {
        "score": score,
        "align_score": align_score,
        "canvas_size": canvas_size,
        "total_dx": total_dx,
        "total_dy": total_dy,
        "overlap_img_path": overlap_img_path,
        "user_extract_path": user_extract_path,
        "master_extract_path": master_extract_path,
        "master_vis": std_vis,
        "user_vis": stu_vis,
        "overlap_vis": overlap_vis,
    }


def extract_character(response_text: str) -> str:
    response_text = response_text.strip()
    match = re.search(r"[\u4e00-\u9fff]", response_text)
    if match:
        return match.group(0)
    return response_text


def recognize_character(image_path: Path) -> str:
    processor, model, _ = load_qwen_ocr_resources()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": "识别改书法字，并直接中文只回复答案即可，用简体字"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=64)
    response = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    return extract_character(response)


def pic_seeker(style_label: str, target_char: str) -> Path | None:
    pseudo_df = load_pseudo_df(style_label)
    matched_rows = pseudo_df[pseudo_df["pseudo_label"].astype(str) == str(target_char)]
    if matched_rows.empty:
        return None
    raw_path = str(matched_rows.iloc[0]["path"])
    return resolve_workspace_path(raw_path)


def to_relative_path(path_like: str | Path) -> str:
    path = normalize_path(path_like)
    if not path.is_absolute():
        root_candidate = ROOT / path
        if root_candidate.exists():
            path = root_candidate
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def get_query_embedding_from_db(match_img_path: Path, merged_paths: list[str], merged_emb: Any) -> tuple[Any, str]:
    query_abs = match_img_path.resolve()
    query_rel = to_relative_path(query_abs)

    for idx, raw_path in enumerate(merged_paths):
        rel_path = to_relative_path(raw_path)
        abs_path = resolve_workspace_path(raw_path).resolve()
        if rel_path == query_rel or abs_path == query_abs:
            return merged_emb[idx : idx + 1], rel_path

    query_name = query_abs.name
    name_matches = [idx for idx, raw_path in enumerate(merged_paths) if normalize_path(raw_path).name == query_name]
    if len(name_matches) == 1:
        idx = name_matches[0]
        return merged_emb[idx : idx + 1], to_relative_path(merged_paths[idx])

    raise ValueError("Matched image path is not present in the embedding database.")


def generate_qwen_response(content: list[dict[str, Any]], max_new_tokens: int = 400) -> str:
    processor, model, _ = load_qwen_ocr_resources()
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()


def generate_llm_feedback(
    style_master_name: str,
    recognized_char: str,
    skeleton_result: dict[str, Any],
    stroke_result: dict[str, Any],
    similar_results: list[dict[str, Any]],
) -> dict[str, str]:
    master_extract = Image.open(stroke_result["master_extract_path"]).convert("RGB")
    user_extract = Image.open(stroke_result["user_extract_path"]).convert("RGB")
    overlap_skeleton = Image.open(skeleton_result["overlap_img_path"]).convert("RGB")
    stroke_overlap = Image.open(stroke_result["overlap_img_path"]).convert("RGB")

    prompt_1 = f"""
The style of the master is {style_master_name}
Can you analyze the gap between the student and the master, only focus on the character's structure itself, and give a score from 0 to 10?
""".strip()

    structure_feedback = generate_qwen_response(
        [
            {"type": "text", "text": "the image_1 is the master's work"},
            {"type": "image", "name": "image_1", "image": master_extract},
            {"type": "text", "text": "the image_2 is the student's work"},
            {"type": "image", "name": "image_2", "image": user_extract},
            {"type": "text", "text": "the image_3 is the skeleton overlap of the master and student"},
            {"type": "image", "name": "image_3", "image": overlap_skeleton},
            {"type": "text", "text": "the image_4 is the stroke overlap of the master and student"},
            {"type": "image", "name": "image_4", "image": stroke_overlap},
            {"type": "text", "text": prompt_1},
        ],
        max_new_tokens=800,
    )

    skill_feedback = ""
    if similar_results:
        prompt_2 = f"""
The style of the master is {style_master_name};
The character is {recognized_char};
Here are some similar works retrieved with similar structure, please analyze the written skill of these characters,
and give advices of how to write better.
""".strip()

        content = [
            {"type": "text", "text": "the images are the master's similar works"},
            {"type": "image", "name": "image_1", "image": master_extract},
        ]
        for idx, item in enumerate(similar_results[:3], start=2):
            image_path = item.get("copied_path") or item["absolute_path"]
            content.append(
                {"type": "image", "name": f"image_{idx}", "image": Image.open(image_path).convert("RGB")}
            )
        content.append({"type": "text", "text": prompt_2})
        skill_feedback = generate_qwen_response(content, max_new_tokens=700)

    return {
        "structure_feedback": structure_feedback,
        "skill_feedback": skill_feedback,
    }


def noop_progress(_: float, __: str) -> None:
    return None


def search_mixed_topk(
    match_img_path: Path,
    train_embedding_result: dict[str, Any],
    test_embedding_result: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    deps = get_runtime_deps()
    train_paths = train_embedding_result["image_paths"]
    test_paths = test_embedding_result["image_paths"]
    train_emb = train_embedding_result["embeddings"]
    test_emb = test_embedding_result["embeddings"]

    merged_paths = train_paths + test_paths
    if train_emb.shape[0] > 0 and test_emb.shape[0] > 0:
        merged_emb = deps.torch.cat([train_emb, test_emb], dim=0)
    elif train_emb.shape[0] > 0:
        merged_emb = train_emb
    else:
        merged_emb = test_emb

    if len(merged_paths) == 0 or merged_emb.shape[0] == 0:
        return []

    query_emb, query_rel = get_query_embedding_from_db(match_img_path, merged_paths, merged_emb)

    query_norm = deps.torch.nn.functional.normalize(query_emb, p=2, dim=1)
    db_norm = deps.torch.nn.functional.normalize(merged_emb, p=2, dim=1)
    sims = (query_norm @ db_norm.T).squeeze(0)
    values, indices = deps.torch.topk(sims, k=min(top_k + 1, merged_emb.shape[0]))

    mixed_results = []
    train_count = len(train_paths)
    for score, index in zip(values.tolist(), indices.tolist()):
        rel_path = to_relative_path(merged_paths[index])
        if rel_path == query_rel:
            continue
        mixed_results.append(
            {
                "relative_path": rel_path,
                "absolute_path": resolve_workspace_path(rel_path),
                "similarity": float(score),
                "split": "train" if index < train_count else "test",
            }
        )
        if len(mixed_results) >= top_k:
            break

    return mixed_results


def run_pipeline(
    uploaded_file: Any,
    style_label: str,
    top_k: int,
    manual_character: str,
    need_attention: bool,
    need_llm_feedback: bool,
    progress_callback: Any = None,
) -> dict[str, Any]:
    progress = progress_callback or noop_progress
    progress(0.05, "Preparing workspace")
    processed_dir = make_processed_dir()
    upload_path = save_uploaded_image(uploaded_file, processed_dir)
    _, label_to_master, _ = load_summary()
    assets_df = scan_style_assets().set_index("label")
    asset_row = assets_df.loc[style_label]

    results: dict[str, Any] = {
        "processed_dir": processed_dir,
        "upload_path": upload_path,
        "selected_style_label": style_label,
        "selected_calligrapher": label_to_master.get(style_label, style_label),
        "warnings": [],
    }

    progress(0.18, "Running style classification")
    results["style_prediction"] = predict_style_distribution(upload_path)

    if need_attention:
        try:
            progress(0.32, "Generating attention heatmap")
            results["attention_overlay"] = generate_attention_overlay(upload_path)
        except Exception as exc:
            results["warnings"].append(f"Attention heatmap skipped: {exc}")

    if manual_character.strip():
        recognized_char = manual_character.strip()
    else:
        try:
            progress(0.45, "Recognizing character")
            recognized_char = recognize_character(upload_path)
        except Exception as exc:
            results["warnings"].append(
                "Automatic character recognition is unavailable in the current environment. "
                f"Please fill the character manually. Details: {exc}"
            )
            recognized_char = ""

    results["recognized_char"] = recognized_char
    if not recognized_char:
        progress(1.0, "Finished with manual character required")
        return results

    if not bool(asset_row["pseudo_csv"]):
        results["warnings"].append(
            f"No pseudo label CSV is available for style '{style_label}', so matching and overlap analysis were skipped."
        )
        progress(1.0, "Finished without reference matching")
        return results

    progress(0.58, "Matching reference image")
    match_img_path = pic_seeker(style_label, recognized_char)
    if match_img_path is None or not match_img_path.exists():
        results["warnings"].append(
            f"No reference image found for style '{style_label}' and character '{recognized_char}'."
        )
        progress(1.0, "Finished without reference image")
        return results

    results["matched_reference_path"] = match_img_path
    progress(0.68, "Computing skeleton overlap")
    results["skeleton_result"] = compare_skeleton_and_save(upload_path, match_img_path, processed_dir)
    progress(0.76, "Computing stroke overlap")
    results["stroke_result"] = compare_stroke_and_save(upload_path, match_img_path, processed_dir)

    if bool(asset_row["embedding_ready"]):
        progress(0.84, f"Retrieving top-{top_k} similar works")
        train_embedding_result, test_embedding_result = load_embedding_results(style_label)
        results["similar_results"] = search_mixed_topk(
            match_img_path,
            train_embedding_result,
            test_embedding_result,
            top_k=top_k,
        )
    else:
        results["similar_results"] = []
        results["warnings"].append(
            f"Embedding assets for style '{style_label}' are incomplete, so top-k similar works were skipped."
        )

    if results["similar_results"]:
        similar_dir = processed_dir / "similar"
        similar_dir.mkdir(parents=True, exist_ok=True)
        copied_paths = []
        for index, item in enumerate(results["similar_results"], start=1):
            src = item["absolute_path"]
            if not src.exists():
                continue
            dst = similar_dir / f"similar_top_{index}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            item["copied_path"] = dst
            copied_paths.append(dst)
        results["copied_similar_paths"] = copied_paths

    if need_llm_feedback and "stroke_result" in results:
        try:
            progress(0.93, "Generating LLM feedback")
            results["llm_feedback"] = generate_llm_feedback(
                style_master_name=results["selected_calligrapher"],
                recognized_char=recognized_char,
                skeleton_result=results["skeleton_result"],
                stroke_result=results["stroke_result"],
                similar_results=results.get("similar_results", []),
            )
        except Exception as exc:
            results["warnings"].append(f"LLM feedback skipped: {exc}")

    progress(1.0, "Finished")
    return results


def render_header() -> None:
    st.set_page_config(page_title="Calligraphy Combine Demo", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(193, 154, 107, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(74, 58, 41, 0.16), transparent 28%),
                linear-gradient(180deg, #f7f1e5 0%, #efe3cf 52%, #e9dcc8 100%);
        }
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }
        html, body, [class*="css"] {
            font-family: "Songti SC", "STSong", "Noto Serif SC", serif;
        }
        .hero-card {
            border: 1px solid rgba(72, 52, 33, 0.12);
            background: rgba(255, 250, 243, 0.82);
            border-radius: 24px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 20px 45px rgba(73, 54, 36, 0.08);
            backdrop-filter: blur(8px);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.76rem;
            color: #7d5f3b;
            margin-bottom: 0.5rem;
        }
        .hero-title {
            font-size: 2.2rem;
            line-height: 1.1;
            color: #2f2216;
            margin: 0;
        }
        .hero-copy {
            color: #5c4733;
            font-size: 1rem;
            margin-top: 0.8rem;
            margin-bottom: 0;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 252, 247, 0.78);
            border: 1px solid rgba(72, 52, 33, 0.1);
            border-radius: 18px;
            padding: 0.7rem 0.9rem;
        }
        </style>
        <div class="hero-card">
            <div class="hero-kicker">combine notebook to local web app</div>
            <h1 class="hero-title">Calligraphy Review Studio</h1>
            <p class="hero-copy">
                Upload one calligraphy image, choose the target calligrapher, set top-k,
                and the page will follow the same combine pipeline: style prediction,
                character match, overlap comparison, and similar work retrieval.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    asset_df = scan_style_assets()
    st.sidebar.markdown("## Workspace support")
    st.sidebar.caption("The table below is detected from the current folder, not hard-coded.")
    st.sidebar.dataframe(
        asset_df.rename(
            columns={
                "label": "label",
                "calligrapher": "calligrapher",
                "pseudo_csv": "pseudo_csv",
                "embedding_ready": "embedding_ready",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.sidebar.markdown("## Run note")
    st.sidebar.caption(
        "If the OCR model is not cached locally, fill the character manually. "
        "The app will still run the rest of the combine flow."
    )


def render_results(results: dict[str, Any]) -> None:
    style_prediction = results["style_prediction"]
    top_three_df = pd.DataFrame(style_prediction["top_three"])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Selected style", results["selected_style_label"])
    metric_cols[1].metric("Selected calligrapher", results["selected_calligrapher"])
    metric_cols[2].metric("Predicted style", style_prediction["predicted_label"])
    metric_cols[3].metric("Predicted calligrapher", style_prediction["predicted_calligrapher"])

    if results["warnings"]:
        for warning in results["warnings"]:
            st.warning(warning)

    col_left, col_right = st.columns([1.05, 1.0], gap="large")
    with col_left:
        st.markdown("### Input")
        st.image(str(results["upload_path"]), use_container_width=True)
        st.markdown("### Top-3 style prediction")
        st.dataframe(top_three_df, use_container_width=True, hide_index=True)
        st.pyplot(style_prediction["figure"], clear_figure=True)

    with col_right:
        st.markdown("### Character and attention")
        if results.get("recognized_char"):
            st.metric("Recognized character", results["recognized_char"])
        else:
            st.info("Character is still empty. Fill it manually if OCR is unavailable.")

        if "attention_overlay" in results:
            st.image(results["attention_overlay"], caption="ViT attention overlay", use_container_width=True)
        else:
            st.info("Attention heatmap is optional and may be skipped if CAM dependencies are missing.")

    matched_reference_path = results.get("matched_reference_path")
    if matched_reference_path:
        st.markdown("### Reference match")
        ref_col_1, ref_col_2 = st.columns(2, gap="large")
        with ref_col_1:
            st.image(str(matched_reference_path), caption="Matched reference image", use_container_width=True)
        with ref_col_2:
            stroke_result = results["stroke_result"]
            st.metric("Stroke overlap score", f"{stroke_result['score']:.2f}")
            st.metric("Alignment score", f"{stroke_result['align_score']:.4f}")
            st.image(stroke_result["overlap_vis"], caption="Stroke overlap", use_container_width=True)

        skeleton_result = results["skeleton_result"]
        compare_cols = st.columns(3, gap="medium")
        compare_cols[0].image(stroke_result["master_vis"], caption="Master extract", use_container_width=True)
        compare_cols[1].image(stroke_result["user_vis"], caption="User extract", use_container_width=True)
        compare_cols[2].image(skeleton_result["overlap_vis"], caption="Skeleton overlap", use_container_width=True)

    similar_results = results.get("similar_results") or []
    if similar_results:
        st.markdown("### Similar works")
        table_rows = [
            {
                "path": item["relative_path"],
                "split": item["split"],
                "similarity": round(item["similarity"], 4),
            }
            for item in similar_results
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        image_cols = st.columns(min(4, len(similar_results)))
        for idx, item in enumerate(similar_results):
            column = image_cols[idx % len(image_cols)]
            image_path = item.get("copied_path") or item["absolute_path"]
            column.image(
                str(image_path),
                caption=f"{item['split']} | {item['similarity']:.4f}",
                use_container_width=True,
            )

    llm_feedback = results.get("llm_feedback")
    if llm_feedback:
        st.markdown("### LLM feedback")
        feedback_col_1, feedback_col_2 = st.columns(2, gap="large")
        with feedback_col_1:
            st.markdown("#### Structure analysis")
            st.write(llm_feedback.get("structure_feedback", ""))
        with feedback_col_2:
            st.markdown("#### Similar work advice")
            if llm_feedback.get("skill_feedback"):
                st.write(llm_feedback["skill_feedback"])
            else:
                st.info("No similar-work advice was generated for this run.")

    st.caption(f"Processed output folder: {results['processed_dir']}")


def main() -> None:
    render_header()
    render_sidebar()

    options = style_option_map()
    labels = list(options.keys())
    default_index = 0
    for idx, option in enumerate(labels):
        if option.startswith("yzq |"):
            default_index = idx
            break

    with st.form("combine_form"):
        st.markdown("### Controls")
        uploaded_file = st.file_uploader("Upload a calligraphy image", type=["png", "jpg", "jpeg", "bmp", "webp"])
        selected_option = st.selectbox("Calligrapher / style", labels, index=default_index)
        manual_character = st.text_input("Character (optional, leave empty to try OCR)", "")
        top_k = st.slider("Top-k similar works", min_value=1, max_value=8, value=3, step=1)
        need_attention = st.checkbox("Generate attention heatmap if CAM support exists", value=True)
        need_llm_feedback = st.checkbox("Generate LLM feedback with Qwen", value=True)
        submitted = st.form_submit_button("Run combine pipeline")

    if not submitted:
        st.info("Upload one image and click the button to run the local pipeline.")
        return

    if uploaded_file is None:
        st.error("Please upload an image first.")
        return

    style_label = options[selected_option]
    progress_placeholder = st.empty()
    progress_bar = st.progress(0, text="Waiting to start")

    def update_progress(value: float, message: str) -> None:
        percent = max(0, min(100, int(round(value * 100))))
        progress_bar.progress(percent, text=f"{percent}% | {message}")
        progress_placeholder.caption(f"Current step: {message}")

    try:
        results = run_pipeline(
            uploaded_file=uploaded_file,
            style_label=style_label,
            top_k=top_k,
            manual_character=manual_character,
            need_attention=need_attention,
            need_llm_feedback=need_llm_feedback,
            progress_callback=update_progress,
        )
    except Exception as exc:
        progress_bar.empty()
        progress_placeholder.empty()
        st.exception(exc)
        return

    progress_bar.progress(100, text="100% | Finished")
    progress_placeholder.caption("Current step: Finished")

    render_results(results)


if __name__ == "__main__":
    main()
