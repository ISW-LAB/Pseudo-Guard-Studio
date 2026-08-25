#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ultra_det_loader.py

Ultralytics Detection Dataset Loader Module (Headless-safe)

- Ultralytics를 import하지 않고,
  load_dir 아래에 다운로드된 YOLO detection 데이터셋을
  자동 탐지/로드하기 위한 모듈.

- 서버 환경(cv2/libGL 이슈) 안전:
  PIL 기반 로더만 사용.

지원하는 일반 구조:
  <dataset_root>/
    images/...
    labels/...
    data.yaml 또는 <name>.yaml

Ultralytics YAML 형식 지원:
  path: <root>
  train: images/train  또는 train.txt(이미지 경로 목록)
  val: images/val 또는 images/valid
  test: ...

출력 포맷:
  - torchvision detection 스타일 target:
      target = {
        "boxes": FloatTensor[N, 4] (xyxy, absolute),
        "labels": LongTensor[N],
        "image_id": LongTensor[1],
        "orig_size": (H, W),
        "size": (H', W')  # resize 후
      }

✅ 추가 기능 (요구 반영):
- discover/inspect/load_all_splits 에 dataset_names(폴더명 allowlist) 옵션 추가
  -> 호출부에서 ["BCCD", "Custom_Blood"] 처럼 지정하면
     해당 이름에 매칭되는 dataset root만 로드.

의존성:
  pip install pyyaml torch torchvision pillow
"""

from __future__ import annotations

import os
import glob
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

try:
    from torchvision import transforms as T
except Exception:
    T = None


# -----------------------------------------------------------------------------
# Dataclass: Dataset Spec
# -----------------------------------------------------------------------------
@dataclass
class DetDatasetSpec:
    name: str
    root: Path
    yaml_path: Optional[Path]
    data: Dict

    def has_split(self, split: str) -> bool:
        return split in self.data

    def __repr__(self) -> str:
        return f"DetDatasetSpec(name={self.name}, root={str(self.root)}, yaml={self.yaml_path})"


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _safe_read_yaml(p: Path) -> Optional[Dict]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _normalize_name(x: str) -> str:
    # ✅ underscore/space 통일 + lower
    return (
        str(x)
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
        .strip()
    )


def _guess_dataset_name_from_root(root: Path) -> str:
    return root.name


def _is_ultra_det_yaml(d: Dict) -> bool:
    if not isinstance(d, dict):
        return False
    keys = set(d.keys())
    has_split = any(k in keys for k in ["train", "val", "test"])
    return has_split


def _resolve_root_from_yaml(yaml_path: Path, data: Dict) -> Path:
    """
    Ultralytics YAML의 `path` 해석 + 중첩/오탐 보정.

    흔한 문제:
      - yaml이 <ds>/<ds>.yaml 에 있고
      - path: <ds> 로 되어 있으면
        root가 <ds>/<ds> 로 한 번 더 중첩되어
        images가 0개로 잡히는 케이스가 있다.
    """
    base_dir = yaml_path.parent
    p = data.get("path", None)

    # 1) 기본 해석
    if p is None:
        cand = base_dir
    else:
        cand = Path(str(p))
        if not cand.is_absolute():
            cand = (base_dir / cand).resolve()

    # 2) 이미 정상 구조면 그대로
    if (cand / "images").exists() or (cand / "labels").exists():
        return cand

    # 3) "폴더/폴더" 중첩 의심 보정
    #    예) /datasets/african-wildlife/african-wildlife
    if cand.name == yaml_path.stem and (cand.parent / "images").exists():
        return cand.parent

    # 4) base_dir가 더 그럴싸한 경우
    if (base_dir / "images").exists() or (base_dir / "labels").exists():
        return base_dir

    return cand


# -----------------------------------------------------------------------------
# ✅ Allowlist name matching
# -----------------------------------------------------------------------------
def _prepare_allow_norms(dataset_names: Optional[List[str]]) -> Optional[List[str]]:
    if not dataset_names:
        return None
    allow = []
    for n in dataset_names:
        if n is None:
            continue
        nn = _normalize_name(str(n))
        if nn:
            allow.append(nn)
    return list(dict.fromkeys(allow)) if allow else None


def _match_allowlist(name: str, allow_norms: Optional[List[str]]) -> bool:
    """
    폴더명 allowlist 매칭:
    - 정규화 후 정확 일치
    - 또는 부분 포함 매칭(요구를 고려한 유연성)
      예: allow="custom-blood" 가 dir="custom-blood-cell"에 매칭
    """
    if not allow_norms:
        return True

    n = _normalize_name(name)
    if not n:
        return False

    if n in allow_norms:
        return True

    # 부분 매칭 허용
    for a in allow_norms:
        if not a:
            continue
        if a in n or n in a:
            return True

    return False


# -----------------------------------------------------------------------------
# Split entry expand
# -----------------------------------------------------------------------------
def _expand_split_entry(root: Path, entry: Union[str, List[str]]) -> List[Path]:
    """
    train/val/test 항목은:
      1) images/xxx 폴더 경로
      2) txt 파일(이미지 경로 리스트)
      3) 리스트 형태
    를 가질 수 있다.
    """
    paths: List[Path] = []

    if isinstance(entry, list):
        for e in entry:
            paths.extend(_expand_split_entry(root, e))
        return paths

    entry = str(entry)
    p = Path(entry)

    if not p.is_absolute():
        p = (root / p).resolve()

    # txt 리스트 형태
    if p.suffix.lower() == ".txt" and p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_p = Path(line)
                if not img_p.is_absolute():
                    cand1 = (p.parent / img_p).resolve()
                    cand2 = (root / img_p).resolve()
                    img_p = cand1 if cand1.exists() else cand2
                paths.append(img_p)
        return paths

    # 폴더/글롭
    if p.is_dir():
        _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        paths.extend([
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in _IMG_EXTS
        ])
        return sorted(list(set(paths)))

    # 단일 파일
    if p.exists():
        paths.append(p)

    return paths


def _image_to_label_path(img_path: Path, root: Path) -> Path:
    """
    기본 YOLO 폴더 대응 규칙:
      .../images/.../xxx.jpg -> .../labels/.../xxx.txt
    """
    # Normalize to forward slashes for cross-platform compatibility
    s = str(img_path).replace("\\", "/")
    if "/images/" in s:
        s2 = s.replace("/images/", "/labels/")
        lp = Path(os.path.splitext(s2)[0] + ".txt")
        return lp

    return root / "labels" / (img_path.stem + ".txt")


def _read_yolo_label_file(label_path: Path) -> Tuple[List[int], List[List[float]]]:
    """
    YOLO bbox txt:
      class cx cy w h  (normalized)
    """
    classes: List[int] = []
    boxes_norm: List[List[float]] = []

    if not label_path.exists():
        return classes, boxes_norm

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                classes.append(cls)
                boxes_norm.append([cx, cy, w, h])
    except Exception:
        return [], []

    return classes, boxes_norm


def _yolo_norm_to_xyxy_abs(
    boxes_norm: List[List[float]],
    img_w: int,
    img_h: int
) -> List[List[float]]:
    out: List[List[float]] = []
    for cx, cy, w, h in boxes_norm:
        bw = w * img_w
        bh = h * img_h
        x1 = (cx * img_w) - bw / 2.0
        y1 = (cy * img_h) - bh / 2.0
        x2 = x1 + bw
        y2 = y1 + bh

        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))

        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out


def _infer_class_names(data: Dict) -> List[str]:
    """
    YAML의 names/nc를 기반으로 클래스 이름 리스트 구성.
    """
    names = data.get("names", None)
    nc = data.get("nc", None)

    if isinstance(names, dict):
        max_k = max(names.keys()) if names else -1
        out = []
        for i in range(max_k + 1):
            out.append(str(names.get(i, i)))
        return out

    if isinstance(names, list):
        return [str(x) for x in names]

    if isinstance(nc, int) and nc > 0:
        return [str(i) for i in range(nc)]

    return []


# -----------------------------------------------------------------------------
# Split 후보 경로 헬퍼 (val/valid 대응)
# -----------------------------------------------------------------------------
def _split_aliases(split: str) -> List[str]:
    s = split.lower()
    if s in ["val", "valid", "validation"]:
        return ["val", "valid"]
    return [split]


def _candidate_image_dirs(root: Path, split: str) -> List[Path]:
    cands = []
    for sp in _split_aliases(split):
        cands.append(root / "images" / sp)
    # 혹시 images만 있고 하위 split이 없는 구조
    cands.append(root / "images")
    return cands


def _candidate_label_dirs(root: Path, split: str) -> List[Path]:
    cands = []
    for sp in _split_aliases(split):
        cands.append(root / "labels" / sp)
    # 혹시 labels만 있고 하위 split이 없는 구조
    cands.append(root / "labels")
    return cands


def _list_images_by_heuristic(root: Path, split: str) -> List[Path]:
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for d in _candidate_image_dirs(root, split):
        if d.exists() and d.is_dir():
            imgs = [
                f for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in _IMG_EXTS
            ]
            imgs = sorted(list(set(imgs)))
            if imgs:
                return imgs
    return []


# -----------------------------------------------------------------------------
# Label 기반 통계 수집 (YAML이 없거나 불완전할 때)
# -----------------------------------------------------------------------------
def _iter_label_files_for_split(root: Path, split: str) -> List[Path]:
    # 1) labels/{split|valid} 가 있으면 우선 사용
    for d in _candidate_label_dirs(root, split):
        if d.exists() and d.is_dir():
            files = sorted([p for p in d.rglob("*.txt") if p.is_file()])
            # labels 루트에 다른 메타 txt가 섞일 수 있어 split 폴더가 있으면 그걸 우선
            if (root / "labels" / split).exists() or (
                split in ["val", "valid"] and (root / "labels" / "valid").exists()
            ):
                return files
            # split 폴더가 없는 단일 구조면 files 반환
            if files:
                return files

    # 2) labels split 폴더가 없으면 images를 기반으로 label 경로 매핑
    imgs = _list_images_by_heuristic(root, split)
    out = []
    for ip in imgs:
        lp = _image_to_label_path(ip, root)
        if lp.exists():
            out.append(lp)
    return sorted(list(set(out)))


def _collect_label_stats(root: Path, split: str) -> Dict:
    """
    labels txt를 전수 순회하며
    - class id set
    - class별 object count
    - total objects
    - label files count
    - image count(가능하면 images 기반, 아니면 label files 기반)
    """
    label_files = _iter_label_files_for_split(root, split)

    class_ids = set()
    class_counter = Counter()
    total_objects = 0

    for lf in label_files:
        try:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    class_ids.add(cls)
                    class_counter[cls] += 1
                    total_objects += 1
        except Exception:
            continue

    imgs = _list_images_by_heuristic(root, split)
    num_images = len(imgs) if imgs else len(label_files)

    return {
        "split": split,
        "num_images": num_images,
        "num_label_files": len(label_files),
        "total_objects": total_objects,
        "class_ids": sorted(list(class_ids)),
        "class_counter": class_counter,
    }


def _infer_names_from_class_ids(class_ids: List[int]) -> List[str]:
    if not class_ids:
        return []
    max_id = max(class_ids)
    return [f"class_{i}" for i in range(max_id + 1)]


# -----------------------------------------------------------------------------
# Transform Builder
# -----------------------------------------------------------------------------
def build_default_transform(img_size: int = 640):
    """
    일반적인 detection 입력 전처리:
      - Resize (정사각)
      - ToTensor
      - Normalize(ImageNet)
    """
    if T is None:
        raise ImportError("torchvision이 필요합니다. pip install torchvision")

    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class YoloDetDataset(Dataset):
    def __init__(
        self,
        spec: DetDatasetSpec,
        split: str = "train",
        img_size: int = 640,
        transform=None,
        return_normalized: bool = False,
        strict_exists: bool = False,
    ):
        self.spec = spec
        self.split = split
        self.img_size = img_size
        self.transform = transform if transform is not None else build_default_transform(img_size)
        self.return_normalized = return_normalized

        entry = spec.data.get(split, None)
        self.images: List[Path] = []

        # 1) YAML entry 기반
        if entry is not None:
            self.images = _expand_split_entry(spec.root, entry)

        # 2) heuristic fallback (val/valid 포함)
        if not self.images:
            self.images = _list_images_by_heuristic(spec.root, split)

        self.images = sorted(self.images)

        if strict_exists and len(self.images) == 0:
            raise FileNotFoundError(f"[{spec.name}] split={split} 이미지가 발견되지 않았습니다.")

        self.class_names = _infer_class_names(spec.data)

        # YAML에 names/nc가 없고 yaml 자체도 없으면 라벨 기반 추정
        if not self.class_names:
            st = _collect_label_stats(spec.root, split="train")
            self.class_names = _infer_names_from_class_ids(st["class_ids"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        label_path = _image_to_label_path(img_path, self.spec.root)
        classes, boxes_norm = _read_yolo_label_file(label_path)
        boxes_xyxy = _yolo_norm_to_xyxy_abs(boxes_norm, orig_w, orig_h)

        labels_t = torch.as_tensor(classes, dtype=torch.long)
        boxes_t = torch.as_tensor(boxes_xyxy, dtype=torch.float32)

        new_w = self.img_size
        new_h = self.img_size
        sx = new_w / float(orig_w)
        sy = new_h / float(orig_h)

        if boxes_t.numel() > 0:
            boxes_t[:, 0] *= sx
            boxes_t[:, 2] *= sx
            boxes_t[:, 1] *= sy
            boxes_t[:, 3] *= sy

        img_t = self.transform(img)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.long),
            "orig_size": (orig_h, orig_w),
            "size": (new_h, new_w),
            "img_path": str(img_path),
            "label_path": str(label_path),
        }

        if self.return_normalized:
            target["boxes_norm_cxcywh"] = torch.as_tensor(boxes_norm, dtype=torch.float32)

        return img_t, target


# -----------------------------------------------------------------------------
# Discovery (중복 제거 포함)
# -----------------------------------------------------------------------------
def _count_images_for_spec(spec: DetDatasetSpec, split: str) -> int:
    entry = spec.data.get(split, None)
    imgs: List[Path] = []
    if entry is not None:
        imgs = _expand_split_entry(spec.root, entry)

    if not imgs:
        imgs = _list_images_by_heuristic(spec.root, split)

    return len(imgs)


def _spec_score(spec: DetDatasetSpec) -> int:
    return (
        _count_images_for_spec(spec, "train")
        + _count_images_for_spec(spec, "val")
        + _count_images_for_spec(spec, "test")
    )


def discover_det_datasets(load_dir: str, dataset_names: Optional[List[str]] = None) -> List[DetDatasetSpec]:
    """
    load_dir 아래에서 detection dataset 후보를 탐지.

    ✅ 개선:
      - YAML 기반 spec + heuristic spec이 같은 name으로 중복될 경우
        이미지 수가 더 많은 spec만 남김.
      - val/valid 구조 대응.
      - ✅ dataset_names allowlist 지원:
          호출부에서 폴더명을 리스트로 주면
          해당 목록에 매칭되는 dataset root만 반환.
    """
    load_root = Path(load_dir).resolve()
    if not load_root.exists():
        return []

    allow_norms = _prepare_allow_norms(dataset_names)

    yamls = []
    for pattern in ["*.yaml", "**/*.yaml"]:
        yamls.extend(load_root.glob(pattern))

    specs: List[DetDatasetSpec] = []
    seen = set()

    # 1) YAML 기반 spec 수집
    for y in yamls:
        data = _safe_read_yaml(y)
        if not data or not _is_ultra_det_yaml(data):
            continue

        root = _resolve_root_from_yaml(y, data)
        name = _guess_dataset_name_from_root(root)

        # ✅ allowlist 필터
        if not _match_allowlist(name, allow_norms) and not _match_allowlist(root.name, allow_norms):
            continue

        key = (str(root), str(y))
        if key in seen:
            continue
        seen.add(key)

        specs.append(DetDatasetSpec(
            name=name,
            root=root,
            yaml_path=y,
            data=data,
        ))

    # 2) YAML이 없는 폴더 heuristic
    #    - allowlist가 있으면 해당 폴더만
    #    - 2-depth 탐색: images/가 없으면 하위 디렉토리도 검색 (e.g., WBC datasets/1-1. .../images/)
    def _try_add_heuristic_spec(d: Path):
        """Helper: d 폴더에 images/가 있으면 spec으로 추가"""
        if not (d / "images").exists():
            return
        key = (str(d), "no_yaml")
        if key in seen:
            return
        seen.add(key)

        data = {
            "path": str(d),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
        }
        specs.append(DetDatasetSpec(
            name=d.name,
            root=d,
            yaml_path=None,
            data=data,
        ))

    for d in load_root.iterdir():
        if not d.is_dir():
            continue

        # ✅ allowlist 필터
        if not _match_allowlist(d.name, allow_norms):
            # Even if parent dir doesn't match, check subdirs for allowlist match
            if (d / "images").exists():
                continue  # Parent has images/ but doesn't match allowlist, skip
            # 2-depth: check subdirectories (e.g., "WBC datasets" → "1-1. Detection (...)")
            for sub_d in d.iterdir():
                if not sub_d.is_dir():
                    continue
                if not _match_allowlist(sub_d.name, allow_norms):
                    continue
                _try_add_heuristic_spec(sub_d)
            continue

        if (d / "images").exists():
            _try_add_heuristic_spec(d)
        else:
            # 2-depth: parent matches allowlist but no images/ → check subdirs
            for sub_d in d.iterdir():
                if not sub_d.is_dir():
                    continue
                if allow_norms and not _match_allowlist(sub_d.name, allow_norms):
                    continue
                _try_add_heuristic_spec(sub_d)

    # 3) ✅ name 기준 중복 제거
    grouped: Dict[str, List[DetDatasetSpec]] = {}
    for s in specs:
        k = _normalize_name(s.name)
        grouped.setdefault(k, []).append(s)

    deduped: List[DetDatasetSpec] = []
    for _, arr in grouped.items():
        arr_sorted = sorted(
            arr,
            key=lambda s: (_spec_score(s), 1 if s.yaml_path is not None else 0),
            reverse=True
        )
        deduped.append(arr_sorted[0])

    # 4) ✅ 이미지 많은 순으로 우선 정렬
    deduped = sorted(deduped, key=lambda s: (-_spec_score(s), _normalize_name(s.name)))
    return deduped


# -----------------------------------------------------------------------------
# Factory helpers
# -----------------------------------------------------------------------------
def build_dataset(
    spec: DetDatasetSpec,
    split: str = "train",
    img_size: int = 640,
    transform=None,
    strict_exists: bool = False,
) -> YoloDetDataset:
    return YoloDetDataset(
        spec=spec,
        split=split,
        img_size=img_size,
        transform=transform,
        strict_exists=strict_exists,
    )


__all__ = [
    "DetDatasetSpec",
    "YoloDetDataset",
    "discover_det_datasets",
    "build_dataset",
]
