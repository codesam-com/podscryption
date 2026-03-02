from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import math


def _l2norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) + 1e-12


def cosine(a: List[float], b: List[float]) -> float:
    na = _l2norm(a)
    nb = _l2norm(b)
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _weighted_mean(old: List[float], new: List[float], w_old: float, w_new: float) -> List[float]:
    total = w_old + w_new + 1e-9
    return [(o * w_old + n * w_new) / total for o, n in zip(old, new)]


@dataclass
class Profile:
    path: Path
    raw: Dict[str, Any]

    @property
    def alias(self) -> str:
        return self.raw["alias"]

    @property
    def full_name(self) -> str:
        return self.raw.get("full_name", self.alias)

    @property
    def centroid(self) -> Optional[List[float]]:
        v = self.raw.get("voiceprints", {}).get("centroid_embedding")
        return v if isinstance(v, list) else None

    @property
    def threshold_personal(self) -> Optional[float]:
        v = self.raw.get("voiceprints", {}).get("calibration", {}).get("threshold_personal")
        return float(v) if v is not None else None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.raw, f, ensure_ascii=False, indent=2)


def load_profiles(profiles_dir: Path) -> Dict[str, Profile]:
    profiles: Dict[str, Profile] = {}
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for p in profiles_dir.glob("*.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "alias" in raw:
                profiles[raw["alias"]] = Profile(path=p, raw=raw)
        except Exception:
            continue
    return profiles


def new_profile(profiles_dir: Path, alias: str, full_name: Optional[str] = None) -> Profile:
    p = profiles_dir / f"{alias}.json"
    raw = {
        "schema_version": "1.0",
        "alias": alias,
        "full_name": full_name or alias,
        "voiceprints": {
            "embedding_model_id": "spkrec_ecapa_voxceleb",
            "centroid_embedding": None,
            "sample_count": 0,
            "total_speech_seconds": 0.0,
            "anchors": [],
            "calibration": {
                "threshold_personal": None,
                "accepted_scores": []
            }
        },
        "global_features_agg": {
            "acoustic": {},
            "linguistic": {}
        },
        "history": []
    }
    prof = Profile(path=p, raw=raw)
    prof.save()
    return prof


def update_profile_with_cluster(
    profile: Profile,
    cluster_embedding: List[float],
    speech_seconds: float,
    anchors: List[Dict[str, Any]],
    episode_ref: Dict[str, Any],
    episode_features: Dict[str, Any],
    max_anchors: int = 50,
) -> None:
    vp = profile.raw.setdefault("voiceprints", {})
    old_centroid = vp.get("centroid_embedding")
    old_count = int(vp.get("sample_count", 0))
    old_speech = float(vp.get("total_speech_seconds", 0.0))

    if not isinstance(old_centroid, list):
        vp["centroid_embedding"] = cluster_embedding
        vp["sample_count"] = 1
        vp["total_speech_seconds"] = float(speech_seconds)
    else:
        # Weight by speech seconds (more speech => more reliable centroid)
        vp["centroid_embedding"] = _weighted_mean(old_centroid, cluster_embedding, old_speech, speech_seconds)
        vp["sample_count"] = old_count + 1
        vp["total_speech_seconds"] = old_speech + float(speech_seconds)

    # Keep anchors bounded
    a = vp.setdefault("anchors", [])
    if isinstance(a, list):
        a.extend(anchors)
        vp["anchors"] = a[-max_anchors:]

    # Calibration: update accepted scores (self-sim vs centroid)
    cal = vp.setdefault("calibration", {})
    acc = cal.setdefault("accepted_scores", [])
    if isinstance(acc, list) and isinstance(vp.get("centroid_embedding"), list):
        score = cosine(cluster_embedding, vp["centroid_embedding"])
        acc.append(float(score))
        acc = acc[-200:]  # bound
        cal["accepted_scores"] = acc
        # Conservative personal threshold: 10th percentile of accepted_scores minus margin
        if len(acc) >= 10:
            s = sorted(acc)
            p10 = s[max(0, int(0.10 * (len(s) - 1)))]
            cal["threshold_personal"] = max(0.0, float(p10) - 0.02)

    # Aggregate features (simple running mean by episodes)
    agg = profile.raw.setdefault("global_features_agg", {"acoustic": {}, "linguistic": {}})
    for k in ("acoustic", "linguistic"):
        src = episode_features.get(k, {})
        dst = agg.setdefault(k, {})
        if isinstance(src, dict) and isinstance(dst, dict):
            # naive merge: keep last + count; (you can replace with weighted mean later)
            for fk, fv in src.items():
                dst[fk] = fv

    # history
    hist = profile.raw.setdefault("history", [])
    if isinstance(hist, list):
        hist.append(episode_ref)
        hist[:] = hist[-200:]

    profile.save()
