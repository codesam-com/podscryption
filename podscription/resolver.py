from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from podscription.jobfile import load_job, ensure_job_defaults
from podscription.pipeline import render_srt
from podscription.profiles import load_profiles, new_profile, update_profile_with_cluster


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_unknowns(job_path: Path) -> None:
    job = load_job(job_path)
    ensure_job_defaults(job)
    raw = job.raw
    job_id = raw.get("job_id") or job_path.stem

    raw["results"]["run"]["resolve_started_at"] = datetime.now().isoformat()

    outputs = raw.get("results", {}).get("outputs", [])
    if not outputs:
        raise RuntimeError("No outputs recorded in job; run the job first.")

    # Review file convention: review/<job_id>_unknowns.yaml
    review_path = Path("review") / f"{job_id}_unknowns.yaml"
    if not review_path.exists():
        raise RuntimeError(f"Review file not found: {review_path}")

    review = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict) or "unknowns" not in review:
        raise RuntimeError("Invalid review file schema.")

    unknowns = review.get("unknowns", [])
    # Build mapping cluster_id -> alias from filled answers
    # Rule: if multiple questions filled with different aliases, treat as error.
    cluster_to_alias: Dict[str, str] = {}
    for u in unknowns:
        cid = u.get("cluster_id")
        if not cid:
            continue
        fills = []
        for q in (u.get("questions") or []):
            a = (q.get("fill_alias") or "").strip()
            if a:
                fills.append(a)
        fills = list(dict.fromkeys(fills))
        if len(fills) == 1:
            cluster_to_alias[cid] = fills[0]
        elif len(fills) > 1:
            raise RuntimeError(f"Conflicting aliases for {cid}: {fills}")

    if not cluster_to_alias:
        raise RuntimeError("No aliases filled in review file.")

    profiles_dir = Path("profiles")
    profiles = load_profiles(profiles_dir)

    # For each output episode, relabel if it had UNKNOWNs
    for o in outputs:
        out_srt = o.get("transcript_srt")
        out_diar = o.get("diarization_json")
        out_tseg = o.get("transcript_segments_json")

        if not (out_diar and out_tseg and out_srt):
            continue

        diar_path = Path(out_diar)
        seg_path = Path(out_tseg)
        if not (diar_path.exists() and seg_path.exists()):
            continue

        diar = _load_json(diar_path)
        tseg = _load_json(seg_path)
        segments = tseg.get("segments", [])
        diar_segments = diar.get("segments", [])

        # Apply mapping to diarization segments
        for ds in diar_segments:
            cid = ds.get("cluster_id")
            if cid in cluster_to_alias:
                ds["assigned_alias"] = cluster_to_alias[cid]

        # Apply mapping to transcript segments
        for s in segments:
            cid = s.get("cluster_id")
            if cid in cluster_to_alias:
                s["speaker_label"] = cluster_to_alias[cid]

        # Re-render SRT
        srt_text = render_srt(segments)
        Path(out_srt).write_text(srt_text, encoding="utf-8")

        # Update diarization clusters metadata assigned_alias + update profiles
        # We need cluster centroids from diar["clusters"]
        cluster_infos = {c["cluster_id"]: c for c in diar.get("clusters", []) if "cluster_id" in c}
        # Build anchors from transcript segments
        by_cluster = {}
        for s in segments:
            by_cluster.setdefault(s["cluster_id"], []).append(s)

        for cid, alias in cluster_to_alias.items():
            cinfo = cluster_infos.get(cid)
            if not cinfo:
                continue
            centroid = cinfo.get("centroid_embedding")
            speech_seconds = float(cinfo.get("speech_seconds", 0.0))
            if not isinstance(centroid, list) or not centroid:
                continue

            # Ensure profile exists
            prof = profiles.get(alias)
            if prof is None:
                prof = new_profile(profiles_dir, alias, full_name=alias)
                profiles[alias] = prof

            anchors = []
            for s in by_cluster.get(cid, [])[:5]:
                if len(s.get("text", "")) >= 10:
                    anchors.append({"timecode": float(s["start"]), "text": s["text"][:140]})

            # Update profile with features if available
            feat_path = Path(o.get("features_by_speaker_json", ""))
            episode_feats = {}
            if feat_path.exists():
                feat = _load_json(feat_path)
                episode_feats = feat.get("by_speaker", {}).get(alias, {})

            episode_ref = {
                "job_id": job_id,
                "item_id": diar.get("item_id"),
                "episode_slug": o.get("episode_slug")
            }
            update_profile_with_cluster(prof, centroid, speech_seconds, anchors, episode_ref, episode_feats)

        # Write updated JSON back
        diar_path.write_text(json.dumps(diar, ensure_ascii=False, indent=2), encoding="utf-8")
        seg_path.write_text(json.dumps(tseg, ensure_ascii=False, indent=2), encoding="utf-8")

    raw["status"] = "done"
    raw["results"]["run"]["resolve_finished_at"] = datetime.now().isoformat()
    job.save()
