from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from podscription.jobfile import load_job, ensure_job_defaults
from podscription.pipeline import render_srt, format_srt_time
from podscription.profiles import load_profiles, new_profile, update_profile_with_cluster


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _majority_vote(aliases: List[str]) -> Tuple[str, Dict[str, int]]:
    """
    Returns (majority_alias, counts).
    Tie-break: pick lexicographically smallest to be deterministic.
    """
    counts = Counter([a.strip() for a in aliases if a and a.strip()])
    if not counts:
        return ("", {})
    top = counts.most_common()
    best_n = top[0][1]
    tied = sorted([a for a, n in top if n == best_n])
    return (tied[0], dict(counts))


def resolve_unknowns(job_path: Path) -> None:
    job = load_job(job_path)
    ensure_job_defaults(job)
    raw = job.raw
    job_id = raw.get("job_id") or job_path.stem

    raw["results"]["run"]["resolve_started_at"] = datetime.now().isoformat()

    outputs = raw.get("results", {}).get("outputs", [])
    if not outputs:
        raise RuntimeError("No outputs recorded in job; run the job first.")

    review_path = Path("review") / f"{job_id}_unknowns.yaml"
    if not review_path.exists():
        raise RuntimeError(f"Review file not found: {review_path}")

    review = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict) or "unknowns" not in review:
        raise RuntimeError("Invalid review file schema.")

    unknowns = review.get("unknowns", [])
    if not isinstance(unknowns, list):
        raise RuntimeError("Invalid review file: unknowns must be a list.")

    # Build mapping cluster_id -> majority alias
    cluster_to_alias: Dict[str, str] = {}

    # Collect conflicts to produce round2 review
    round2_conflicts: List[Dict[str, Any]] = []

    for u in unknowns:
        cid = u.get("cluster_id")
        if not cid:
            continue

        qlist = u.get("questions") or []
        if not isinstance(qlist, list):
            continue

        fills: List[str] = []
        by_question: List[Tuple[Dict[str, Any], str]] = []
        for q in qlist:
            if not isinstance(q, dict):
                continue
            a = (q.get("fill_alias") or "").strip()
            if a:
                fills.append(a)
                by_question.append((q, a))

        if not fills:
            continue

        maj, counts = _majority_vote(fills)
        if not maj:
            continue
        cluster_to_alias[cid] = maj

        # Anything not equal to majority goes into round2 conflicts
        for q, a in by_question:
            if a != maj:
                round2_conflicts.append({
                    "cluster_id": cid,
                    "majority_alias": maj,
                    "chosen_alias": a,
                    "timecode": q.get("timecode") or "",
                    "text": q.get("text") or ""
                })

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

        # Update diarization clusters metadata + update profiles
        cluster_infos = {c["cluster_id"]: c for c in diar.get("clusters", []) if "cluster_id" in c}

        # Anchors from transcript segments
        by_cluster: Dict[str, List[Dict[str, Any]]] = {}
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

            # Episode features if available
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
        _dump_json(diar_path, diar)
        _dump_json(seg_path, tseg)

    # If conflicts exist, write round2 review
    round2_path = Path("review") / f"{job_id}_unknowns_round2.yaml"
    if round2_conflicts:
        payload = {
            "schema_version": "1.0",
            "job_id": job_id,
            "status": "needs_user_input",
            "purpose": "Conflicts found within same diarization cluster. Majority vote applied for now. "
                       "If you confirm the outlier alias, it implies cluster split (not yet automatic).",
            "conflicts": round2_conflicts,
            "instructions": [
                "For each conflict, set fill_alias to either the majority_alias (accept) or another alias (requires split).",
                "If you choose another alias, keep it consistent across same cluster_id/timecodes you want reassigned."
            ],
            "fill": [
                {"cluster_id": c["cluster_id"], "timecode": c["timecode"], "text": c["text"], "fill_alias": ""}
                for c in round2_conflicts
            ]
        }
        round2_path.parent.mkdir(parents=True, exist_ok=True)
        round2_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

        raw["status"] = "awaiting_review"
        raw["results"]["notes"].append(
            f"Round2 review generated due to cluster conflicts: {round2_path.as_posix()}"
        )
    else:
        raw["status"] = "done"

    raw["results"]["run"]["resolve_finished_at"] = datetime.now().isoformat()
    job.save()
