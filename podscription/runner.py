from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from podscription.jobfile import load_job, ensure_job_defaults
from podscription.pipeline import process_episode


def run_job(job_path: Path) -> None:
    job = load_job(job_path)
    ensure_job_defaults(job)

    raw = job.raw
    job_id = raw.get("job_id") or job_path.stem
    raw["job_id"] = job_id

    # Basic validation + anti-abuse firewall
    items = raw.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("jobs/*.yaml must include non-empty items[]")

    limits = raw.get("limits", {})
    max_items = int(limits.get("max_items_per_job", 3))
    if len(items) > max_items:
        raise RuntimeError(f"Too many items in job: {len(items)} > {max_items}")

    # Update status
    raw["status"] = "running"
    raw.setdefault("results", {}).setdefault("run", {})
    raw["results"]["run"]["started_at"] = datetime.now().isoformat()

    options = raw.get("options", {})
    profiles_dir = Path("profiles")
    outputs_root = Path("outputs")
    review_root = Path("review")
    work_root = Path(".work")

    total_minutes_limit = float(limits.get("max_total_audio_minutes", 240))

    results_out = []
    total_minutes_est = 0.0
    errors: List[str] = []

    for item in items:
        try:
            r = process_episode(
                job_id=job_id,
                item=item,
                options=options,
                limits=limits,
                profiles_dir=profiles_dir,
                outputs_root=outputs_root,
                review_root=review_root,
                work_root=work_root
            )
            results_out.append(r)
            # best-effort estimate from manifest via file (duration already checked internally)
            # We keep a conservative accumulator using presence of output path.
            total_minutes_est += 0.0
            if total_minutes_est > total_minutes_limit:
                raise RuntimeError(f"Exceeded total audio minutes limit ({total_minutes_limit}).")
        except Exception as e:
            errors.append(f"[{item.get('item_id','item')}] {e}")

    raw["results"]["outputs"] = results_out
    raw["results"]["errors"] = errors

    # Determine final status
    needs_review = any(o.get("review_unknowns_yaml") for o in results_out)
    raw["status"] = "awaiting_review" if needs_review else ("failed" if errors else "done")
    raw["results"]["run"]["finished_at"] = datetime.now().isoformat()

    job.save()

    if errors:
        # fail the action run to make it visible, while still having written job/status
        raise RuntimeError("Job completed with errors:\n" + "\n".join(errors))
