from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import soundfile as sf
import torch
import yaml
from sklearn.cluster import AgglomerativeClustering

from faster_whisper import WhisperModel
from speechbrain.inference.speaker import EncoderClassifier

from podscription.profiles import cosine


# --------------------------
# Utility / helpers
# --------------------------

def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sanitize_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "-", s, flags=re.UNICODE)
    return s[:80].strip("-") or "episode"


def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )


def ffprobe_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr}")
    return float(p.stdout.strip())


def to_wav_mono_16k(inp: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(inp), "-ac", "1", "-ar", "16000", "-vn", str(out)])


# --------------------------
# Ingest
# --------------------------

def _requests_get_stream(url: str, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": "podscryption/0.1 (GitHub Actions)"}
    return requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)


def download_with_retries(url: str, dest: Path, max_retries: int, backoff0: int, backoff_max: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for i in range(max_retries):
        try:
            r = _requests_get_stream(url)
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return
        except Exception as e:
            last_err = e
            sleep = min(backoff_max, backoff0 * (2 ** i)) + float(np.random.rand())
            time.sleep(float(sleep))
    raise RuntimeError(f"Failed to download after {max_retries} retries: {url}\n{last_err}")


def parse_gdrive_file_id(url: str) -> Optional[str]:
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def download_gdrive_public(url: str, dest: Path, max_retries: int, backoff0: int, backoff_max: int) -> None:
    file_id = parse_gdrive_file_id(url)
    if not file_id:
        raise ValueError("Could not parse Google Drive file id from URL.")

    base = f"https://drive.google.com/uc?export=download&id={file_id}"
    sess = requests.Session()
    headers = {"User-Agent": "podscryption/0.1 (GitHub Actions)"}

    def _save(resp: requests.Response) -> None:
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    last_err = None
    for i in range(max_retries):
        try:
            resp = sess.get(base, stream=True, timeout=60, headers=headers)
            if "text/html" in resp.headers.get("Content-Type", ""):
                # confirm token for large files
                text = resp.text
                m = re.search(r"confirm=([0-9A-Za-z_]+)", text)
                if m:
                    token = m.group(1)
                    resp = sess.get(f"{base}&confirm={token}", stream=True, timeout=60, headers=headers)
            _save(resp)
            return
        except Exception as e:
            last_err = e
            sleep = min(backoff_max, backoff0 * (2 ** i)) + float(np.random.rand())
            time.sleep(float(sleep))
    raise RuntimeError(f"Failed to download GDrive after {max_retries} retries: {url}\n{last_err}")


def ingest_audio(source: Dict[str, Any], dest: Path, allow_ivoox: bool, limits: Dict[str, Any]) -> None:
    t = source.get("type")
    url = source.get("url")
    if not isinstance(url, str):
        raise ValueError("source.url must be a string")

    max_retries = int(limits.get("max_download_retries", 4))
    backoff0 = int(limits.get("backoff_initial_seconds", 5))
    backoff_max = int(limits.get("backoff_max_seconds", 60))

    if t == "direct_url":
        download_with_retries(url, dest, max_retries, backoff0, backoff_max)
        return

    if t == "gdrive_public":
        download_gdrive_public(url, dest, max_retries, backoff0, backoff_max)
        return

    if t == "ivoox_page":
        if not allow_ivoox:
            raise RuntimeError("iVoox ingest disabled (options.ingest.allow_ivoox=false).")
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["yt-dlp", "-f", "bestaudio/best", "-o", str(dest), url]
        run(cmd)
        return

    raise ValueError(f"Unknown source.type: {t}")


# --------------------------
# VAD (simple energy-based)
# --------------------------

def compute_rms_energy(x: np.ndarray, frame: int = 1600, hop: int = 800) -> np.ndarray:
    x = x.astype(np.float32)
    n = max(1, 1 + (len(x) - frame) // hop)
    e = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        f = x[s : s + frame]
        e[i] = float(np.sqrt(np.mean(f * f) + 1e-12))
    return e


def vad_intervals(wav_path: Path, sr: int = 16000, threshold_quantile: float = 0.6) -> List[Tuple[float, float]]:
    x, file_sr = sf.read(str(wav_path))
    if file_sr != sr:
        raise ValueError(f"Expected {sr}Hz wav, got {file_sr}.")
    if x.ndim > 1:
        x = x[:, 0]
    e = compute_rms_energy(x)
    thr = float(np.quantile(e, threshold_quantile))
    frame = 1600
    hop = 800
    mask = e > thr

    intervals: List[Tuple[int, int]] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m) and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(mask)))

    out: List[Tuple[float, float]] = []
    for a, b in intervals:
        s = a * hop / sr
        t = (b * hop + frame) / sr
        out.append((s, t))

    merged: List[Tuple[float, float]] = []
    for s, t in out:
        if not merged:
            merged.append((s, t))
            continue
        ps, pt = merged[-1]
        if s - pt <= 0.25:
            merged[-1] = (ps, max(pt, t))
        else:
            merged.append((s, t))

    merged = [(s, t) for s, t in merged if (t - s) >= 0.30]
    return merged


# --------------------------
# Chunk plan (VAD-guided)
# --------------------------

def build_chunk_plan(
    speech: List[Tuple[float, float]],
    target_minutes: float,
    max_minutes: float,
    split_on_silence_ms: int,
    overlap_seconds: float,
    total_duration: float,
) -> List[Tuple[float, float]]:
    target = target_minutes * 60.0
    maxd = max_minutes * 60.0
    if not speech:
        return [(0.0, min(total_duration, maxd))]

    chunks: List[Tuple[float, float]] = []
    cur_start = speech[0][0]
    cur_end = cur_start

    for (s, t) in speech:
        if s < cur_start:
            cur_start = s

        if (t - cur_start) >= target and (cur_end - cur_start) >= 60.0:
            end = cur_end
            chunks.append((max(0.0, cur_start - overlap_seconds), min(total_duration, end + overlap_seconds)))
            cur_start = s
            cur_end = t
        else:
            cur_end = t

        if (cur_end - cur_start) >= maxd:
            end = cur_start + maxd
            chunks.append((max(0.0, cur_start - overlap_seconds), min(total_duration, end + overlap_seconds)))
            cur_start = end
            cur_end = end

    if cur_end > cur_start + 10.0:
        chunks.append((max(0.0, cur_start - overlap_seconds), min(total_duration, cur_end + overlap_seconds)))

    chunks2: List[Tuple[float, float]] = []
    for s, t in sorted(chunks):
        if not chunks2:
            chunks2.append((s, t))
            continue
        ps, pt = chunks2[-1]
        if s <= pt:
            chunks2[-1] = (ps, max(pt, t))
        else:
            chunks2.append((s, t))
    return chunks2


# --------------------------
# Diarization (ECAPA embeddings + AHC clustering)
# --------------------------

@dataclass
class ClusterInfo:
    cluster_id: str
    speech_seconds: float
    centroid: List[float]
    tightness: float


def extract_embeddings(
    wav_path: Path,
    speech: List[Tuple[float, float]],
    window: float,
    hop: float,
    encoder: EncoderClassifier,
    sr: int = 16000,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Extract SpeechBrain ECAPA embeddings from VAD speech intervals.
    IMPORTANT: SpeechBrain expects torch Tensors (not numpy).
    """
    x, file_sr = sf.read(str(wav_path))
    if file_sr != sr:
        raise ValueError(f"Expected {sr}Hz wav, got {file_sr}.")
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float32)

    windows: List[Tuple[float, float]] = []
    feats: List[np.ndarray] = []

    # Make sure model is on CPU for GH runners
    encoder = encoder.to(torch.device("cpu"))

    for (s0, t0) in speech:
        s = s0
        while s + window <= t0:
            a = int(s * sr)
            b = int((s + window) * sr)
            seg = x[a:b]

            seg_t = torch.from_numpy(seg).unsqueeze(0)  # [1, T]
            if seg_t.dtype != torch.float32:
                seg_t = seg_t.float()

            # SpeechBrain returns torch tensor; convert to numpy
            emb_t = encoder.encode_batch(seg_t)
            emb = emb_t.squeeze().detach().cpu().numpy()

            feats.append(emb.astype(np.float32))
            windows.append((s, s + window))
            s += hop

    if not feats:
        # fallback: whole file (cap to 20s if enormous to avoid slow encode)
        cap_seconds = min(float(len(x) / sr), 20.0)
        x_cap = x[: int(cap_seconds * sr)]
        x_t = torch.from_numpy(x_cap).unsqueeze(0)
        if x_t.dtype != torch.float32:
            x_t = x_t.float()
        emb_t = encoder.encode_batch(x_t)
        emb = emb_t.squeeze().detach().cpu().numpy()
        return np.expand_dims(emb.astype(np.float32), 0), [(0.0, cap_seconds)]

    return np.stack(feats, axis=0), windows


def diarize(
    embeddings: np.ndarray,
    windows: List[Tuple[float, float]],
    max_speakers: int,
) -> Tuple[List[Dict[str, Any]], List[ClusterInfo]]:
    best_k = 1
    best_score = float("inf")

    E = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

    def tightness(labels: np.ndarray) -> float:
        val = 0.0
        for k in np.unique(labels):
            idx = np.where(labels == k)[0]
            if len(idx) <= 1:
                continue
            c = E[idx].mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            d = 1.0 - (E[idx] @ c)
            val += float(d.mean())
        return val / max(1, len(np.unique(labels)))

    for k in range(1, max_speakers + 1):
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = model.fit_predict(E)
        t = tightness(labels)
        score = t + 0.02 * k
        if score < best_score:
            best_score = score
            best_k = k

    model = AgglomerativeClustering(n_clusters=best_k, metric="cosine", linkage="average")
    labels = model.fit_predict(E)

    per_win = []
    for (s, t), lab in zip(windows, labels):
        per_win.append({"start": float(s), "end": float(t), "cluster_id": f"SPEAKER_{int(lab)}"})

    merged: List[Dict[str, Any]] = []
    for seg in per_win:
        if not merged:
            merged.append(seg)
            continue
        last = merged[-1]
        if seg["cluster_id"] == last["cluster_id"] and seg["start"] <= last["end"] + 0.15:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(seg)

    clusters: List[ClusterInfo] = []
    for lab in sorted(set(labels.tolist())):
        idx = np.where(labels == lab)[0]
        cid = f"SPEAKER_{int(lab)}"
        centroid = E[idx].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        spk_secs = sum((s["end"] - s["start"]) for s in merged if s["cluster_id"] == cid)
        d = 1.0 - (E[idx] @ centroid)
        tight = float(d.mean()) if len(idx) > 0 else 1.0
        clusters.append(
            ClusterInfo(
                cluster_id=cid,
                speech_seconds=float(spk_secs),
                centroid=centroid.astype(float).tolist(),
                tightness=tight,
            )
        )
    return merged, clusters


# --------------------------
# ASR (faster-whisper) + language detection
# --------------------------

def detect_language_quick(model: WhisperModel, wav_path: Path, duration: float) -> Tuple[str, float]:
    pts = [
        min(30.0, max(0.0, duration * 0.05)),
        min(duration - 25.0, max(0.0, duration * 0.50)),
        min(duration - 25.0, max(0.0, duration * 0.90)),
    ]
    langs = []
    for s in pts:
        segments, info = model.transcribe(
            str(wav_path),
            vad_filter=False,
            beam_size=1,
            language=None,
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=None,
            word_timestamps=False,
            vad_parameters=None,
            clip_timestamps=f"{s},{s+20.0}",
        )
        langs.append((info.language or "unknown", float(info.language_probability or 0.0)))

    agg: Dict[str, float] = {}
    for l, p in langs:
        agg[l] = agg.get(l, 0.0) + p
    best = max(agg.items(), key=lambda kv: kv[1])
    return best[0], float(best[1] / max(1, len(langs)))


def transcribe_chunks(
    model: WhisperModel,
    wav_path: Path,
    chunks: List[Tuple[float, float]],
    beam_size: int,
    language_mode: str,
    per_chunk_language: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    lang_summary: Dict[str, float] = {}

    for i, (s, t) in enumerate(chunks):
        clip = f"{s},{t}"
        lang = None
        if language_mode in ("es", "en"):
            lang = language_mode
        elif per_chunk_language and language_mode in ("auto", "mixed"):
            lang = None

        segments, info = model.transcribe(
            str(wav_path),
            vad_filter=False,
            beam_size=beam_size,
            language=lang,
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=True,
            word_timestamps=False,
            clip_timestamps=clip,
        )

        detected = info.language or "unknown"
        prob = float(info.language_probability or 0.0)
        lang_summary[detected] = lang_summary.get(detected, 0.0) + prob

        for seg in segments:
            out.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": (seg.text or "").strip(),
                    "language": detected,
                    "confidence": prob,
                    "chunk_id": f"chunk_{i:03d}",
                }
            )

    total = sum(lang_summary.values()) + 1e-9
    for k in list(lang_summary.keys()):
        lang_summary[k] /= total
    return out, {"language_distribution": lang_summary}


# --------------------------
# Alignment diarization <-> ASR
# --------------------------

def assign_cluster_to_asr_segment(asr_seg: Dict[str, Any], diar_segments: List[Dict[str, Any]]) -> str:
    s, t = float(asr_seg["start"]), float(asr_seg["end"])
    if t <= s:
        return "SPEAKER_0"
    overlaps: Dict[str, float] = {}
    for d in diar_segments:
        ds, dt = float(d["start"]), float(d["end"])
        inter = max(0.0, min(t, dt) - max(s, ds))
        if inter > 0:
            overlaps[d["cluster_id"]] = overlaps.get(d["cluster_id"], 0.0) + inter
    if not overlaps:
        return diar_segments[0]["cluster_id"] if diar_segments else "SPEAKER_0"
    best = max(overlaps.items(), key=lambda kv: kv[1])[0]
    return best


# --------------------------
# Speaker ID (strict) + intro name mining
# --------------------------

INTRO_PATTERNS_ES = [
    r"\bsoy\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})\b",
    r"\bme\s+llamo\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})\b",
    r"\bcon\s+nosotros\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})\b",
    r"\bhoy\s+nos\s+acompaña\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]+){0,2})\b",
]
INTRO_PATTERNS_EN = [
    r"\bi'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
    r"\bmy\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
    r"\bjoined\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
]


def mine_intro_names(transcript: List[Dict[str, Any]], first_minutes: int) -> List[Tuple[str, float, str]]:
    out = []
    horizon = first_minutes * 60.0
    for seg in transcript:
        if float(seg["start"]) > horizon:
            break
        text = seg.get("text", "")
        cid = seg.get("cluster_id", "")
        for pat in INTRO_PATTERNS_ES + INTRO_PATTERNS_EN:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                out.append((name, float(seg["start"]), cid))
    return out


def strict_speaker_id(
    clusters: List[ClusterInfo],
    profiles: Dict[str, Any],
    global_threshold: float,
    min_margin: float,
    intro_evidence: List[Tuple[str, float, str]],
) -> Dict[str, Dict[str, Any]]:
    name_to_alias = {}
    for alias, prof in profiles.items():
        fn = (prof.raw.get("full_name") or alias).lower()
        name_to_alias[alias.lower()] = alias
        name_to_alias[fn] = alias

    intro_suggest: Dict[str, str] = {}
    names_found: Dict[str, List[str]] = {}
    for name, _, cid in intro_evidence:
        key = name.lower()
        if key in name_to_alias:
            intro_suggest[cid] = name_to_alias[key]
        names_found.setdefault(cid, []).append(name)

    result: Dict[str, Dict[str, Any]] = {}
    centroids = {}
    thresholds = {}
    for alias, prof in profiles.items():
        if prof.centroid:
            centroids[alias] = prof.centroid
            thresholds[alias] = prof.threshold_personal

    for c in clusters:
        scores = []
        for alias, cent in centroids.items():
            sc = cosine(c.centroid, cent)
            scores.append((alias, float(sc)))
        scores.sort(key=lambda kv: kv[1], reverse=True)

        best_alias, best_score = (scores[0] if scores else ("", 0.0))
        second_alias, second_score = (scores[1] if len(scores) > 1 else ("", 0.0))

        personal_thr = thresholds.get(best_alias)
        thr = max(global_threshold, float(personal_thr)) if personal_thr is not None else float(global_threshold)

        suggested = intro_suggest.get(c.cluster_id)
        if suggested and suggested == best_alias and best_score >= (thr - 0.02) and (best_score - second_score) >= (min_margin * 0.5):
            thr_eff = thr - 0.01
        else:
            thr_eff = thr

        if best_score >= thr_eff and (best_score - second_score) >= min_margin:
            decision = "assigned"
            assigned = best_alias
        else:
            decision = "unknown"
            assigned = None

        result[c.cluster_id] = {
            "assigned_alias": assigned,
            "best_alias": best_alias,
            "best_score": best_score,
            "second_best_alias": second_alias,
            "second_best_score": second_score,
            "threshold_used": thr_eff,
            "margin_used": min_margin,
            "decision": decision,
            "names_found": names_found.get(c.cluster_id, []),
        }

    return result


# --------------------------
# SRT rendering
# --------------------------

def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render_srt(transcript: List[Dict[str, Any]], max_chars: int = 84) -> str:
    blocks = []
    cur = None

    def flush():
        nonlocal cur
        if cur:
            blocks.append(cur)
            cur = None

    for seg in transcript:
        label = seg["speaker_label"]
        s, t = float(seg["start"]), float(seg["end"])
        text = seg["text"].strip()
        if not text:
            continue
        if cur is None:
            cur = {"start": s, "end": t, "label": label, "text": text}
            continue
        if label == cur["label"] and s <= cur["end"] + 0.6 and (t - cur["start"]) <= 7.0:
            cur["end"] = max(cur["end"], t)
            cur["text"] = (cur["text"] + " " + text).strip()
        else:
            flush()
            cur = {"start": s, "end": t, "label": label, "text": text}

    flush()

    lines = []
    idx = 1
    for b in blocks:
        txt = f"{b['label']}: {b['text']}"
        wrapped = []
        while len(txt) > max_chars:
            cut = txt.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            wrapped.append(txt[:cut])
            txt = txt[cut:].lstrip()
        wrapped.append(txt)

        lines.append(str(idx))
        lines.append(f"{format_srt_time(b['start'])} --> {format_srt_time(b['end'])}")
        lines.extend(wrapped)
        lines.append("")
        idx += 1

    return "\n".join(lines)


# --------------------------
# Features (lightweight)
# --------------------------

MUletillas_ES = ["eh", "este", "o sea", "vale", "pues", "en plan", "¿sabes?", "digamos"]
MUletillas_EN = ["um", "uh", "like", "you know", "i mean", "well"]


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\wáéíóúñü]+(?:'[\w]+)?\b", text.lower(), flags=re.UNICODE)


def linguistic_features(texts: List[str]) -> Dict[str, Any]:
    full = " ".join(texts)
    toks = tokenize(full)
    if not toks:
        return {"token_count": 0}
    types = set(toks)
    ttr = len(types) / max(1, len(toks))

    low = full.lower()
    mu_es = {m: low.count(m) for m in MUletillas_ES}
    mu_en = {m: low.count(m) for m in MUletillas_EN}

    freq = {}
    for w in toks:
        if len(w) <= 2:
            continue
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:15]
    return {
        "token_count": len(toks),
        "type_count": len(types),
        "ttr": round(float(ttr), 4),
        "muletillas_es": mu_es,
        "muletillas_en": mu_en,
        "top_terms": top,
    }


def acoustic_features_from_vad(speech: List[Tuple[float, float]], total_duration: float) -> Dict[str, Any]:
    speech_secs = sum(t - s for s, t in speech)
    silence_secs = max(0.0, total_duration - speech_secs)
    return {
        "speech_seconds": round(float(speech_secs), 3),
        "silence_seconds": round(float(silence_secs), 3),
        "speech_ratio": round(float(speech_secs / max(1e-9, total_duration)), 4),
        "mean_segment_seconds": round(float((speech_secs / max(1, len(speech)))), 3),
    }


# --------------------------
# Review file
# --------------------------

def build_review_unknowns(
    job_id: str,
    unknown_clusters: List[str],
    transcript: List[Dict[str, Any]],
    cluster_matches: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    unknowns = []
    by_cluster = {}
    for seg in transcript:
        by_cluster.setdefault(seg["cluster_id"], []).append(seg)

    for i, cid in enumerate(unknown_clusters, start=1):
        segs = by_cluster.get(cid, [])
        picks = []
        for s in segs[:]:
            if len(picks) >= 2:
                break
            if len(s["text"]) >= 8:
                picks.append(s)
        if segs:
            picks.append(segs[min(len(segs) - 1, max(0, len(segs) // 2))])

        questions = []
        for p in picks[:3]:
            questions.append(
                {
                    "timecode": format_srt_time(float(p["start"])).replace(",", "."),
                    "text": p["text"][:140],
                    "fill_alias": "",
                }
            )

        cand = []
        vm = cluster_matches.get(cid, {})
        if vm.get("best_alias"):
            cand.append({"alias": vm["best_alias"], "score": vm.get("best_score", 0.0)})
        if vm.get("second_best_alias"):
            cand.append({"alias": vm["second_best_alias"], "score": vm.get("second_best_score", 0.0)})

        unknowns.append(
            {
                "unknown_id": f"UNKNOWN_{i}",
                "cluster_id": cid,
                "candidates": cand,
                "questions": questions,
            }
        )

    payload = {
        "schema_version": "1.0",
        "job_id": job_id,
        "status": "needs_user_input",
        "unknowns": unknowns,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


# --------------------------
# Main per-episode processing
# --------------------------

def process_episode(
    job_id: str,
    item: Dict[str, Any],
    options: Dict[str, Any],
    limits: Dict[str, Any],
    profiles_dir: Path,
    outputs_root: Path,
    review_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    item_id = item["item_id"]
    podcast = item.get("podcast", "podcast")
    episode_title = item.get("episode_title", "episode")
    episode_slug = sanitize_slug(episode_title)

    out_dir = outputs_root / sanitize_slug(podcast) / episode_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    wdir = work_root / job_id / item_id
    wdir.mkdir(parents=True, exist_ok=True)

    audio_in = wdir / "audio.input"
    audio_wav = wdir / "audio.wav"

    ingest_cfg = options.get("ingest", {})
    allow_ivoox = bool(ingest_cfg.get("allow_ivoox", False))
    ingest_audio(item["source"], audio_in, allow_ivoox, limits)

    max_bytes = int(limits.get("max_audio_bytes", 2500000000))
    if audio_in.stat().st_size > max_bytes:
        raise RuntimeError(f"Audio too large ({audio_in.stat().st_size} bytes) > limit {max_bytes}")

    to_wav_mono_16k(audio_in, audio_wav)
    dur = ffprobe_duration_seconds(audio_wav)

    speech = vad_intervals(audio_wav, threshold_quantile=0.6)

    ch = options.get("asr", {}).get("chunking", {})
    chunks = build_chunk_plan(
        speech=speech,
        target_minutes=float(ch.get("target_minutes", 8)),
        max_minutes=float(ch.get("max_minutes", 10)),
        split_on_silence_ms=int(ch.get("split_on_silence_ms", 900)),
        overlap_seconds=float(ch.get("overlap_seconds", 0.5)),
        total_duration=dur,
    )

    diar_cfg = options.get("diarization", {})
    emb_model_id = diar_cfg.get("embedding_model_id", "spkrec_ecapa_voxceleb")
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    window = float(diar_cfg.get("window_seconds", 3.0))
    hop = float(diar_cfg.get("hop_seconds", 1.5))
    E, win = extract_embeddings(audio_wav, speech, window, hop, encoder)
    max_spk = int(diar_cfg.get("clustering", {}).get("max_speakers", 6))
    diar_segments, cluster_infos = diarize(E, win, max_spk)

    asr_cfg = options.get("asr", {})
    beam = int(asr_cfg.get("beam_size", 5))
    language_hint = str(item.get("language_hint", "auto"))
    bilingual = asr_cfg.get("bilingual", {})
    per_chunk_language = bool(bilingual.get("per_chunk_language", True))
    detect_language = bool(bilingual.get("detect_language", True))

    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    if detect_language and language_hint == "auto":
        detected_lang, conf = detect_language_quick(model, audio_wav, dur)
        if detected_lang in ("es", "en"):
            language_mode = detected_lang
            lang_conf = conf
        else:
            language_mode = "auto"
            lang_conf = conf
    else:
        language_mode = language_hint if language_hint in ("es", "en", "mixed") else "auto"
        lang_conf = 0.0

    asr_segments, lang_meta = transcribe_chunks(model, audio_wav, chunks, beam, language_mode, per_chunk_language)

    transcript = []
    for seg in asr_segments:
        cid = assign_cluster_to_asr_segment(seg, diar_segments)
        transcript.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"],
                "language": seg.get("language", "unknown"),
                "confidence": float(seg.get("confidence", 0.0)),
                "cluster_id": cid,
            }
        )

    from podscription.profiles import load_profiles

    profiles = load_profiles(profiles_dir)

    spk_cfg = options.get("speaker_id", {})
    global_thr = float(spk_cfg.get("global_threshold", 0.74))
    margin = float(spk_cfg.get("min_margin_to_second_best", 0.06))
    intro_cfg = spk_cfg.get("intro_name_mining", {})
    intro_enabled = bool(intro_cfg.get("enabled", True))
    first_minutes = int(intro_cfg.get("first_minutes", 10))

    intro_evidence = mine_intro_names(transcript, first_minutes) if intro_enabled else []
    matches = strict_speaker_id(cluster_infos, profiles, global_thr, margin, intro_evidence)

    unknown_clusters: List[str] = []
    unk_i = 1
    for c in sorted(cluster_infos, key=lambda z: z.cluster_id):
        m = matches.get(c.cluster_id, {})
        if m.get("decision") == "assigned" and m.get("assigned_alias"):
            label = str(m["assigned_alias"])
        else:
            label = f"UNKNOWN_{unk_i}"
            unknown_clusters.append(c.cluster_id)
            unk_i += 1
        for seg in diar_segments:
            if seg["cluster_id"] == c.cluster_id:
                seg["assigned_alias"] = label

    for seg in transcript:
        seg["speaker_label"] = next(
            (d.get("assigned_alias") for d in diar_segments if d["cluster_id"] == seg["cluster_id"]),
            seg["cluster_id"],
        )

    by_speaker_text: Dict[str, List[str]] = {}
    for seg in transcript:
        by_speaker_text.setdefault(seg["speaker_label"], []).append(seg["text"])

    features = {"schema_version": "1.0", "job_id": job_id, "item_id": item_id, "by_speaker": {}}
    for speaker_label, texts in by_speaker_text.items():
        features["by_speaker"][speaker_label] = {
            "acoustic": acoustic_features_from_vad(speech, dur),
            "linguistic": linguistic_features(texts),
        }

    manifest = {
        "schema_version": "1.0",
        "job_id": job_id,
        "item_id": item_id,
        "podcast": podcast,
        "episode_title": episode_title,
        "episode_slug": episode_slug,
        "source": item["source"],
        "audio": {"duration_seconds": dur, "sample_rate_hz": 16000, "sha256": sha256_file(audio_in)},
        "models": {"asr_model_id": "asr_fasterwhisper_large_v3", "speaker_embedding_model_id": emb_model_id},
        "language": {"mode": language_mode, "detected": (lang_meta.get("language_distribution") or {}), "confidence": lang_conf},
        "chunk_plan": [{"chunk_id": f"chunk_{i:03d}", "start": float(s), "end": float(t)} for i, (s, t) in enumerate(chunks)],
    }
    (out_dir / "episode_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    diar_json = {"schema_version": "1.0", "job_id": job_id, "item_id": item_id, "segments": diar_segments, "clusters": []}
    for ci in cluster_infos:
        m = matches.get(ci.cluster_id, {})
        diar_json["clusters"].append(
            {
                "cluster_id": ci.cluster_id,
                "speech_seconds": ci.speech_seconds,
                "centroid_embedding": ci.centroid,
                "tightness": ci.tightness,
                "voice_match": {
                    "best_alias": m.get("best_alias"),
                    "best_score": m.get("best_score"),
                    "second_best_alias": m.get("second_best_alias"),
                    "second_best_score": m.get("second_best_score"),
                    "threshold_used": m.get("threshold_used"),
                    "margin_used": m.get("margin_used"),
                    "decision": m.get("decision"),
                },
                "text_evidence": {"names_found": m.get("names_found", []), "notes": []},
            }
        )
    (out_dir / "diarization.json").write_text(json.dumps(diar_json, ensure_ascii=False, indent=2), encoding="utf-8")

    tjson = {"schema_version": "1.0", "job_id": job_id, "item_id": item_id, "segments": transcript}
    (out_dir / "transcript_segments.json").write_text(json.dumps(tjson, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "features_by_speaker.json").write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "transcript.srt").write_text(render_srt(transcript), encoding="utf-8")

    review_path = ""
    if unknown_clusters:
        review_file = review_root / f"{job_id}_unknowns.yaml"
        build_review_unknowns(job_id, unknown_clusters, transcript, matches, review_file)
        review_path = str(review_file)

    from podscription.profiles import update_profile_with_cluster, new_profile

    episode_ref = {"job_id": job_id, "item_id": item_id, "podcast": podcast, "episode_slug": episode_slug, "duration_seconds": dur}
    cluster_to_label = {}
    for d in diar_segments:
        cluster_to_label[d["cluster_id"]] = d.get("assigned_alias") or d["cluster_id"]

    ci_by_id = {c.cluster_id: c for c in cluster_infos}
    for cid, label in cluster_to_label.items():
        if label.startswith("UNKNOWN_"):
            continue
        prof = profiles.get(label)
        if prof is None:
            prof = new_profile(profiles_dir, label, full_name=label)
            profiles[label] = prof
        ci = ci_by_id.get(cid)
        if not ci:
            continue
        anchors = []
        for seg in transcript:
            if seg["cluster_id"] == cid and len(anchors) < 5 and len(seg["text"]) >= 10:
                anchors.append({"timecode": float(seg["start"]), "text": seg["text"][:140], "episode": episode_ref})
        episode_feats = features["by_speaker"].get(label, {})
        update_profile_with_cluster(prof, ci.centroid, ci.speech_seconds, anchors, episode_ref, episode_feats)

    return {
        "item_id": item_id,
        "podcast": podcast,
        "episode_slug": str(out_dir.relative_to(outputs_root)),
        "transcript_srt": str((out_dir / "transcript.srt")),
        "diarization_json": str((out_dir / "diarization.json")),
        "features_by_speaker_json": str((out_dir / "features_by_speaker.json")),
        "transcript_segments_json": str((out_dir / "transcript_segments.json")),
        "review_unknowns_yaml": review_path,
    }
