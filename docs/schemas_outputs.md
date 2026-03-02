# Output Schemas (v1.0)

This repo writes small, versioned JSON outputs (no audio, no artifacts):

- outputs/<podcast>/<episode_slug>/episode_manifest.json
- outputs/<podcast>/<episode_slug>/diarization.json
- outputs/<podcast>/<episode_slug>/transcript_segments.json
- outputs/<podcast>/<episode_slug>/features_by_speaker.json
- outputs/<podcast>/<episode_slug>/transcript.srt

JSON Schemas live in /schemas.

Design goals:
- Reproducible: manifest stores model IDs and chunk plan.
- Resolve without re-ASR: transcript_segments + diarization clusters carry enough data to relabel and re-render.
- Auditable: diarization.clusters[*].voice_match stores scores/margins/decision reasons.
