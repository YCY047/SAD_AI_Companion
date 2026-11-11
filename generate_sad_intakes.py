"""Generate draft, synthetic SAD-aware intake forms from a PatternReframe shortlist.

Run inside sad_patient_starter:
  python generate_sad_intakes.py --dry-run --limit 1
  python generate_sad_intakes.py --limit 3

Requires: pip install openai
Environment (for an OpenAI-compatible Qwen server):
  QWEN_BASE_URL, QWEN_MODEL, and optionally QWEN_API_KEY.

IMPORTANT: Generated scenarios are synthetic research fixtures, not patient records,
validated SAD diagnoses, or evidence of clinical effectiveness.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_LOCATIONS = [
    PROJECT_ROOT / "outputs" / "SAD_pattern_shortlist_20.csv",
    PROJECT_ROOT / "data" / "SAD_pattern_shortlist_20.csv",
    PROJECT_ROOT / "SAD_pattern_shortlist_20.csv",
]

SYSTEM_PROMPT = """You create SYNTHETIC, FICTIONAL user intake profiles for research evaluating an AI companion for winter-related low mood. You are NOT diagnosing, counseling, or treating anyone.

Produce ONE believable fictional adult profile grounded in the supplied PatternReframe persona, unhelpful thought and cognitive pattern, and SAD-related contextual constraints.
- Preserve the source persona's important non-sensitive life context and the named cognitive pattern. Keep the winter adaptation natural, not a literal word swap.
- Use the provided symptoms only if they fit the scenario; do not invent extra severe symptoms, suicidal content, mania, diagnoses, medications, or treatment history.
- Keep confirmed SAD diagnosis as NOT ESTABLISHED. Winter-related low mood alone does not establish SAD.
- Seasonal recurrence should be 'unknown' unless the input explicitly establishes repeated seasonal episodes.
- Avoid unsupported assumptions about age, gender, race, religion, and clinical history. You may invent a plausible adult age if needed, but identify any added detail as synthetic.
- Do not include psychological jargon such as the cognitive pattern name in the character's first-person speech.
- Avoid instant improvement, artificial eagerness, and excessive disclosure.
- Do NOT write a therapist response, CBT treatment plan, or ideal answer for a support agent.
- Return ONLY one valid JSON object. No markdown or commentary.

Required JSON schema:
{
  "profile_name": "fictional short name",
  "age": 18,
  "occupation_or_role": "string",
  "personal_background": "1-3 sentences, consistent with persona",
  "presenting_concern": "1-2 sentences describing current winter-related difficulty",
  "winter_context": "string",
  "symptoms": ["string"],
  "duration_or_onset": "string or unknown",
  "seasonal_recurrence": "unknown",
  "daily_life_impact": "string",
  "unhelpful_thought": "natural thought consistent with supplied pattern",
  "thought_pattern": "copy input pattern exactly",
  "support_preference": "e.g. wants listening before suggestions",
  "disclosure_style": "gradual",
  "diagnosis_status": "not established",
  "initial_utterance": "1-2 natural conversational sentences; no clinical jargon",
  "additional_private_context": ["2-3 details to reveal only if asked"]
}
"""


def get_input_path(input_arg: str | None) -> Path:
    if input_arg:
        path = Path(input_arg).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find input CSV: {path}")
        return path
    for path in DEFAULT_INPUT_LOCATIONS:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Put SAD_pattern_shortlist_20.csv in outputs/, data/, or the project root, "
        "or pass --input path/to/file.csv"
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"source_index", "persona", "thought", "pattern"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")
    return rows


def build_user_prompt(row: dict[str, str]) -> str:
    fields = {
        "source_index": row.get("source_index", ""),
        "source_persona": row.get("persona", ""),
        "source_thought": row.get("thought", ""),
        "source_pattern": row.get("pattern", ""),
        "pattern_definition": row.get("pattern_def", ""),
        "proposed_winter_context": row.get("seasonal_context", ""),
        "proposed_symptoms": row.get("sad_related_symptoms", ""),
        "adaptation_direction": row.get("adaptation_direction", ""),
        "review_caution": row.get("review_caution", ""),
    }
    return (
        "Generate exactly one synthetic SAD-aware intake profile from the "
        "following source record. The input is NOT a clinical patient record. "
        "Treat symptoms and proposed adaptation as context, not a diagnosis.\n\n"
        + json.dumps(fields, ensure_ascii=False, indent=2)
    )


def parse_json(text: str) -> dict:
    txt = text.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
    txt = re.sub(r"\s*```$", "", txt)
    try:
        result = json.loads(txt)
    except json.JSONDecodeError:
        first, last = txt.find("{"), txt.rfind("}")
        if first < 0 or last <= first:
            raise
        result = json.loads(txt[first : last + 1])
    if not isinstance(result, dict):
        raise ValueError("Model response is not a JSON object")
    return result


def check_draft(profile: dict, row: dict[str, str]) -> list[str]:
    warnings = []
    required = [
        "presenting_concern", "winter_context", "symptoms", "daily_life_impact",
        "unhelpful_thought", "thought_pattern", "initial_utterance",
        "diagnosis_status", "seasonal_recurrence", "disclosure_style",
    ]
    for key in required:
        if key not in profile:
            warnings.append(f"missing:{key}")
    if profile.get("thought_pattern") != row.get("pattern"):
        warnings.append("thought_pattern_mismatch")
    if str(profile.get("diagnosis_status", "")).lower() != "not established":
        warnings.append("diagnosis_claim_check")
    if str(profile.get("seasonal_recurrence", "")).lower() != "unknown":
        warnings.append("recurrence_needs_evidence")
    if not isinstance(profile.get("symptoms"), list):
        warnings.append("symptoms_not_list")
    if any(w.lower() in str(profile.get("initial_utterance", "")).lower()
           for w in ["catastrophizing", "overgeneralization", "cognitive distortion"]):
        warnings.append("clinical_jargon_in_utterance")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input shortlist CSV (otherwise auto-locate)")
    parser.add_argument("--output", default="outputs/sad_intakes_draft.jsonl")
    parser.add_argument("--limit", type=int, default=3,
                        help="Number of source records to process; use 20 for the full shortlist")
    parser.add_argument("--dry-run", action="store_true", help="Print first prompt, do not call Qwen")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Maximum generated tokens; Qwen thinking may require more")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Request disabled thinking for compatible Qwen/vLLM servers")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    source = get_input_path(args.input)
    rows = load_rows(source)[: args.limit]
    if not rows:
        raise ValueError(f"No rows in {source}")
    print(f"Input: {source} | Cases: {len(rows)}")
    if args.dry_run:
        print("\n=== SYSTEM PROMPT ===\n", SYSTEM_PROMPT)
        print("\n=== FIRST USER PROMPT ===\n", build_user_prompt(rows[0]))
        return

    from openai import OpenAI

    endpoint = os.getenv("QWEN_BASE_URL")
    model = os.getenv("QWEN_MODEL")
    if not endpoint or not model:
        raise RuntimeError("Set QWEN_BASE_URL and QWEN_MODEL before calling Qwen")
    client = OpenAI(
        base_url=endpoint,
        api_key=os.getenv("QWEN_API_KEY", "EMPTY"),
        timeout=120,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    # For reproducibility, write one independent output file for each run.
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose another --output to avoid overwriting."
        )

    with output.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            try:
                request = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(row)},
                    ],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                }
                if args.no_thinking:
                    request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                res = client.chat.completions.create(**request)
                message = res.choices[0].message
                raw = message.content or ""
                finish_reason = res.choices[0].finish_reason
                print(f"[{idx}/{len(rows)}] finish_reason={finish_reason}; content_chars={len(raw)}")
                if not raw.strip():
                    raise ValueError(
                        f"Qwen returned empty content (finish_reason={finish_reason}). "
                        "Try --no-thinking if supported, or increase --max-tokens."
                    )
                try:
                    profile = parse_json(raw)
                except (json.JSONDecodeError, ValueError):
                    debug_path = output.with_name(
                        f"{output.stem}_debug_{row.get('source_index', idx)}.txt"
                    )
                    debug_path.write_text(
                        f"finish_reason={finish_reason}\n"
                        f"model={model}\n"
                        f"reasoning_content={getattr(message, 'reasoning_content', None)}\n"
                        f"content=\n{raw}",
                        encoding="utf-8",
                    )
                    print(f"Raw Qwen response saved to: {debug_path}")
                    raise
                warnings = check_draft(profile, row)
                item = {
                    "case_id": f"SAD_{int(row['source_index']):04d}",
                    "synthetic": True,
                    "source_dataset": "PatternReframe, adapted for SAD-related contexts",
                    "source_index": row["source_index"],
                    "source_pattern": row["pattern"],
                    "source_thought": row["thought"],
                    "source_persona": row["persona"],
                    "input_seasonal_context": row.get("seasonal_context", ""),
                    "input_symptoms": row.get("sad_related_symptoms", ""),
                    "model_id": model,
                    "intake_form": profile,
                    "automatic_warnings": warnings,
                    "human_review_status": "pending",
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{idx}/{len(rows)}] source_index={row['source_index']} OK; warnings={warnings}")
            except Exception as exc:
                print(f"[{idx}/{len(rows)}] source_index={row.get('source_index')} ERROR: {exc}")
                print("Fix the error and rerun with a NEW --output filename; completed rows stay saved.")
                break
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
