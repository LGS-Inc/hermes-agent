"""Dumbledore local routing engine.

Home mode (default resting state): deterministic per-task selection among the
local Ollama operators. Model picker (`/model ...`) pins a cloud model — that
path is owned by Hermes core and is untouched here; this module only decides
what to do when NO pin is active (home mode), plus the control-word, refusal,
and uncut lanes.

Hard invariants (asserted + tested):
  * No code path in the router-selectable set ever resolves to an abliterated
    model. `assert_not_abliterated()` guards the home/pin resolution path.
  * The uncut lane is advisory-only: a direct Ollama /api/chat call with NO
    tools array, tagged "[UNCUT]", never entering a tool loop.

This module is dependency-light on purpose so it can be unit-tested standalone
(no Hermes imports at module load).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Model / provider constants
# ---------------------------------------------------------------------------

# Home-mode operators (router auto-selectable set = default + image).
# Default operator serves an honest 131072 (probe: 8.6GB, 100% GPU). Because it
# now spans the whole non-code/non-image space, there is NO separate long-context
# auto-lane — a 40-60k paste stays on the default operator (Rule 3 removed).
HOME_DEFAULT_MODEL = "qwen3.5:9b-131k-fleet"  # default operator: honest 131072 serving
                                              # + fleet-qualified sampling (temp 0,
                                              # presence_penalty 0). The plain -131k tag
                                              # inherited the base's hostile temp 1 /
                                              # presence 1.5 -> degenerate open-ended
                                              # output (2026-07-31 defect).
IMAGE_MODEL = "gemma4:12b"                  # image attachments only (~32k window)
CODER_MODEL = "qwen2.5-coder:14b"           # dispatched via aider-wrapper, NOT here

# Name-callable via /model but NEVER auto-selected by the classifier:
QWEN35_131K_MODEL = "qwen3.5:9b-131k"       # superseded default (hostile sampling)
QWEN35_64K_MODEL = "qwen3.5:9b-64k"         # kept resident-friendly alternate
LONG_MODEL = "ministral-3:14b"              # kept name-callable (32768 serving)

# Image lane window (gemma4 serves the 32768 global). A prompt above this plus an
# image is near/over the window → still gemma4, but warn the Chairman to trim.
IMAGE_OVERFLOW_TOKENS = 28_000

# custom_providers slugs (config.yaml) resolved via
# _resolve_runtime_agent_kwargs_for_provider("custom:<slug>"). All four are
# mapped so a name-called model still resolves; the classifier only emits the
# default (131k) and image (gemma4) entries.
PROVIDER_BY_MODEL = {
    HOME_DEFAULT_MODEL: "custom:home-9b-131k-fast,-full-gpu",
    IMAGE_MODEL: "custom:vision-12b-fast,-full-gpu",
    QWEN35_131K_MODEL: "custom:avoid-9b-131k,-bad-sampling",
    QWEN35_64K_MODEL: "custom:avoid-9b-64k,-bad-sampling",
    LONG_MODEL: "custom:long-14b,-not-installed",
}

# Abliterated models — uncut lane ONLY. Never routable by home/pin logic.
UNCUT_MODEL = "qwen3-abliterated-hermes:8b"
UNCUT_MODEL_ALT = "huihui_ai/qwen3-abliterated:8b"
ABLITERATED_MODELS = frozenset({UNCUT_MODEL, UNCUT_MODEL_ALT})

OLLAMA_BASE = os.environ.get("DUMBLEDORE_OLLAMA_URL", "http://127.0.0.1:11434")

# Paths
HERMES_HOME = os.environ.get("HERMES_HOME", "/home/qws/.hermes")
TELEMETRY_PATH = os.path.join(HERMES_HOME, "logs", "router.jsonl")
MODE_PATH = os.path.join(HERMES_HOME, "router_mode.json")

# ---------------------------------------------------------------------------
# Control-word parsing
# ---------------------------------------------------------------------------

_PUNCT_STRIP = re.compile(r"^[\s\W]+|[\s\W]+$")


def _norm_word(s: str) -> str:
    """Lowercase + strip surrounding punctuation/space for exact control match."""
    return _PUNCT_STRIP.sub("", (s or "").strip().lower())


@dataclass
class ControlResult:
    kind: str            # "home" | "uncut" | "uncut_alt" | "none"
    payload: str = ""    # remaining prompt for uncut lanes


def parse_control(prompt: str) -> ControlResult:
    """Detect control words. Exact-match, case-insensitive, punctuation-stripped.

    "/model ..." is intentionally NOT handled here — it stays with the existing
    Hermes slash handler (the picker).
    """
    raw = (prompt or "").strip()
    first = _norm_word(raw.split(maxsplit=1)[0]) if raw else ""

    # bare "home"
    if _norm_word(raw) == "home":
        return ControlResult("home")

    # "/uncut ..." or bare "uncut ..."
    if first in ("uncut",):
        rest = raw.split(maxsplit=1)
        rest_txt = rest[1] if len(rest) > 1 else ""
        # "uncut alt <prompt>"
        alt = rest_txt.split(maxsplit=1)
        if alt and _norm_word(alt[0]) == "alt":
            return ControlResult("uncut_alt", alt[1] if len(alt) > 1 else "")
        return ControlResult("uncut", rest_txt)
    if first == "/uncut":
        rest = raw.split(maxsplit=1)
        rest_txt = rest[1] if len(rest) > 1 else ""
        alt = rest_txt.split(maxsplit=1)
        if alt and _norm_word(alt[0]) == "alt":
            return ControlResult("uncut_alt", alt[1] if len(alt) > 1 else "")
        return ControlResult("uncut", rest_txt)

    return ControlResult("none")


# ---------------------------------------------------------------------------
# Classifier (home mode)
# ---------------------------------------------------------------------------

# Conservative code-work intent signal (rule c). Chat stays on the default
# operator; the actual code job is dispatched through the existing
# aider-wrapper pipeline (unchanged) — the router only labels the turn.
_CODE_INTENT = re.compile(
    r"\b(implement|refactor|debug|write tests?|unit tests?|fix the bug|"
    r"build the|compile|multi-?file|patch the|add a function|add a method|"
    r"stack ?trace|traceback|npm |cargo |pytest|git diff)\b",
    re.IGNORECASE,
)


def estimate_tokens(prompt: str) -> int:
    """Rough token estimate (chars/4), matching Hermes' rough estimator scale."""
    return len(prompt or "") // 4


@dataclass
class RoutingDecision:
    rule: str                     # image|image_overflow_warn|code|default|control|pin|refusal_suggest|uncut
    model: str
    provider: Optional[str] = None
    note: str = ""
    est_prompt_tokens: int = 0
    notice: str = ""              # user-facing one-liner to append (e.g. image overflow)


def assert_not_abliterated(model: str) -> None:
    """Guardrail: the home/pin path must never resolve to an abliterated model."""
    if model in ABLITERATED_MODELS:
        raise AssertionError(
            f"Refusing to auto-route to abliterated model {model!r}: "
            "abliterated models are uncut-lane only."
        )


def classify_home(prompt: str, has_image: bool, est_tokens: Optional[int] = None) -> RoutingDecision:
    """Deterministic home-mode routing. Order: image > code > default.

    Rule 3 (a separate long-context lane) is REMOVED: the default operator now
    serves 131072, so a long paste stays on it. qwen3.5 NEVER handles images —
    images are gemma4's lane, period. When an image arrives with a very large
    text body (> IMAGE_OVERFLOW_TOKENS) it still goes to gemma4, but the reply
    carries a one-line trim/split notice (rule image_overflow_warn).
    """
    if est_tokens is None:
        est_tokens = estimate_tokens(prompt)

    notice = ""
    if has_image:
        model = IMAGE_MODEL
        if est_tokens > IMAGE_OVERFLOW_TOKENS:
            rule = "image_overflow_warn"
            notice = (
                f"⚠️ This attachment plus ~{est_tokens:,} tokens of text exceeds "
                "the vision lane's ~32k window — trim or split it for a reliable read."
            )
        else:
            rule = "image"
    elif _CODE_INTENT.search(prompt or ""):
        # Chat stays on the default operator; code work goes to aider-wrapper.
        model = HOME_DEFAULT_MODEL
        rule = "code"
    else:
        model = HOME_DEFAULT_MODEL
        rule = "default"

    assert_not_abliterated(model)
    return RoutingDecision(
        rule=rule,
        model=model,
        provider=PROVIDER_BY_MODEL.get(model),
        est_prompt_tokens=est_tokens,
        notice=notice,
    )


# ---------------------------------------------------------------------------
# Code-lane hard dispatch (Chairman-authorized, code-lane lineage attempt 1)
# ---------------------------------------------------------------------------

CODER_TIMEOUT = 900.0  # matches the proven MCP path; slow-local is latency, not death

# BUILD/EDIT GUARD — the line between "production" (hard-dispatch to the coder)
# and "discussion" (stays on the home default).
#
# Precedence: DISCUSSION WINS. A false "discussion" verdict is the harmless
# status quo (home default answers, as today); a false "production" verdict
# sends conversation into a code pipeline — the exact failure mode the guard
# exists to prevent. So discussion markers veto production markers.
#
# Production = imperative code-making intent: an instruction to produce,
# refactor, debug, or modify code.
_CODE_PRODUCTION = re.compile(
    r"\b(refactor|rewrite|implement|debug( this|ging)?|fix (this|the|my) (bug|code|function|script|error)|"
    r"write (a |an |the |some )?(function|class|script|module|test|unit test|code|program)|"
    r"create (a |an |the )?(function|class|script|module|test|program)|"
    r"generate (a |an |the |some )?(function|class|script|module|test|code|program)|"
    r"build (a |an |the )?(function|class|script|module|program)|"
    r"add (a |an )?(function|method|error handling|unit tests?|tests?)|"
    r"convert (this|the|my) (code|script|function)|"
    r"optimi[sz]e (this|the|my) (code|script|function|query)|"
    r"patch (this|the|my)|write tests?|unit tests?)\b",
    re.IGNORECASE,
)

# Discussion = question/explanation forms about code, not instructions to make it.
_CODE_DISCUSSION = re.compile(
    r"(^|\b)(what does\b|what is\b|what's\b|what are\b|explain\b|should i\b|"
    r"which (is|one|db|database|framework|language)\b|difference between\b|"
    r"\bvs\.?\b|versus\b|why (is|does|do|did|would)\b|how does\b|how do(es)? .{0,40}work\b|"
    r"recommend\b|is it (worth|better|ok|okay)\b|pros and cons\b|compare\b|"
    r"can you explain\b|help me understand\b|\bmean\b\??)",
    re.IGNORECASE,
)


def is_code_production(prompt: str) -> bool:
    """True only for produce/refactor/debug/modify asks. Discussion vetoes."""
    text = prompt or ""
    if not _CODE_PRODUCTION.search(text):
        return False
    if _CODE_DISCUSSION.search(text):
        return False
    return True


def run_coder_dispatch(prompt: str, timeout: float = CODER_TIMEOUT) -> str:
    """Deterministic dispatch to the proven qwen-coder pipeline.

    Byte-for-byte the same call the green ask_qwen_coder MCP tool makes
    (same model, system prompt, temperature, num_ctx, endpoint) — no chat-model
    discretion anywhere in the loop. Raises on error/empty so the caller can
    fall back WITH notice; never fails silently.
    """
    import httpx

    system_prompt = (
        "You are the local coding specialist served through Ollama.\n\n"
        "Your configured backend model identifier is exactly: qwen2.5-coder:14b\n"
        "When asked which model you are, report that exact identifier.\n"
        "Do not claim to be GPT-4, Claude, Gemini, or any other model.\n\n"
        "Provide technically precise and directly usable work.\n"
        "Do not invent files, commands, APIs, execution results, or successful tests.\n"
        "State clearly when more evidence is required.\n"
        "Prefer complete commands, patches, or replacement files.\n"
        "Separate findings, proposed changes, verification steps, and risks."
    )
    payload = {
        "model": CODER_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"TASK:\n{(prompt or '').strip()}"},
        ],
        "options": {"temperature": 0.2, "num_ctx": 32768},
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = (data.get("message") or {}).get("content", "").strip()
    if not content:
        raise RuntimeError(f"coder returned no content: {str(data)[:200]}")
    return content


# ---------------------------------------------------------------------------
# Image-GENERATION lane (Phase 2 — flag: DUMBLEDORE_IMAGE_LANE=1)
# ---------------------------------------------------------------------------
# Ordering contract (enforced at the gateway hook, restated here):
#   image ATTACHED                     -> Rule 2 (gemma4 reading), UNCHANGED
#   image ATTACHED + generative verb   -> img2img refusal message (out of scope)
#   no attachment + generative intent  -> HARD DISPATCH to ComfyUI/FLUX.2 Klein
#   otherwise                          -> Rules 1/3/4, UNCHANGED

COMFY_URL = "http://127.0.0.1:8188"
COMFY_LAUNCHER = "/home/qws/comfyui/run-comfy.sh"
COMFY_OUTPUT_DIR = "/home/qws/comfyui/output"
COMFY_LAUNCH_LOG = "/home/qws/comfyui/comfy-launch.log"
FLUX_CKPT = "flux-2-klein-4b-fp8.safetensors"
FLUX_TEXT_ENCODER = "qwen_3_4b.safetensors"
FLUX_VAE = "flux2-vae.safetensors"
IMAGE_GEN_MODEL = "flux-2-klein-4b-fp8"         # telemetry label
IMAGE_GEN_STEPS = 4
IMAGE_GEN_QUALITY_STEPS = 16
IMAGE_GEN_WIDTH = 1024
IMAGE_GEN_HEIGHT = 1024
IMAGE_GEN_QUALITY_WIDTH = 1280
IMAGE_GEN_QUALITY_HEIGHT = 1280
BFL_BRAND_MODEL = "flux-2-flex"
BFL_BRAND_PROVIDER = "black-forest-labs"
BFL_BRAND_WIDTH = 1024
BFL_BRAND_HEIGHT = 1024
BFL_BRAND_DIMENSIONS = {
    "x": (1200, 675),
    "fb": (1200, 630),
    "ig": (1080, 1080),
    "ig-portrait": (1080, 1350),
    "landscape": (1536, 1024),
}
IMAGE_GEN_CFG = 1.0
IMAGE_GEN_SAMPLER = "euler"
COMFY_IDLE_TIMEOUT = 600.0                       # warm-until-chat backstop (s)

IMG2IMG_REFUSAL = (
    "🖼️ Image-to-image isn't supported yet — I can't edit or transform an "
    "attached picture. Describe what you want generated fresh and I'll create it."
)

# Strip only a leading command wrapper.  An image noun is part of that wrapper
# only when it immediately follows the verb ("create an image of ...").  If a
# modifier comes first ("create a futuristic image ..."), every word from that
# modifier onward belongs to the Chairman's prompt and survives intact.
_IMG_COMMAND_WRAPPER = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:make|create|generate|draw|render|design|produce|paint)\b"
    r"(?:\s+(?:me|us))?"
    r"(?:\s+(?:(?:an?|the)\s+)?"
    r"(?:image|images|picture|pictures|photo|photos|logo|logos|render|rendering|"
    r"artwork|illustration|illustrations)\b"
    r"(?:\s+(?:of|showing|depicting|with|for))?)?"
    r"\s*[:,-]?\s*",
    re.IGNORECASE,
)

# Question-form veto — mirrors the code lane's discussion-wins precedence.
# "can you make images?" / "what image models do we have?" are questions.
_IMG_QUESTION_VETO = re.compile(
    r"(^|\b)(what|which|how|why|when|who|do (we|you)|does|did|was|were|can|could|would|is there|are there|"
    r"have we|has it|should i)\b",
    re.IGNORECASE,
)

_IMG_VERBLESS_ORDER = re.compile(
    r"^(?:an?\s+|the\s+)?(?:[\w'-]+\s+){0,8}"
    r"(?:image|picture|photo(?:graph)?|portrait|illustration|artwork|render(?:ing)?|poster)\b",
    re.IGNORECASE,
)

def extract_image_subject(prompt: str) -> str:
    """Remove only a leading image-generation command wrapper."""
    text = (prompt or "").strip()
    m = _IMG_COMMAND_WRAPPER.match(text)
    if m:
        return text[m.end():].strip().rstrip("?!. ")
    if not _IMG_QUESTION_VETO.search(text) and _IMG_VERBLESS_ORDER.match(text):
        return text.rstrip("?!. ")
    return ""


def is_image_generation(prompt: str) -> bool:
    """True only for an ORDER to generate an image with a non-empty subject.

    Question forms veto (a false 'no' harmlessly falls through to the home
    default; a false 'yes' would fire a 60-90s generation on a question).
    A bare 'can you make images?' also fails because no subject survives
    the strip.
    """
    text = (prompt or "").strip()
    if _IMG_QUESTION_VETO.search(text):
        return False
    return bool(extract_image_subject(text))


def parse_image_controls(prompt: str) -> dict:
    """Consume leading /quality, /literal, and /brand flags in any order.

    Only the two registered image controls are consumed. An unrelated slash
    token remains in ``prompt`` so the gateway's normal unknown-command guard
    retains ownership of it.
    """
    tokens = (prompt or "").strip().split()
    quality = False
    literal = False
    brand = False
    consumed = 0
    for token in tokens:
        normalized = token.lower()
        if normalized == "/quality":
            quality = True
        elif normalized == "/literal":
            literal = True
        elif normalized == "/brand":
            brand = True
        else:
            break
        consumed += 1
    return {
        "quality": quality,
        "literal": literal,
        "brand": brand,
        "prompt": " ".join(tokens[consumed:]),
        "consumed": consumed,
    }


_COMPANY_MARKER = re.compile(
    r"\b(?:inc\.?|incorporated|llc|ltd\.?|limited|corp\.?|corporation|company|"
    r"studios?|holdings?)\b",
    re.IGNORECASE,
)

_COMPANY_NAME = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&.'-]*\s+){1,8}"
    r"(?:Inc\.?|Incorporated|LLC|Ltd\.?|Limited|Corp\.?|Corporation|Company|Studios?|Holdings?)\b"
)
_LETTERING_TRIGGERS = re.compile(
    r"\b(?:ancient|runes?|inscriptions?|carvings?|glyphs?|engraved|poster|ui|signage|labels?)\b",
    re.IGNORECASE,
)
_TYPOGRAPHY_DIRECTION = re.compile(
    r"\b(?:font|typeface|typography|lettering|letters?|letterforms?|serif|sans[- ]?serif|"
    r"script|cursive|blackletter|stencil|engraved|etched|embossed|debossed|"
    r"dimensional|built[- ]?up|beveled|brushed[- ]?(?:gold|metal|steel|brass)|"
    r"matte|satin|metallic|gold|silver|bronze|bold|semibold|medium|light|thin|"
    r"heavy|uppercase|lowercase|title[- ]?case|italic|condensed|expanded)\b",
    re.IGNORECASE,
)
_PREMIUM_TYPOGRAPHY_DEFAULT = (
    "professionally fabricated dimensional lettering, refined brushed-metal "
    "or crisp matte finish, subtle physical depth and natural shadow, premium "
    "corporate signage quality"
)


def _extract_company_name(original: str) -> Optional[str]:
    matches = list(_COMPANY_NAME.finditer(original or ""))
    if matches:
        return matches[-1].group(0)
    text = (original or "").strip()
    return text if _COMPANY_MARKER.search(text) else None


def extract_paid_brand_text(original: str) -> str:
    """Resolve one explicit quoted string, then fall back to company detection."""
    raw = original or ""
    if raw.count('"') % 2:
        raise ValueError(
            '/brand requires one complete double-quoted string, for example "Exact Text"'
        )
    quoted = re.findall(r'"([^"\r\n]*)"', raw)
    if len(quoted) > 1:
        raise ValueError(
            "/brand accepts exactly one contiguous quoted string; multiple quoted strings were supplied"
        )
    if quoted:
        if not quoted[0]:
            raise ValueError("/brand quoted text cannot be empty")
        return quoted[0]
    company = _extract_company_name(raw)
    if company:
        return company
    raise ValueError(
        '/brand requires one exact double-quoted string or a detectable company name'
    )


def _scene_without_approved_text(original: str, approved_text: str) -> str:
    """Remove one lettering occurrence while preserving every other byte."""
    quoted = f'"{approved_text}"'
    needle = quoted if quoted in original else approved_text
    start = original.find(needle)
    if start < 0:
        return original
    return original[:start] + original[start + len(needle):]


def parse_brand_dimensions(subject: str) -> dict:
    """Consume one optional channel-size argument after an explicit /brand."""
    raw = (subject or "").strip()
    first, separator, remainder = raw.partition(" ")
    preset = first.lower().rstrip(":")
    dimensions = BFL_BRAND_DIMENSIONS.get(preset)
    if dimensions and separator and remainder.strip():
        return {
            "prompt": remainder.strip(), "preset": preset,
            "width": dimensions[0], "height": dimensions[1],
        }
    return {
        "prompt": raw, "preset": "default",
        "width": BFL_BRAND_WIDTH, "height": BFL_BRAND_HEIGHT,
    }


def _commercial_scene_scaffold(scene: str) -> str:
    """Add only missing commercial-photo slots; never rewrite supplied scene."""
    lower = scene.lower()
    additions = []
    if not re.search(r"\b(?:close-up|wide|medium|full|foreground|background|centered|composition|framing|focal)\b", lower):
        additions.append("clear centered composition with an uncluttered focal hierarchy")
    if not re.search(r"\b(?:light|lit|lighting|sunlight|daylight|golden hour|neon|shadow)\b", lower):
        additions.append("soft directional studio daylight")
    if not re.search(r"\b(?:camera|lens|mm|photograph|render|illustration|cinematic|isometric|3d)\b", lower):
        additions.append("commercial photography captured with a 50mm lens")
    additions.append("premium polished client-facing finish")
    return ", ".join(part for part in [scene.strip(), *additions] if part)


def _extract_typography_direction(scene: str) -> Optional[str]:
    """Return authored comma/semicolon clauses that contain typography steering."""
    clauses = re.split(r"(?<=[,;])", scene)
    directed = [clause.strip(" \t,;") for clause in clauses if _TYPOGRAPHY_DIRECTION.search(clause)]
    directed = [clause for clause in directed if clause]
    return ", ".join(directed) if directed else None


def build_brand_prompt(original: str, approved_text: str) -> str:
    """Object-first typography prompt with one exact brand occurrence."""
    scene = _scene_without_approved_text(original, approved_text)
    typography_direction = _extract_typography_direction(scene)
    typography = (
        "the Chairman's exact typography direction: " + typography_direction
        if typography_direction
        else _PREMIUM_TYPOGRAPHY_DEFAULT
    )
    if re.search(r"#[0-9a-fA-F]{6}\b", original):
        color_rule = " Preserve every supplied hex color on its named object exactly."
    else:
        color_rule = ""
    return (
        "A large front-facing architectural wall sign displaying the exact "
        f"single contiguous string \"{approved_text}\" rendered with {typography}, "
        "centered, highly legible, and occupying a "
        "substantial fraction of the frame, and no other text, letters, "
        "numbers, or symbols anywhere. Do not invent company names, contact "
        "details, statistics, logos, claims, or filler text."
        f"{color_rule}; scene: {_commercial_scene_scaffold(scene)}"
    )


def build_paid_brand_prompt(original: str) -> str:
    approved_text = extract_paid_brand_text(original)
    prompt = build_brand_prompt(original, approved_text)
    remainder = _scene_without_approved_text(original, approved_text)
    if (
        remainder not in prompt
        or prompt.count(approved_text) != 1
        or f'"{approved_text}"' not in prompt
    ):
        raise ValueError("brand prompt failed the exact-string/remainder invariant")
    return prompt


def check_paid_brand_spelling(
    image_path: str, approved_text: str, timeout: float = 45.0,
) -> dict:
    """Advisory local Gemma OCR. Failure or uncertainty never blocks delivery."""
    import base64
    import httpx

    started = time.time()
    try:
        with open(image_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": IMAGE_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": (
                            "Read the main brand text in this image. Return only the "
                            "characters you can see, with exact capitalization and spacing. "
                            "If uncertain, return exactly INCONCLUSIVE."
                        ),
                        "images": [encoded],
                    }],
                    "stream": False, "think": False, "keep_alive": "0",
                    "options": {"temperature": 0, "num_predict": 80},
                },
            )
            response.raise_for_status()
            observed = ((response.json().get("message") or {}).get("content") or "").strip()
        if not observed or observed.upper() == "INCONCLUSIVE":
            status = "inconclusive"
        else:
            status = "match" if observed == approved_text else "mismatch"
        return {"status": status, "observed_text": observed, "seconds": time.time() - started}
    except Exception as exc:
        return {
            "status": "unavailable", "observed_text": "",
            "reason": type(exc).__name__, "seconds": time.time() - started,
        }


def run_bfl_brand_generation(
    prompt: str, timeout: float = 300.0,
    *, width: int = BFL_BRAND_WIDTH, height: int = BFL_BRAND_HEIGHT,
) -> dict:
    """Run the explicit paid BFL lane. Never falls back to a local model."""
    import httpx
    from PIL import Image

    key = os.environ.get("BFL_API_KEY", "").strip()
    if not key:
        try:
            from dotenv import dotenv_values
            key = str(
                dotenv_values(os.path.join(HERMES_HOME, ".env")).get(
                    "BFL_API_KEY", ""
                ) or ""
            ).strip()
        except Exception:
            key = ""
    if not key:
        raise RuntimeError("BFL_API_KEY is not available to the gateway")
    headers = {
        "accept": "application/json", "x-key": key,
        "Content-Type": "application/json",
    }
    started = time.time()
    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"https://api.bfl.ai/v1/{BFL_BRAND_MODEL}",
            headers=headers,
            json={
                "prompt": prompt,
                "width": width,
                "height": height,
                "output_format": "png",
            },
        )
        response.raise_for_status()
        submission = response.json()
        polling_url = submission.get("polling_url")
        if not polling_url:
            raise RuntimeError(f"BFL submission returned no polling URL: {submission}")
        raw_cost = submission.get("cost")
        if not isinstance(raw_cost, (int, float)):
            raise RuntimeError("BFL submission omitted its actual charged cost")
        while time.time() - started < timeout:
            time.sleep(1)
            polled = client.get(polling_url, headers=headers)
            polled.raise_for_status()
            result = polled.json()
            status = result.get("status")
            if status == "Ready":
                url = (result.get("result") or {}).get("sample")
                if not url:
                    raise RuntimeError("BFL returned Ready without an image URL")
                image_response = client.get(url)
                image_response.raise_for_status()
                break
            if status in ("Error", "Failed", "Request Moderated", "Content Moderated"):
                raise RuntimeError(f"BFL generation failed with status {status}")
        else:
            raise TimeoutError(f"BFL generation timed out after {timeout:.0f}s")

    out_dir = os.path.join(HERMES_HOME, "cache", "images")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"bfl_{BFL_BRAND_MODEL}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    )
    with open(path, "wb") as fh:
        fh.write(image_response.content)
    with Image.open(path) as image:
        actual_width, actual_height = image.size
    return {
        "path": path,
        "provider": BFL_BRAND_PROVIDER,
        "model": BFL_BRAND_MODEL,
        "width": actual_width,
        "height": actual_height,
        "cost_credits": float(raw_cost),
        "cost_usd": float(raw_cost) * 0.01,
        "request_id": submission.get("id"),
        "seconds": time.time() - started,
    }


def enrich_image_prompt(subject: str, timeout: float = 120.0) -> dict:
    """Expand a terse prompt with the resident home model before ComfyUI starts.

    The complete original subject must occur verbatim in the result.  This is a
    stronger invariant than token/POS checks: every noun, modifier, style word,
    and named entity supplied by the Chairman necessarily survives.  Any model
    failure or invariant violation falls back to the original unchanged.
    """
    import httpx

    original = (subject or "").strip()
    if not original:
        return {"prompt": original, "seconds": 0.0, "enriched": False}
    company = _extract_company_name(original)
    if company:
        candidate = build_brand_prompt(original, company)
        remainder = _scene_without_approved_text(original, company)
        if (
            remainder in candidate
            and candidate.count(company) == 1
            and f'"{company}"' in candidate
        ):
            return {
                "prompt": candidate,
                "seconds": 0.0,
                "enriched": True,
                "reason": "brand_structure",
            }
        return {"prompt": original, "seconds": 0.0, "enriched": False,
                "reason": "brand_exact_string_remainder_guard"}
    original_triggers = {
        m.group(0).lower() for m in _LETTERING_TRIGGERS.finditer(original)
    }
    system = (
        "You expand image-generation prompts. Return only one finished prompt. "
        "Begin with ORIGINAL copied as one exact, case-sensitive, verbatim "
        "substring, then add concrete visual detail about lighting, camera "
        "framing, materials, mood, and detail level. Add detail; never replace, "
        "paraphrase, delete, reorder, or correct any word in ORIGINAL. Do not "
        "invent visible text, captions, watermarks, signage, or logos. When "
        "ORIGINAL did not request text, do not introduce ancient, runes, "
        "inscriptions, carvings, glyphs, engraved, poster, UI, signage, or "
        "label; describe visible surfaces positively as blank, smooth, and "
        "unmarked."
    )
    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout) as c:
            response = c.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": HOME_DEFAULT_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"ORIGINAL:\n{original}"},
                    ],
                    "stream": False,
                    "think": False,
                    "keep_alive": "10m",
                    "options": {"temperature": 0, "num_predict": 180},
                },
            )
            response.raise_for_status()
            candidate = ((response.json().get("message") or {}).get("content") or "").strip()
        if original not in candidate:
            return {
                "prompt": original,
                "seconds": time.time() - t0,
                "enriched": False,
                "reason": "verbatim_guard",
            }
        added_triggers = {
            m.group(0).lower() for m in _LETTERING_TRIGGERS.finditer(candidate)
        } - original_triggers
        if added_triggers:
            return {
                "prompt": original + ", blank smooth unmarked visible surfaces",
                "seconds": time.time() - t0,
                "enriched": True,
                "reason": "lettering_trigger_guard",
            }
        return {
            "prompt": candidate,
            "seconds": time.time() - t0,
            "enriched": candidate != original,
            "reason": "ok",
        }
    except Exception as exc:
        return {
            "prompt": original,
            "seconds": time.time() - t0,
            "enriched": False,
            "reason": type(exc).__name__,
        }


# ---- ComfyUI lifecycle -----------------------------------------------------

def comfy_is_up(timeout: float = 2.0) -> bool:
    import httpx
    try:
        with httpx.Client(timeout=timeout) as c:
            return c.get(f"{COMFY_URL}/system_stats").status_code == 200
    except Exception:
        return False


def _unload_ollama_models() -> None:
    """Interlock: keep_alive:0 for every resident model. Service NEVER touched."""
    import httpx
    try:
        with httpx.Client(timeout=10) as c:
            ps = c.get("http://127.0.0.1:11434/api/ps").json()
            for m in ps.get("models", []):
                c.post("http://127.0.0.1:11434/api/generate",
                       json={"model": m["name"], "keep_alive": 0})
    except Exception:
        pass


COMFY_UNIT = "comfyui.service"          # on-demand systemd --user unit (static, no [Install])
COMFY_LAST_START_MECHANISM = "none"     # "systemd" | "launcher" | "already_up" (telemetry)


def _comfy_systemctl(*args: str, timeout: float = 30.0):
    import subprocess
    return subprocess.run(
        ["systemctl", "--user", *args], capture_output=True, text=True, timeout=timeout,
    )


def comfy_unit_available() -> bool:
    """True when the on-demand comfyui.service user unit is loaded."""
    try:
        out = _comfy_systemctl("show", COMFY_UNIT, "-p", "LoadState", timeout=10)
        return "LoadState=loaded" in (out.stdout or "")
    except Exception:
        return False


def comfy_unit_state() -> str:
    try:
        out = _comfy_systemctl("is-active", COMFY_UNIT, timeout=10)
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def start_comfy(ready_timeout: float = 180.0) -> float:
    """Start ComfyUI on demand. Returns startup seconds.

    Prefers the bounded systemd user unit (``systemctl --user start comfyui``,
    which runs the same launcher with its Ollama unload interlock); falls back
    to the Phase-1 detached launcher when the unit is unavailable. We ALSO run
    the unload interlock here so it happens even when ComfyUI is already warm.
    """
    import subprocess

    global COMFY_LAST_START_MECHANISM
    _unload_ollama_models()
    if comfy_is_up():
        COMFY_LAST_START_MECHANISM = "already_up"
        return 0.0
    t0 = time.time()
    mechanism = "launcher"
    if comfy_unit_available():
        try:
            if _comfy_systemctl("start", COMFY_UNIT, timeout=30).returncode == 0:
                mechanism = "systemd"
        except Exception:
            mechanism = "launcher"
    if mechanism == "launcher":
        with open(COMFY_LAUNCH_LOG, "a") as log:
            subprocess.Popen(
                [COMFY_LAUNCHER],
                stdout=log, stderr=log,
                start_new_session=True,   # survive the caller; killed by shutdown_comfy
            )
    COMFY_LAST_START_MECHANISM = mechanism
    while time.time() - t0 < ready_timeout:
        if comfy_is_up():
            return time.time() - t0
        if (
            mechanism == "systemd"
            and time.time() - t0 > 10
            and comfy_unit_state() in ("failed", "inactive")
        ):
            break
        time.sleep(2)
    if mechanism == "systemd":
        try:
            _comfy_systemctl("stop", COMFY_UNIT, timeout=40)
        except Exception:
            pass
    raise RuntimeError(f"ComfyUI did not become ready within {ready_timeout:.0f}s ({mechanism})")


def shutdown_comfy(wait: float = 25.0) -> bool:
    """Kill ComfyUI and confirm the port is released. MANDATORY before a chat
    model load — 15.3GB ComfyUI + 8.6GB chat model on a 16GB card spills to CPU.

    Phase-1 gotcha: the exec'd process cmdline is `./venv/bin/python main.py
    --listen …` (relative path), so match on `main.py --listen`, never on the
    comfyui/venv path.
    """
    import subprocess, signal as _signal

    # Unit-managed instance: stop it through systemd first (bounded by the
    # unit's TimeoutStopSec); the process-path below handles launcher runs.
    try:
        if comfy_unit_state() in ("active", "activating", "deactivating"):
            _comfy_systemctl("stop", COMFY_UNIT, timeout=wait + 15)
    except Exception:
        pass
    if not comfy_is_up():
        return True
    try:
        out = subprocess.run(
            ["pgrep", "-f", "main.py --listen 127.0.0.1 --port 8188"],
            capture_output=True, text=True,
        )
        pids = [int(p) for p in out.stdout.split()]
        for pid in pids:
            os.kill(pid, _signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < wait:
            if not comfy_is_up():
                return True
            time.sleep(1)
        for pid in pids:                       # escalate
            try:
                os.kill(pid, _signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(2)
        return not comfy_is_up()
    except Exception:
        return not comfy_is_up()


def run_image_generation(
    subject: str, timeout: float = 600.0, *, steps: int = IMAGE_GEN_STEPS,
    width: int = 1024, height: int = 1024, seed: Optional[int] = None,
) -> dict:
    """Generate one 1024x1024 FLUX.2 Klein 4B FP8 image. Returns
    {path, seconds, startup_seconds, cold}. Raises on any failure so the
    caller reports it — never fails silently.
    """
    import httpx

    was_up = comfy_is_up()
    startup = 0.0 if was_up else start_comfy()
    if was_up:
        _unload_ollama_models()   # interlock even on the warm path

    workflow = {
        "prompt": {
            "1": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": FLUX_CKPT,
                             "weight_dtype": "default"}},
            "8": {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": FLUX_TEXT_ENCODER,
                             "type": "flux2"}},
            "9": {"class_type": "VAELoader",
                  "inputs": {"vae_name": FLUX_VAE}},
            "2": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": subject, "clip": ["8", 0]}},
            "3": {"class_type": "ConditioningZeroOut",
                  "inputs": {"conditioning": ["2", 0]}},
            "4": {"class_type": "EmptyFlux2LatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
            "10": {"class_type": "CFGGuider",
                   "inputs": {"model": ["1", 0], "positive": ["2", 0],
                              "negative": ["3", 0], "cfg": IMAGE_GEN_CFG}},
            "11": {"class_type": "RandomNoise",
                   "inputs": {"noise_seed": seed if seed is not None else int(time.time()) % (2**31)}},
            "12": {"class_type": "KSamplerSelect",
                   "inputs": {"sampler_name": IMAGE_GEN_SAMPLER}},
            "13": {"class_type": "Flux2Scheduler",
                   "inputs": {"steps": steps,
                              "width": width, "height": height}},
            "5": {"class_type": "SamplerCustomAdvanced",
                  "inputs": {"noise": ["11", 0], "guider": ["10", 0],
                             "sampler": ["12", 0], "sigmas": ["13", 0],
                             "latent_image": ["4", 0]}},
            "6": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["5", 0], "vae": ["9", 0]}},
            "7": {"class_type": "SaveImage",
                  "inputs": {"images": ["6", 0],
                             "filename_prefix": "dumbledore_imagelane"}},
        }
    }
    t0 = time.time()
    with httpx.Client(timeout=30) as c:
        resp = c.post(f"{COMFY_URL}/prompt", json=workflow)
        resp.raise_for_status()
        pid = resp.json()["prompt_id"]
    while time.time() - t0 < timeout:
        try:
            with httpx.Client(timeout=10) as c:
                hist = c.get(f"{COMFY_URL}/history/{pid}").json()
        except Exception:
            hist = {}
        entry = hist.get(pid)
        if entry:
            status = (entry.get("status") or {}).get("status_str", "")
            if status == "error":
                raise RuntimeError(f"ComfyUI reported error for prompt {pid}")
            outputs = entry.get("outputs") or {}
            for node in outputs.values():
                for img in node.get("images", []):
                    path = os.path.join(
                        COMFY_OUTPUT_DIR, img.get("subfolder", ""), img["filename"]
                    )
                    if os.path.exists(path) and os.path.getsize(path) > 0:
                        return {
                            "path": path,
                            "seconds": time.time() - t0,
                            "startup_seconds": startup,
                            "cold": not was_up,
                            "provider": "comfyui",
                            "model": IMAGE_GEN_MODEL,
                            "width": width,
                            "height": height,
                        }
        time.sleep(2)
    raise RuntimeError(f"image generation timed out after {timeout:.0f}s")


# ---------------------------------------------------------------------------
# Refusal signature (SUGGEST-ONLY — never auto-reroute)
# ---------------------------------------------------------------------------

# Conservative signatures. False positives are harmless by design (they only
# append a one-line suggestion). Keep this list conservative.
_REFUSAL_SIGNATURES = (
    "i can't help with that",
    "i cannot help with that",
    "i can't assist with that",
    "i cannot assist with that",
    "i'm not able to help with that",
    "i am not able to help with that",
    "i won't be able to help",
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i am unable to provide",
    "i can't provide that",
    "i cannot provide that",
    "as an ai, i cannot",
    "i must decline",
    "i cannot comply",
    "this request violates",
    "against my guidelines",
    "i'm not comfortable",
)

REFUSAL_SUGGESTION = "Blocked — say uncut to run this on the uncut model."


def is_refusal(reply: str) -> bool:
    low = (reply or "").lower()
    return any(sig in low for sig in _REFUSAL_SIGNATURES)


def maybe_append_refusal_suggestion(reply: str) -> tuple[str, bool]:
    """Return (possibly-suffixed reply, fired?). NEVER reroutes."""
    if is_refusal(reply):
        sep = "" if reply.endswith("\n") else "\n\n"
        return reply + sep + REFUSAL_SUGGESTION, True
    return reply, False


# ---------------------------------------------------------------------------
# Uncut lane (advisory-only, hard-isolated: NO tools, direct /api/chat)
# ---------------------------------------------------------------------------

UNCUT_PREFIX = "[UNCUT]"


def build_uncut_request(prompt: str, alt: bool = False) -> dict:
    """Build the exact /api/chat payload. Guarantees NO tools array present."""
    payload = {
        "model": UNCUT_MODEL_ALT if alt else UNCUT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    assert "tools" not in payload, "uncut lane must never carry a tools array"
    return payload


def run_uncut(prompt: str, alt: bool = False, timeout: float = 900.0) -> str:
    """Advisory-only call to the abliterated model. Returns a [UNCUT]-tagged reply."""
    import httpx

    payload = build_uncut_request(prompt, alt=alt)
    url = f"{OLLAMA_BASE}/api/chat"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = (data.get("message") or {}).get("content", "")
    return f"{UNCUT_PREFIX} {content}".rstrip()


# ---------------------------------------------------------------------------
# Mode persistence (outside session state → boot returns to home)
# ---------------------------------------------------------------------------

def load_mode() -> dict:
    try:
        with open(MODE_PATH, "r") as fh:
            return json.load(fh)
    except Exception:
        return {"mode": "home"}


def save_mode(mode: str, model: Optional[str] = None) -> None:
    data = {"mode": mode}
    if model:
        data["model"] = model
    tmp = MODE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, MODE_PATH)


def reset_to_home() -> None:
    """Called on gateway boot: force home mode regardless of any stale pin."""
    save_mode("home")


STATE_DB_PATH = os.path.join(HERMES_HOME, "state.db")
SESSION_SCOPE = os.path.join(HERMES_HOME, "sessions")
_BACKUP_SUFFIX = ".bak-dumbledore-router-20260731"


def clear_gateway_pins(scope: str = SESSION_SCOPE, state_db: str = STATE_DB_PATH) -> int:
    """Delete ONLY this gateway's persisted session pins so boot returns to home.

    Scoped by `scope` (the gateway's session-store path) — NEVER truncates the
    table. Makes a one-time backup of state.db (idempotent: won't overwrite an
    existing backup). Returns the number of rows deleted. Failures are swallowed
    so a boot is never blocked by this hygiene step.
    """
    import sqlite3
    import shutil

    try:
        if not os.path.exists(state_db):
            return 0
        backup = state_db + _BACKUP_SUFFIX
        if not os.path.exists(backup):
            shutil.copy2(state_db, backup)
        conn = sqlite3.connect(state_db)
        try:
            cur = conn.execute(
                "DELETE FROM gateway_routing WHERE scope = ?", (scope,)
            )
            deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return deleted
    except Exception:
        return 0


def boot_home(scope: str = SESSION_SCOPE, state_db: str = STATE_DB_PATH) -> int:
    """One-time gateway-startup hook: force home mode + clear stale session pins."""
    reset_to_home()
    return clear_gateway_pins(scope=scope, state_db=state_db)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def log_decision(
    *,
    mode: str,
    rule_fired: str,
    model: str,
    swap: bool = False,
    swap_seconds: Optional[float] = None,
    est_prompt_tokens: int = 0,
    outcome: str = "ok",
    ts: Optional[float] = None,
    original_prompt: Optional[str] = None,
    enriched_prompt: Optional[str] = None,
    enrichment_seconds: Optional[float] = None,
    preset: Optional[str] = None,
    provider: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    cost_estimate_usd: Optional[float] = None,
    cost_actual_usd: Optional[float] = None,
    cost_credits: Optional[float] = None,
    text_mode: Optional[str] = None,
    approved_text: Optional[str] = None,
    requested_width: Optional[int] = None,
    requested_height: Optional[int] = None,
    actual_width: Optional[int] = None,
    actual_height: Optional[int] = None,
    generation_count: Optional[int] = None,
    spelling_status: Optional[str] = None,
    observed_text: Optional[str] = None,
) -> None:
    """Append one JSONL routing-decision record."""
    rec = {
        "ts": ts if ts is not None else time.time(),
        "mode": mode,
        "rule_fired": rule_fired,
        "model": model,
        "swap": bool(swap),
        "swap_seconds": round(swap_seconds, 2) if swap_seconds else None,
        "est_prompt_tokens": est_prompt_tokens,
        "outcome": outcome,
    }
    if original_prompt is not None:
        rec["original_prompt"] = original_prompt
    if enriched_prompt is not None:
        rec["enriched_prompt"] = enriched_prompt
    if enrichment_seconds is not None:
        rec["enrichment_seconds"] = round(enrichment_seconds, 2)
    if preset is not None:
        rec["preset"] = preset
    if provider is not None:
        rec["provider"] = provider
    if width is not None:
        rec["width"] = int(width)
    if height is not None:
        rec["height"] = int(height)
    if cost_estimate_usd is not None:
        rec["cost_estimate_usd"] = round(float(cost_estimate_usd), 4)
    if cost_actual_usd is not None:
        rec["cost_actual_usd"] = round(float(cost_actual_usd), 4)
    if cost_credits is not None:
        rec["cost_credits"] = float(cost_credits)
    if text_mode is not None:
        rec["text_mode"] = text_mode
    if approved_text is not None:
        rec["approved_text"] = approved_text
    if requested_width is not None:
        rec["requested_width"] = int(requested_width)
    if requested_height is not None:
        rec["requested_height"] = int(requested_height)
    if actual_width is not None:
        rec["actual_width"] = int(actual_width)
    if actual_height is not None:
        rec["actual_height"] = int(actual_height)
    if generation_count is not None:
        rec["generation_count"] = int(generation_count)
    if spelling_status is not None:
        rec["spelling_status"] = spelling_status
    if observed_text is not None:
        rec["observed_text"] = observed_text
    os.makedirs(os.path.dirname(TELEMETRY_PATH), exist_ok=True)
    with open(TELEMETRY_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec
