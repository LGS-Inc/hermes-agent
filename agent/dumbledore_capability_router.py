"""Dumbledore resource-aware capability router (Chairman mission 2026-08-27).

Layers a route-class model on top of ``agent.dumbledore_router`` (which keeps
owning the control words, the uncut lane, the explicit /quality /literal
/brand image controls, and the FLUX/ComfyUI workflow).  This module owns:

  * the route classes (HOME_FAST, DEEP_LOCAL, CODE_FAST, CODE_HEAVY, VISION,
    IMAGE_GENERATION, CLOUD_SAFE, EXPLICIT_PIN, UNCUT);
  * a deterministic, testable classifier that emits a visible route-reason
    object (no LLM is ever consulted to pick a model);
  * the ``/route`` one-turn override grammar;
  * the bounded specialist context pack;
  * the shared local-accelerator lock and the Ollama/ComfyUI resource
    preflight (unload conflicts, load target, bounded timeouts, never kill);
  * keep-alive policy, CLOUD_SAFE fallback rules, and bounded telemetry.

Hard rules restated from the mission:
  * Governance/authority/privacy/safety controls come first — this module
    never widens tool authority, never touches QFB, never logs prompt text.
  * A persistent ``/model`` pin is never rewritten by a specialist dispatch.
  * Prompt length alone never promotes to DEEP_LOCAL.
  * If classification confidence is insufficient the answer is HOME_FAST.
  * No route failure may crash the gateway: every public entry point here
    either returns a result or raises a typed, catchable exception, and the
    gateway wraps each call in its own try/except.

Dependency-light on purpose (stdlib + ``agent.dumbledore_router`` only at
import; httpx/PIL imported lazily inside the functions that need them).
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agent import dumbledore_router as _dr

# ---------------------------------------------------------------------------
# Route classes
# ---------------------------------------------------------------------------

HOME_FAST = "HOME_FAST"
DEEP_LOCAL = "DEEP_LOCAL"
CODE_FAST = "CODE_FAST"
CODE_HEAVY = "CODE_HEAVY"
VISION = "VISION"
IMAGE_GENERATION = "IMAGE_GENERATION"
CLOUD_SAFE = "CLOUD_SAFE"
EXPLICIT_PIN = "EXPLICIT_PIN"
UNCUT = "UNCUT"

ROUTE_CLASSES = (
    HOME_FAST, DEEP_LOCAL, CODE_FAST, CODE_HEAVY, VISION,
    IMAGE_GENERATION, CLOUD_SAFE, EXPLICIT_PIN, UNCUT,
)

# ---------------------------------------------------------------------------
# Models / providers (verified live 2026-08-27 — see mission report)
# ---------------------------------------------------------------------------

HOME_FAST_MODEL = _dr.HOME_DEFAULT_MODEL                 # qwen3.5:9b-131k-fleet
HOME_FAST_PROVIDER = _dr.PROVIDER_BY_MODEL[HOME_FAST_MODEL]
DEEP_LOCAL_MODEL = "qwen3.6:35b-a3b-64k"
DEEP_LOCAL_PROVIDER = "custom:qwen36-35b-64k"
DEEP_LOCAL_ALIAS = "qwen36"
CODE_FAST_MODEL = _dr.CODER_MODEL                        # qwen2.5-coder:14b
CODE_HEAVY_MODEL = "qwen3-coder-next:latest"
VISION_MODEL = _dr.IMAGE_MODEL                           # gemma4:12b
VISION_PROVIDER = _dr.PROVIDER_BY_MODEL[VISION_MODEL]
CLOUD_SAFE_MODEL = "gpt-5.6-sol"
CLOUD_SAFE_PROVIDER = "openai-codex"
IMAGE_GEN_LABEL = _dr.IMAGE_GEN_MODEL                    # flux-2-klein-4b-fp8

# Ollama tags the router may dispatch to directly (specialist /api/chat).
# The abliterated models are deliberately absent: uncut lane only.
LOCAL_SPECIALIST_MODELS = frozenset({
    HOME_FAST_MODEL, DEEP_LOCAL_MODEL, CODE_FAST_MODEL, CODE_HEAVY_MODEL,
    VISION_MODEL,
})

# Heavy residents: never allowed to share the accelerator with another
# heavy resident.  The 9B home model is light and only displaced by heavy.
HEAVY_LOCAL_MODELS = frozenset({
    DEEP_LOCAL_MODEL, "qwen3.6:35b-a3b", CODE_FAST_MODEL, CODE_HEAVY_MODEL,
    VISION_MODEL,
})

# Served context windows the specialist pack must respect.
CONTEXT_WINDOW = {
    HOME_FAST_MODEL: 131072,
    DEEP_LOCAL_MODEL: 65536,
    CODE_FAST_MODEL: 32768,
    CODE_HEAVY_MODEL: 32768,   # num_ctx we request; the tag supports far more
    VISION_MODEL: 32768,
}

# Keep-alive policy (starting candidates per mission; validated in canaries).
KEEP_ALIVE = {
    HOME_FAST: "15m",
    CODE_FAST: "5m",
    "DEEP_LOCAL_AUTO": "5m",
    "DEEP_LOCAL_PIN": "12h",    # "until displaced or pin ends"
    "CODE_HEAVY_ONESHOT": 0,
    "CODE_HEAVY_SESSION": "5m", # only when explicitly in a heavy coding session
    VISION: 0,
}

# Bounded timeouts (seconds).
LOAD_TIMEOUT = {
    HOME_FAST_MODEL: 120.0,
    DEEP_LOCAL_MODEL: 180.0,
    CODE_FAST_MODEL: 180.0,
    CODE_HEAVY_MODEL: 420.0,   # 51 GB from disk, partially CPU-resident
    VISION_MODEL: 120.0,
}
INFERENCE_TIMEOUT = 900.0
LOCK_TIMEOUT_DEFAULT = 20.0        # never block the gateway longer than this
LOCK_TIMEOUT_LIGHT = 3.0           # HOME_FAST/light pins: brief wait, then proceed locally
LOCK_STALE_AFTER = 45 * 60.0       # holder metadata older than this is suspect
COMFY_IDLE_STOP_SECONDS = 120.0    # stop Flux after bounded idle
COMFY_READY_TIMEOUT = 180.0

# DEEP_LOCAL through the agent loop (tools available) is only safe when the
# live session fits the 64k window with the ~20k system prompt and room to
# answer.  Larger sessions go through the bounded specialist pack instead.
DEEP_LOCAL_AGENT_LOOP_HISTORY_BUDGET = 30_000
SPECIALIST_PACK_BUDGET = {
    DEEP_LOCAL_MODEL: 40_000,
    CODE_FAST_MODEL: 20_000,
    CODE_HEAVY_MODEL: 20_000,
}
SPECIALIST_RECENT_TURNS = 6

HERMES_HOME = _dr.HERMES_HOME
OLLAMA_BASE = _dr.OLLAMA_BASE
TELEMETRY_PATH = _dr.TELEMETRY_PATH
LOCK_PATH = os.path.join(HERMES_HOME, "state", "accelerator.lock")
RENDER_DIR = "/tmp/hermes-renders"
COMFY_UNIT = "comfyui.service"

ROUTE_SIGNATURE_ENV = "DUMBLEDORE_ROUTE_SIGNATURE"   # off by default

# Direct specialists do not receive Dumbledore's full agent system prompt.
# Keep this boundary short, deterministic, and identical for every specialist
# model so routing never changes authority or named-protocol semantics.
SPECIALIST_GOVERNANCE_BLOCK = (
    "SPECIALIST GOVERNANCE BOUNDARY (DETERMINISTIC):\n"
    "- You are advisory and task-scoped. Model selection grants no authority.\n"
    "- Follow only the exact bounded task supplied by parent Dumbledore; do not "
    "expand task scope.\n"
    "- You cannot activate, close, pause, supersede, or alter Protocol Alpha, "
    "Protocol OMEGA, FABLE Gate, or Independent Review. Any supplied protocol "
    "context is read-only.\n"
    "- You cannot authorize or perform write, deploy, send, delete, or "
    "authenticate actions.\n"
    "- Return evidence and proposed output only, never authority or approval.\n"
    "- Parent Dumbledore and its runtime authorization gates remain authoritative "
    "for scope, tools, mutation, acceptance, and deployment."
)

SPECIALIST_RESULT_BOUNDARY = (
    "SPECIALIST RESULT — ADVISORY EVIDENCE / PROPOSED OUTPUT — NO AUTHORITY"
)


class LocalLoadError(RuntimeError):
    """Raised when a local model/service cannot be made resident in time."""


class LockUnavailable(RuntimeError):
    """Raised when the accelerator lock cannot be acquired within the bound."""


class LocalOnlyViolation(RuntimeError):
    """Raised when a cloud fallback would be needed but local-only was demanded."""


# ---------------------------------------------------------------------------
# Route-reason object
# ---------------------------------------------------------------------------

@dataclass
class RouteReason:
    route: str
    reason_code: str
    signals: List[str] = field(default_factory=list)
    confidence: str = "high"          # high | medium | low
    overrides_pin: bool = False
    prompt_chars: int = 0
    prompt_sha8: str = ""
    est_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RouteDecision:
    route: str
    model: str
    provider: Optional[str]
    reason: RouteReason
    keep_alive: Any = None
    dispatch: str = "agent"           # agent | specialist | image | uncut | pin
    notice: str = ""
    local_only: bool = False
    explicit: bool = False            # came from /route or /model --once

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reason"] = self.reason.to_dict()
        return d


def _prompt_fingerprint(prompt: str) -> Tuple[int, str, int]:
    text = prompt or ""
    sha8 = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:8]
    return len(text), sha8, _dr.estimate_tokens(text)


def _reason(route: str, code: str, prompt: str, signals: Iterable[str] = (),
            confidence: str = "high", overrides_pin: bool = False) -> RouteReason:
    chars, sha8, est = _prompt_fingerprint(prompt)
    return RouteReason(
        route=route, reason_code=code, signals=sorted(set(signals)),
        confidence=confidence, overrides_pin=overrides_pin,
        prompt_chars=chars, prompt_sha8=sha8, est_tokens=est,
    )


# ---------------------------------------------------------------------------
# Deterministic classifiers
# ---------------------------------------------------------------------------

# Each deep signal is a named category; the score counts DISTINCT categories,
# so repeating one word never promotes.  Length is never a signal.
_DEEP_SIGNALS: Dict[str, re.Pattern] = {
    "architecture": re.compile(
        r"\b(architect(?:ure|ural)|system design|design (?:a|the|our) (?:system|platform|pipeline)|"
        r"high[- ]availability|fault[- ]toleran\w*|scalab\w+|distributed system)\b", re.I),
    "strategy": re.compile(
        r"\b(strateg(?:y|ic|ies)|roadmap|go[- ]to[- ]market|business model|revenue model|"
        r"pricing (?:strategy|model)|market positioning|competitive (?:analysis|landscape)|"
        r"unit economics)\b", re.I),
    "root_cause": re.compile(
        r"\b(root[- ]cause|post[- ]?mortem|why (?:did|does|is) .{0,60}\b(?:fail|break|crash|degrad)\w*"
        r".{0,60}\b(?:across|between|and) )", re.I),
    "threat": re.compile(
        r"\b(threat[- ]model\w*|attack surface|adversar\w+|security posture|kill chain)\b", re.I),
    "conflict": re.compile(
        r"\b(conflicting|contradict\w*|inconsistent (?:evidence|data|reports|results)|"
        r"reconcile (?:the )?(?:evidence|findings|reports|data))\b", re.I),
    "compare": re.compile(
        r"\b(compare|trade[- ]?offs?|pros and cons|weigh|evaluate (?:the )?(?:options|alternatives|approaches)|"
        r"(?:versus|vs\.?)\b)", re.I),
    "constraints": re.compile(
        r"\b(constraints?|dependenc(?:y|ies)|interdependen\w+|interacting|coupled|"
        r"competing (?:requirements|priorities|goals))\b", re.I),
    "causal_chain": re.compile(
        r"\b(cascad\w+|chain of (?:events|failures|causes)|second[- ]order|knock[- ]on|"
        r"downstream (?:effects?|impacts?|consequences)|ripple effects?)\b", re.I),
    "explicit_deep": re.compile(
        r"\b(deep(?:ly)? (?:analy[sz]e|analysis|dive|think|reasoning)|think (?:deeply|hard|carefully)|"
        r"thorough(?:ly)? (?:analy[sz]e|analysis|review|assessment)|in[- ]depth|rigorous(?:ly)?|"
        r"comprehensive (?:analysis|review|assessment|plan)|first[- ]principles)\b", re.I),
    "multi_system": re.compile(
        r"\b((?:multi|cross)[- ]?(?:system|service|team|region|cloud|repo|tenant)\w*|"
        r"across (?:the |our |all |multiple |several )?(?:systems|services|fleet|stack|repos|repositories|"
        r"environments|machines|nodes|teams)|multiple (?:systems|services|components|subsystems|dependencies)|"
        r"end[- ]to[- ]end)\b", re.I),
    "migration": re.compile(
        r"\b(migration plan|migrate (?:the |our )?(?:entire|whole|platform|fleet|stack)|"
        r"re[- ]?platform\w*|re[- ]?architect\w*)\b", re.I),
}

# Explicit categories that alone justify DEEP_LOCAL (the Chairman asked for it).
_DEEP_SELF_SUFFICIENT = frozenset({"explicit_deep", "threat"})

# Simple-form prompts (status, factual lookup, tiny asks) — HOME_FAST unless an
# explicit deep request is present.  Word-bounded so "status" inside a deep
# post-mortem does not veto.
_HOME_SIMPLE_FORM = re.compile(
    r"^\s*(?:hey|hi|hello|ok|okay|please|pls|dumbledore|can you|could you|quick(?:ly)?|just)?[\s,:-]*"
    r"(?:what(?:'s| is| are| time| day)|status|is (?:the |it |my )?\w+ (?:up|down|running|online|active|healthy)|"
    r"how (?:many|much|do i|to)|where (?:is|are)|who (?:is|was)|when (?:is|was|did|does)|"
    r"show me|list|check|ping|remind me|summari[sz]e|tl;?dr|rewrite|rephrase|translate|"
    r"define|spell|convert|calculate|what does .{1,40} mean)\b",
    re.I,
)

_LOCAL_ONLY = re.compile(
    r"\b(local[- ]only|offline[- ]only|(?:do not|don'?t|never) (?:use|touch|call|hit) (?:the |any )?cloud|"
    r"no cloud|keep (?:it|this|everything) local|stay(?:s)? local|on[- ]prem(?:ise|ises)?(?: only)?|"
    r"must (?:run|stay|be) local|local execution (?:only|required))\b", re.I,
)
# Explicit cloud SELECTION: an imperative aimed at the cloud/Sol, never a
# question about it ("which cloud model do we use?" is not a selection).
_EXPLICIT_CLOUD = re.compile(
    r"^\s*(?:please\s+|dumbledore[,\s]+|ok\s+|just\s+)*"
    r"(?:use|switch to|route (?:this|it|that) to|send (?:this|it|that) to|run (?:this|it|that) (?:on|in|via)|"
    r"answer (?:this|it) (?:on|in|via|with)|do (?:this|it) (?:on|in|via|with))\s+"
    r"(?:the\s+)?(?:cloud|sol|gpt-?5\.?6[- ]?sol)\b", re.I,
)

_CODE_HEAVY_SIGNALS: Dict[str, re.Pattern] = {
    "repo_wide": re.compile(
        r"\b(repo(?:sitory)?[- ]wide|across the (?:repo|repository|codebase|project|monorepo)|"
        r"(?:entire|whole|full) (?:repo|repository|codebase|project|monorepo)|"
        r"every (?:file|module|package|service)|all (?:the )?(?:files|modules|packages|services|callers|call sites)|"
        r"codebase[- ]wide|project[- ]wide)\b", re.I),
    "multi_file": re.compile(
        r"\b(multi[- ]?file|multiple files|several files|many files|coordinated (?:change|edit|update)s?|"
        r"across (?:\d+|several|multiple|many) (?:files|modules|packages|services))\b", re.I),
    "cross_language": re.compile(
        r"\b(cross[- ]language|polyglot|"
        r"(?:python|rust|go|golang|typescript|javascript|java|c\+\+|c#|kotlin|swift|ruby|php)\b.{0,40}"
        r"\b(?:to|into|and|with|from) (?:python|rust|go|golang|typescript|javascript|java|c\+\+|c#|kotlin|swift|ruby|php))\b",
        re.I),
    "migration": re.compile(
        r"\b(major (?:migration|refactor|rewrite|upgrade)|migrat(?:e|ion|ing) (?:the |our |from |to |off )|"
        r"port (?:the |our )?(?:whole|entire|app|service|codebase)|framework (?:upgrade|migration)|"
        r"schema migration|monorepo)\b", re.I),
    "concurrency": re.compile(
        r"\b(race condition|deadlock|thread[- ]safe\w*|concurren\w+|lock contention|"
        r"distributed state|state machine bug|data race|livelock|reentran\w+)\b", re.I),
    "arch_plus_impl": re.compile(
        r"\b((?:design|architect) and (?:implement|build)|architecture (?:and|plus|with) implementation|"
        r"end[- ]to[- ]end implementation|implement (?:the )?(?:whole|entire|full) (?:feature|system|pipeline|service))\b",
        re.I),
    "dependencies": re.compile(
        r"\b(dependency (?:upgrade|changes|graph|tree|overhaul)|upgrade (?:all|every|the) dependenc\w+|"
        r"bump (?:all|every) (?:the )?(?:deps|dependencies|packages)|breaking changes? (?:across|in) )\b", re.I),
    "explicit_heavy": re.compile(
        r"\b((?:big|large|heavy|large local|big local) coder|qwen3[- ]coder(?:[- ]next)?|coder[- ]next|"
        r"79\.?7?b|heavy code(?: lane| route)?|code[- ]heavy)\b", re.I),
}

_CODE_ARCH_DISCUSSION = re.compile(
    r"\b(architect(?:ure|ural)|system design|design (?:pattern|choice|decision)s?|"
    r"microservices?|monolith|event[- ]driven|cqrs|hexagonal|clean architecture|"
    r"domain[- ]driven|service mesh|sharding|partitioning|consistency model)\b", re.I)

# Prose image-generation order (distinct from the explicit /quality /literal
# /brand controls, which stay exactly as they were).  Requires a leading
# generation verb AND a visual noun within reach, no question form, and no
# reference to an existing file/attachment (that is editing, not generation).
_IMG_VERB = (r"(?:make|create|generate|draw|render|paint|illustrate|conjure|dream up)")
_IMG_NOUN = (r"(?:image|images|picture|pictures|photo|photos|photograph|illustration|"
             r"artwork|art piece|logo|poster|portrait|wallpaper|painting|drawing|"
             r"rendering|icon|banner|sketch|concept art|cover art|thumbnail)")
# verb (me|us)? (article)? (adjective){0,2} NOUN — the noun must sit right after
# at most two modifiers, so "create a script that generates an image" or
# "design a scalable architecture for the image pipeline" never qualify.
_IMG_PROSE_ORDER = re.compile(
    r"^\s*(?:please\s+|hey\s+|dumbledore[,\s]+|can you\s+|could you\s+|i (?:want|need|would like)(?: you to)?\s+|"
    r"let'?s\s+)*"
    rf"{_IMG_VERB}\b(?:\s+(?:me|us))?(?:\s+(?:an?|the|some|another|one))?(?:\s+[\w'-]+){{0,2}}?\s+{_IMG_NOUN}\b",
    re.I,
)
_IMG_PROSE_VETO = re.compile(
    r"\b(this (?:image|picture|photo|file)|the attached|attached (?:image|picture|photo|file)|"
    r"that (?:image|picture|photo)|my (?:image|picture|photo)|existing (?:image|picture|photo)|"
    r"current (?:image|picture|photo)|the (?:image|picture|photo) (?:i|you) (?:sent|uploaded|attached)|"
    r"crop|resize|rotate|edit|modify|retouch|remove (?:the )?background|upscale|convert|compress|"
    r"\.(?:png|jpe?g|webp|gif|bmp|tiff?|svg)\b|"
    r"diagram|chart|graph|plot|flowchart|schematic|wireframe|mockup|svg|html|css|canvas|matplotlib|"
    r"screenshot|screen shot|qr code|barcode|table|"
    r"script|scripts|function|python|javascript|typescript|code|cli|api|json|yaml|sql|react|component|"
    r"class|generator|pipeline|architecture|brief|description|novel|story|characters?|essay|"
    r"copy|caption|set of|in python|in js|in go|in rust)\b", re.I,
)


def deep_signal_categories(prompt: str) -> List[str]:
    text = prompt or ""
    return [name for name, rx in _DEEP_SIGNALS.items() if rx.search(text)]


def code_heavy_signal_categories(prompt: str) -> List[str]:
    text = prompt or ""
    return [name for name, rx in _CODE_HEAVY_SIGNALS.items() if rx.search(text)]


def is_local_only_request(prompt: str) -> bool:
    return bool(_LOCAL_ONLY.search(prompt or ""))


def is_explicit_cloud_request(prompt: str) -> bool:
    text = (prompt or "").strip()
    if not text or _dr._IMG_QUESTION_VETO.search(text) or text.endswith("?"):
        return False
    return bool(_EXPLICIT_CLOUD.search(text))


def is_image_generation_prose(prompt: str, has_image: bool = False) -> bool:
    """True only for a prose ORDER to generate a fresh image.

    Attached image + anything is never generation (VISION or img2img refusal).
    Question forms, references to existing files, and code/diagram nouns veto.
    """
    if has_image:
        return False
    text = (prompt or "").strip()
    if not text or len(text) > 1500:
        return False
    if _dr._IMG_QUESTION_VETO.search(text) or text.endswith("?"):
        return False
    if _IMG_PROSE_VETO.search(text):
        return False
    return bool(_IMG_PROSE_ORDER.search(text))


def classify_general(prompt: str) -> Tuple[str, str, List[str], str]:
    """(route, reason_code, signals, confidence) for non-code, non-image text.

    DEEP_LOCAL requires >= 2 distinct deep categories, or one self-sufficient
    category (explicit deep request / threat modelling).  A simple-form
    prompt vetoes unless the deep request is explicit.  Prompt length is
    never consulted.  Anything below the bar is HOME_FAST.
    """
    text = prompt or ""
    cats = deep_signal_categories(text)
    explicit = [c for c in cats if c in _DEEP_SELF_SUFFICIENT]
    simple = bool(_HOME_SIMPLE_FORM.search(text))
    if "explicit_deep" in cats:
        return DEEP_LOCAL, "deep_explicit", cats, "high"
    if simple:
        code = "home_simple_form" if not cats else "home_simple_form_veto"
        return HOME_FAST, code, cats, "high"
    if explicit:
        return DEEP_LOCAL, "deep_explicit", cats, "high"
    if len(cats) >= 3:
        return DEEP_LOCAL, "deep_multi_signal", cats, "high"
    if len(cats) == 2:
        return DEEP_LOCAL, "deep_two_signal", cats, "medium"
    if len(cats) == 1:
        return HOME_FAST, "home_single_signal_insufficient", cats, "medium"
    return HOME_FAST, "home_default", [], "high"


# Extra imperative code-production forms the original guard misses: an
# imperative fix/change verb aimed at a named source file or a concrete
# defect noun.  Discussion still vetoes (same precedence as the base guard).
_EXTRA_CODE_PRODUCTION = re.compile(
    r"\b(?:fix|patch|repair|resolve|correct|change|update|modify|remove|delete|rename|extract|inline|"
    r"migrate|port|convert|speed up|harden)\b"
    r".{0,80}?"
    r"(?:\b[\w./-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|kt|swift|c|cc|cpp|h|hpp|cs|rb|php|sh|bash|zsh|"
    r"yaml|yml|json|toml|ini|cfg|sql|proto|tf|dockerfile|make(?:file)?)\b|"
    r"\b(?:race condition|deadlock|memory leak|null pointer|segfault|stack overflow|off[- ]by[- ]one|"
    r"regression|flaky test|failing test|type error|lint error|compile error|build error|"
    r"exception|traceback|crash|infinite loop)\b)",
    re.I,
)


# Imperative "make me a <thing>" forms with one optional modifier word
# ("write a bash script", "build a small cli") and code-object imperatives
# ("migrate the project", "port the service", "upgrade all dependencies").
_EXTRA_CODE_PRODUCTION_MAKE = re.compile(
    r"\b(?:write|create|generate|build|make|implement|code up|draft|scaffold)\b(?:\s+me)?"
    r"(?:\s+(?:a|an|the|some|another))?(?:\s+[\w+#.-]+){0,2}?\s+"
    r"(?:script|function|class|module|program|cli|tool|test|tests|regex|query|dockerfile|makefile|"
    r"playbook|workflow|migration|endpoint|handler|parser|schema|snippet|library|package|plugin|"
    r"decorator|middleware|component|hook|cronjob|systemd unit|unit file|api)\b",
    re.I,
)
_EXTRA_CODE_PRODUCTION_OBJECT = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+|go ahead and\s+|dumbledore[,\s]+)*"
    r"(?:migrate|port|convert|upgrade|bump|refactor|rewrite|rename|split|merge|extract|modernize|"
    r"coordinate|vendor|de-?duplicate|consolidate|re-?implement|re-?write)\b.{0,80}?"
    r"\b(?:project|codebase|code|repo|repository|service|app|application|module|modules|package|packages|"
    r"framework|dependenc\w+|deps|library|libraries|monorepo|parser|change|changes|files?|classes?|"
    r"functions?|scripts?|tests?|schema|pipeline|backend|frontend)\b",
    re.I,
)
# Prose objects: a "rewrite/summarize" of text is NOT code production unless a
# code object is also named.
_PROSE_OBJECT = re.compile(
    r"\b(?:paragraph|email|e-mail|essay|letter|message|sentence|blurb|bio|post|tweet|caption|"
    r"headline|copy|summary|abstract|article|story|poem|speech|memo|report|note|announcement|"
    r"description|pitch|proposal|reply|response text|wording)\b", re.I,
)
_CODE_OBJECT = re.compile(
    r"\b(?:code|function|class|script|module|test|tests|bug|file|files|repo|repository|codebase|api|"
    r"endpoint|query|regex|method|program|package|library|cli|handler|parser|schema|migration|"
    r"dependenc\w+|service|pipeline|dockerfile|makefile|yaml|json|sql|python|bash|typescript|"
    r"javascript|rust|golang|java|c\+\+)\b|"
    r"\.(?:py|js|jsx|ts|tsx|go|rs|java|kt|swift|c|cc|cpp|h|hpp|cs|rb|php|sh|bash|zsh|yaml|yml|json|toml|"
    r"ini|cfg|sql|proto|tf)\b", re.I,
)
# "how do I fix …?" / "what's the best way to …" are questions, not orders.
_CODE_INTERROGATIVE = re.compile(
    r"(^|\b)(how (?:do|can|could|should|would|might) (?:i|we|you|one)\b|what'?s the best way\b|"
    r"what is the best way\b|is there a way\b|any idea\w*\b|do you know how\b|"
    r"can (?:i|we|you) (?:just )?(?:fix|change|refactor|migrate|patch)\b.*\?\s*$)", re.I,
)


def is_code_production(prompt: str) -> bool:
    """Base guard OR the extra imperative forms; discussion always vetoes; a
    prose object (paragraph/email/...) without any code object vetoes."""
    text = prompt or ""
    matched = bool(
        _dr.is_code_production(text)
        or _EXTRA_CODE_PRODUCTION.search(text)
        or _EXTRA_CODE_PRODUCTION_MAKE.search(text)
        or _EXTRA_CODE_PRODUCTION_OBJECT.search(text)
    )
    if not matched:
        return False
    if _dr._CODE_DISCUSSION.search(text) or _CODE_INTERROGATIVE.search(text):
        return False
    if _PROSE_OBJECT.search(text) and not _CODE_OBJECT.search(text):
        return False
    return True


def classify_code(prompt: str) -> Tuple[Optional[str], str, List[str]]:
    """(route|None, reason_code, signals).

    Production intent is decided by the build/edit guard (``is_code_production``
    — discussion wins).  Only a production ask can reach CODE_*; heavy
    signals then escalate.
    """
    text = prompt or ""
    if not is_code_production(text):
        return None, "not_code_production", []
    heavy = code_heavy_signal_categories(text)
    if heavy:
        return CODE_HEAVY, "code_heavy_signal", heavy
    return CODE_FAST, "code_fast_default", []


def is_complex_code_architecture_discussion(prompt: str) -> bool:
    """Code DISCUSSION (not production) about architecture with >= 1 other
    deep category -> DEEP_LOCAL.  Plain code discussion stays HOME_FAST."""
    text = prompt or ""
    if is_code_production(text):
        return False
    if not _dr._CODE_DISCUSSION.search(text):
        return False
    if not _CODE_ARCH_DISCUSSION.search(text):
        return False
    cats = set(deep_signal_categories(text)) - {"architecture"}
    return len(cats) >= 1


# ---------------------------------------------------------------------------
# /route command grammar (one-turn override)
# ---------------------------------------------------------------------------

ROUTE_COMMANDS = {
    "home": HOME_FAST,
    "deep": DEEP_LOCAL,
    "code": CODE_FAST,
    "code-heavy": CODE_HEAVY,
    "codeheavy": CODE_HEAVY,
    "heavy": CODE_HEAVY,
    "cloud": CLOUD_SAFE,
    "flux": IMAGE_GENERATION,
    "image": IMAGE_GENERATION,
}
ROUTE_META_COMMANDS = ("status", "auto", "model")


@dataclass
class RouteCommand:
    kind: str                 # "route" | "status" | "auto" | "model" | "help"
    route: Optional[str] = None
    model: Optional[str] = None
    prompt: str = ""
    error: str = ""


def parse_route_command(text: str) -> Optional[RouteCommand]:
    """Parse ``/route <verb> [prompt]``.  Returns None when not a /route."""
    raw = (text or "").strip()
    if not raw:
        return None
    tokens = raw.split()
    if tokens[0].lower() != "/route":
        return None
    if len(tokens) == 1:
        return RouteCommand(kind="help")
    verb = tokens[1].lower().strip(":,")
    rest = tokens[2:]
    if verb == "status":
        return RouteCommand(kind="status")
    if verb in ("auto", "clear", "off", "reset"):
        return RouteCommand(kind="auto")
    if verb == "model":
        if not rest:
            return RouteCommand(kind="model", error="usage: /route model <ollama-tag> [prompt]")
        tag = rest[0]
        if tag in _dr.ABLITERATED_MODELS:
            return RouteCommand(kind="model", error="abliterated models are uncut-lane only")
        if tag not in LOCAL_SPECIALIST_MODELS:
            return RouteCommand(
                kind="model", model=tag,
                error=f"unknown local specialist tag {tag!r}; allowed: "
                      + ", ".join(sorted(LOCAL_SPECIALIST_MODELS)),
            )
        return RouteCommand(kind="model", route=EXPLICIT_PIN, model=tag, prompt=" ".join(rest[1:]))
    if verb in ROUTE_COMMANDS:
        return RouteCommand(kind="route", route=ROUTE_COMMANDS[verb], prompt=" ".join(rest))
    return RouteCommand(kind="help", error=f"unknown route {verb!r}")


ROUTE_HELP = (
    "Route overrides (one turn each):\n"
    "/route home <prompt> — fast 9B\n"
    "/route deep <prompt> — local 35B 64k\n"
    "/route code <prompt> — qwen2.5-coder 14B\n"
    "/route code-heavy <prompt> — Qwen3 Coder Next\n"
    "/route cloud <prompt> — GPT-5.6 Sol\n"
    "/route flux <prompt> — local FLUX image\n"
    "/route model <ollama-tag> <prompt> — force one local specialist\n"
    "/route status — read-only route state\n"
    "/route auto — clear an armed override\n"
    "A bare /route <verb> arms the override for your next message."
)


# ---------------------------------------------------------------------------
# Top-level decision (priority order from the mission)
# ---------------------------------------------------------------------------

def decide_route(
    prompt: str,
    *,
    has_image: bool = False,
    mode: str = "home",
    pinned_model: Optional[str] = None,
    pinned_provider: Optional[str] = None,
    override: Optional[str] = None,
    override_model: Optional[str] = None,
    history_tokens: Optional[int] = None,
    explicit_image_control: bool = False,
) -> RouteDecision:
    """Pure function: (turn facts) -> RouteDecision.  Never touches I/O.

    Priority:
      1. governance/safety (uncut + control words are handled BEFORE this in
         the gateway and never enter here; local-only is honoured throughout)
      2. explicit one-turn override (/route ..., /model --once)
      3. image-generation intent
      4. attached-image understanding
      5. imperative code production
      6. complex code-architecture discussion
      7. general deep reasoning
      8. explicit persistent pin (ordinary reasoning)
      9. HOME_FAST default
    """
    text = prompt or ""
    local_only = is_local_only_request(text)
    pinned = (mode == "pinned") and bool(pinned_model)
    pin_is_local = bool(pinned_provider and str(pinned_provider).startswith("custom:")) or (
        pinned_model in LOCAL_SPECIALIST_MODELS if pinned_model else False
    )

    # 2. explicit one-turn override ---------------------------------------
    if override:
        if override == EXPLICIT_PIN and override_model:
            _dr.assert_not_abliterated(override_model)
            return RouteDecision(
                route=EXPLICIT_PIN, model=override_model, provider=None,
                reason=_reason(EXPLICIT_PIN, "override_model", text, ["explicit"], overrides_pin=pinned),
                keep_alive=KEEP_ALIVE[CODE_FAST], dispatch="specialist",
                local_only=local_only, explicit=True,
            )
        if override == CLOUD_SAFE and local_only:
            return RouteDecision(
                route=HOME_FAST, model=HOME_FAST_MODEL, provider=HOME_FAST_PROVIDER,
                reason=_reason(HOME_FAST, "override_cloud_refused_local_only", text, ["local_only"]),
                keep_alive=KEEP_ALIVE[HOME_FAST], dispatch="agent",
                notice="⚠️ Cloud route refused: this request demanded local-only execution.",
                local_only=True, explicit=True,
            )
        if override == VISION and not has_image:
            override = HOME_FAST
        if override == IMAGE_GENERATION and has_image:
            return RouteDecision(
                route=HOME_FAST, model=HOME_FAST_MODEL, provider=HOME_FAST_PROVIDER,
                reason=_reason(HOME_FAST, "override_flux_refused_attachment", text, ["attachment"]),
                keep_alive=KEEP_ALIVE[HOME_FAST], dispatch="agent",
                notice=_dr.IMG2IMG_REFUSAL, local_only=local_only, explicit=True,
            )
        return _decision_for_route(
            override, text, code="override_" + override.lower(), signals=["explicit"],
            pinned=pinned, local_only=local_only, explicit=True,
            history_tokens=history_tokens,
        )

    # 3. image generation --------------------------------------------------
    if explicit_image_control and not has_image:
        return _decision_for_route(IMAGE_GENERATION, text, code="image_control_explicit",
                                   signals=["image_control"], pinned=pinned, local_only=local_only)
    if is_image_generation_prose(text, has_image=has_image):
        return _decision_for_route(IMAGE_GENERATION, text, code="image_generation_prose",
                                   signals=["image_prose"], pinned=pinned, local_only=local_only)

    # 4. attached image ----------------------------------------------------
    if has_image:
        d = _decision_for_route(VISION, text, code="vision_attachment", signals=["attachment"],
                                pinned=pinned, local_only=local_only)
        if _dr.estimate_tokens(text) > _dr.IMAGE_OVERFLOW_TOKENS:
            d.reason.reason_code = "vision_attachment_overflow_warn"
            d.notice = (
                f"⚠️ This attachment plus ~{_dr.estimate_tokens(text):,} tokens of text exceeds "
                "the vision lane's ~32k window — trim or split it for a reliable read."
            )
        return d

    # explicit cloud in prose (user explicitly selects cloud) ----------------
    if is_explicit_cloud_request(text) and not local_only:
        return _decision_for_route(CLOUD_SAFE, text, code="cloud_explicit_prose",
                                   signals=["explicit_cloud"], pinned=pinned, local_only=False)

    # 5. imperative code production ---------------------------------------
    code_route, code_reason, code_signals = classify_code(text)
    if code_route:
        return _decision_for_route(code_route, text, code=code_reason, signals=code_signals,
                                   pinned=pinned, local_only=local_only)

    # 6. complex code-architecture discussion -----------------------------
    if is_complex_code_architecture_discussion(text):
        return _decision_for_route(DEEP_LOCAL, text, code="deep_code_architecture_discussion",
                                   signals=deep_signal_categories(text) + ["code_discussion"],
                                   pinned=pinned, pin_is_local=pin_is_local, local_only=local_only,
                                   history_tokens=history_tokens)

    # 7. general deep reasoning -------------------------------------------
    route, code, signals, confidence = classify_general(text)
    if route == DEEP_LOCAL:
        return _decision_for_route(DEEP_LOCAL, text, code=code, signals=signals,
                                   confidence=confidence, pinned=pinned, pin_is_local=pin_is_local,
                                   local_only=local_only, history_tokens=history_tokens)

    # 8. persistent pin for ordinary reasoning ----------------------------
    if pinned:
        return RouteDecision(
            route=EXPLICIT_PIN, model=pinned_model, provider=pinned_provider,
            reason=_reason(EXPLICIT_PIN, "pin_persistent", text, signals + ["pinned"], confidence),
            keep_alive=(KEEP_ALIVE["DEEP_LOCAL_PIN"] if pin_is_local else None),
            dispatch="pin", local_only=local_only,
        )

    # 9. HOME_FAST default -------------------------------------------------
    return RouteDecision(
        route=HOME_FAST, model=HOME_FAST_MODEL, provider=HOME_FAST_PROVIDER,
        reason=_reason(HOME_FAST, code, text, signals, confidence),
        keep_alive=KEEP_ALIVE[HOME_FAST], dispatch="agent", local_only=local_only,
    )


def _decision_for_route(
    route: str, text: str, *, code: str, signals: Sequence[str], confidence: str = "high",
    pinned: bool = False, pin_is_local: bool = False, local_only: bool = False,
    explicit: bool = False, history_tokens: Optional[int] = None,
) -> RouteDecision:
    overrides_pin = pinned
    if route == HOME_FAST:
        return RouteDecision(
            route=HOME_FAST, model=HOME_FAST_MODEL, provider=HOME_FAST_PROVIDER,
            reason=_reason(HOME_FAST, code, text, signals, confidence, overrides_pin),
            keep_alive=KEEP_ALIVE[HOME_FAST], dispatch="agent", local_only=local_only, explicit=explicit,
        )
    if route == DEEP_LOCAL:
        # A cloud pin is strictly more capable than the local 35B and is an
        # explicit Chairman preference: preserve it for deep work.  A LOCAL pin
        # (e.g. 9B) is overridden by DEEP_LOCAL, and a 35B pin is the same model.
        if pinned and not pin_is_local and not explicit:
            return RouteDecision(
                route=EXPLICIT_PIN, model="", provider=None,
                reason=_reason(EXPLICIT_PIN, "deep_intent_cloud_pin_preserved", text,
                               list(signals) + ["pinned_cloud"], confidence, overrides_pin=False),
                dispatch="pin", local_only=local_only,
            )
        fits = history_tokens is None or history_tokens <= DEEP_LOCAL_AGENT_LOOP_HISTORY_BUDGET
        return RouteDecision(
            route=DEEP_LOCAL, model=DEEP_LOCAL_MODEL, provider=DEEP_LOCAL_PROVIDER,
            reason=_reason(DEEP_LOCAL, code if fits else code + "_specialist_pack", text,
                           list(signals) + ([] if fits else ["history_exceeds_window"]),
                           confidence, overrides_pin),
            keep_alive=KEEP_ALIVE["DEEP_LOCAL_AUTO"],
            dispatch="agent" if fits else "specialist",
            local_only=local_only, explicit=explicit,
        )
    if route == CODE_FAST:
        return RouteDecision(
            route=CODE_FAST, model=CODE_FAST_MODEL, provider=None,
            reason=_reason(CODE_FAST, code, text, signals, confidence, overrides_pin),
            keep_alive=KEEP_ALIVE[CODE_FAST], dispatch="specialist", local_only=local_only, explicit=explicit,
        )
    if route == CODE_HEAVY:
        return RouteDecision(
            route=CODE_HEAVY, model=CODE_HEAVY_MODEL, provider=None,
            reason=_reason(CODE_HEAVY, code, text, signals, confidence, overrides_pin),
            keep_alive=(KEEP_ALIVE["CODE_HEAVY_SESSION"] if explicit else KEEP_ALIVE["CODE_HEAVY_ONESHOT"]),
            dispatch="specialist", local_only=local_only, explicit=explicit,
        )
    if route == VISION:
        return RouteDecision(
            route=VISION, model=VISION_MODEL, provider=VISION_PROVIDER,
            reason=_reason(VISION, code, text, signals, confidence, overrides_pin),
            keep_alive=KEEP_ALIVE[VISION], dispatch="agent", local_only=local_only, explicit=explicit,
        )
    if route == IMAGE_GENERATION:
        return RouteDecision(
            route=IMAGE_GENERATION, model=IMAGE_GEN_LABEL, provider="comfyui",
            reason=_reason(IMAGE_GENERATION, code, text, signals, confidence, overrides_pin),
            keep_alive=None, dispatch="image", local_only=local_only, explicit=explicit,
        )
    if route == CLOUD_SAFE:
        return RouteDecision(
            route=CLOUD_SAFE, model=CLOUD_SAFE_MODEL, provider=CLOUD_SAFE_PROVIDER,
            reason=_reason(CLOUD_SAFE, code, text, signals, confidence, overrides_pin),
            keep_alive=None, dispatch="agent", local_only=local_only, explicit=explicit,
        )
    raise ValueError(f"unknown route {route!r}")


def cloud_fallback_decision(original: RouteDecision, failure_code: str) -> RouteDecision:
    """CLOUD_SAFE replacement for a failed local route (or raise if local-only)."""
    if original.local_only:
        raise LocalOnlyViolation(
            f"{original.route} failed ({failure_code}) and the request demanded local-only execution"
        )
    return RouteDecision(
        route=CLOUD_SAFE, model=CLOUD_SAFE_MODEL, provider=CLOUD_SAFE_PROVIDER,
        reason=RouteReason(
            route=CLOUD_SAFE, reason_code=f"fallback:{original.route.lower()}:{failure_code}",
            signals=list(original.reason.signals) + ["fallback"],
            confidence="high", overrides_pin=original.reason.overrides_pin,
            prompt_chars=original.reason.prompt_chars, prompt_sha8=original.reason.prompt_sha8,
            est_tokens=original.reason.est_tokens,
        ),
        keep_alive=None, dispatch="agent", local_only=False, explicit=original.explicit,
        notice=f"☁️ Local {original.route} unavailable ({failure_code}); answered by cloud-safe {CLOUD_SAFE_MODEL}.",
    )


# ---------------------------------------------------------------------------
# Specialist context pack (bounded; never mutates the session history)
# ---------------------------------------------------------------------------

def _msg_text(msg: Any) -> str:
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _clip(text: str, max_tokens: int) -> str:
    max_chars = max(0, max_tokens * 4)
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    return text[:head] + "\n…[truncated]…\n" + text[-(max_chars - head):]


def build_specialist_context_pack(
    history: Sequence[Any],
    task: str,
    *,
    budget_tokens: int,
    controlling_instruction: str = "",
    mission_state: str = "",
    critical_refs: Sequence[str] = (),
    qfb_refs: Sequence[str] = (),
    governance: Sequence[str] = (),
    previous_failures: Sequence[str] = (),
    recent_turns: int = SPECIALIST_RECENT_TURNS,
) -> dict:
    """Return {"messages": [...], "tokens": int, "turns_included": int,
    "truncated": bool}.  Builds a NEW message list from a read-only view of
    ``history``; the caller's session history object is never modified.
    """
    budget = max(2_000, int(budget_tokens))
    task_text = (task or "").strip()
    header_lines = []
    if controlling_instruction:
        header_lines.append("CONTROLLING CHAIRMAN INSTRUCTION:\n" + controlling_instruction.strip())
    if mission_state:
        header_lines.append("ACTIVE MISSION STATE:\n" + mission_state.strip())
    if critical_refs:
        header_lines.append("CRITICAL PATHS / IDS:\n" + "\n".join(f"- {r}" for r in critical_refs))
    if qfb_refs:
        header_lines.append("RELEVANT QFB REFERENCES (names only):\n" + "\n".join(f"- {r}" for r in qfb_refs))
    if governance:
        header_lines.append("GOVERNANCE CONSTRAINTS:\n" + "\n".join(f"- {g}" for g in governance))
    if previous_failures:
        header_lines.append("PREVIOUS VALIDATION FAILURES:\n" + "\n".join(f"- {f}" for f in previous_failures))

    # Reserve: task + header always fit (clipped); remaining budget -> recent turns.
    reserve_answer = max(1_500, budget // 6)
    header_text = "\n\n".join(header_lines)
    header_text = _clip(header_text, budget // 4)
    task_clipped = _clip(task_text, budget // 2)
    used = _dr.estimate_tokens(header_text) + _dr.estimate_tokens(task_clipped) + reserve_answer
    remaining = budget - used

    # Walk history from the tail, collecting user/assistant text turns only
    # (tool payloads and system entries are excluded by design).
    recent: List[dict] = []
    truncated = False
    turns = 0
    for msg in reversed(list(history or [])):
        if turns >= recent_turns:
            truncated = True
            break
        role = msg.get("role") if isinstance(msg, dict) else None
        if role not in ("user", "assistant"):
            continue
        text = _msg_text(msg).strip()
        if not text:
            continue
        if role == "assistant" and msg.get("tool_calls"):
            # Keep only the visible text of a tool-calling turn.
            text = text[:2000]
        clipped = _clip(text, min(4_000, max(200, remaining // 2)))
        cost = _dr.estimate_tokens(clipped) + 4
        if cost > remaining:
            truncated = True
            break
        recent.append({"role": role, "content": clipped})
        remaining -= cost
        turns += 1
    recent.reverse()

    messages: List[dict] = []
    if header_text:
        messages.append({"role": "system", "content": header_text})
    if recent:
        messages.append({"role": "system", "content": "RELEVANT RECENT TURNS (bounded excerpt):"})
        messages.extend(recent)
    messages.append({"role": "user", "content": "CURRENT TASK:\n" + task_clipped})
    total = sum(_dr.estimate_tokens(m["content"]) + 4 for m in messages)
    return {
        "messages": messages,
        "tokens": total,
        "turns_included": turns,
        "truncated": truncated or (task_clipped != task_text),
        "budget": budget,
    }


def specialist_system_prompt(model: str, route: str) -> str:
    if route in (CODE_FAST, CODE_HEAVY) or model in (CODE_FAST_MODEL, CODE_HEAVY_MODEL):
        role_prompt = (
            "You are the local coding specialist served through Ollama.\n\n"
            f"Your configured backend model identifier is exactly: {model}\n"
            "When asked which model you are, report that exact identifier.\n"
            "Do not claim to be GPT-4, Claude, Gemini, or any other model.\n\n"
            "Provide technically precise and directly usable work.\n"
            "Do not invent files, commands, APIs, execution results, or successful tests.\n"
            "State clearly when more evidence is required.\n"
            "Prefer complete commands, patches, or replacement files.\n"
            "Separate findings, proposed changes, verification steps, and risks."
        )
    else:
        role_prompt = (
            "You are Dumbledore's local deep-reasoning specialist served through Ollama.\n\n"
            f"Your configured backend model identifier is exactly: {model}\n"
            "You receive a bounded context pack, not the full conversation. Reason "
            "carefully, state assumptions explicitly, separate evidence from inference, "
            "and do not invent facts, files, or results."
        )
    return role_prompt + "\n\n" + SPECIALIST_GOVERNANCE_BLOCK


def mark_specialist_result(content: Any) -> str:
    """Return a specialist result with exactly one leading authority boundary.

    The helper is pure and idempotent so delivery, fallback, and persistence
    layers may all apply it without producing duplicate markers.
    """
    text = "" if content is None else str(content).strip()
    marker = SPECIALIST_RESULT_BOUNDARY
    while text == marker or text.startswith(marker + "\n"):
        text = text[len(marker):].lstrip()
    return marker if not text else marker + "\n\n" + text


def run_specialist(
    model: str, pack: dict, *, route: str, keep_alive: Any = None,
    num_ctx: Optional[int] = None, timeout: float = INFERENCE_TIMEOUT,
    temperature: float = 0.2,
) -> dict:
    """Direct /api/chat call with the bounded pack.  No tools array.  Raises on
    HTTP/empty so the caller can fall back WITH notice."""
    import httpx

    _dr.assert_not_abliterated(model)
    if model not in LOCAL_SPECIALIST_MODELS:
        raise ValueError(f"{model!r} is not a router-dispatchable specialist")
    messages = [{"role": "system", "content": specialist_system_prompt(model, route)}]
    messages.extend(pack["messages"])
    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
        "think": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx or CONTEXT_WINDOW.get(model, 32768)},
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    assert "tools" not in payload
    t0 = time.time()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        if resp.status_code >= 500:
            raise LocalLoadError(f"ollama {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        data = resp.json()
    content = ((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(f"{model} returned no content: {str(data)[:200]}")
    return {
        "content": content,
        "seconds": time.time() - t0,
        "load_seconds": (data.get("load_duration") or 0) / 1e9,
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
    }


# ---------------------------------------------------------------------------
# Ollama controls (official API only: /api/ps, /api/generate keep_alive)
# ---------------------------------------------------------------------------

def ollama_ps(timeout: float = 5.0) -> List[dict]:
    import httpx
    try:
        with httpx.Client(timeout=timeout) as c:
            data = c.get(f"{OLLAMA_BASE}/api/ps").json()
        return list(data.get("models") or [])
    except Exception:
        return []


def ollama_loaded_names(timeout: float = 5.0) -> List[str]:
    return [m.get("name") or m.get("model") or "" for m in ollama_ps(timeout)]


def ollama_unload(model: str, timeout: float = 30.0) -> bool:
    """Equivalent of ``ollama stop <model>`` (keep_alive: 0 via the API)."""
    import httpx
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{OLLAMA_BASE}/api/generate", json={"model": model, "keep_alive": 0})
        return r.status_code < 400
    except Exception:
        return False


def ollama_load(model: str, keep_alive: Any, timeout: float) -> float:
    """Load (or refresh keep_alive of) a model with an empty prompt.  Returns
    seconds.  Raises LocalLoadError on any failure or timeout."""
    import httpx
    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout) as c:
            payload = {"model": model}
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive
            r = c.post(f"{OLLAMA_BASE}/api/generate", json=payload)
    except httpx.TimeoutException as exc:
        raise LocalLoadError(f"load of {model} exceeded {timeout:.0f}s") from exc
    except Exception as exc:
        raise LocalLoadError(f"load of {model} failed: {type(exc).__name__}: {exc}") from exc
    if r.status_code >= 400:
        raise LocalLoadError(f"ollama {r.status_code} loading {model}: {r.text[:200]}")
    return time.time() - t0


def apply_keep_alive(model: str, keep_alive: Any, timeout: float = 30.0) -> bool:
    """Post-turn keep-alive refresh.  Nonfatal."""
    try:
        if keep_alive is None:
            return True
        ollama_load(model, keep_alive, timeout)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ComfyUI / Flux service lifecycle (systemd user unit preferred)
# ---------------------------------------------------------------------------

def comfy_unit_available() -> bool:
    return _dr.comfy_unit_available()


def comfy_unit_state() -> str:
    return _dr.comfy_unit_state()


def comfy_start(ready_timeout: float = COMFY_READY_TIMEOUT) -> dict:
    """Start ComfyUI on demand (systemd unit preferred). Returns
    {seconds, mechanism, cold}. Raises LocalLoadError when not ready in time."""
    if _dr.comfy_is_up():
        return {"seconds": 0.0, "mechanism": "already_up", "cold": False}
    try:
        seconds = _dr.start_comfy(ready_timeout=ready_timeout)
    except Exception as exc:
        raise LocalLoadError(str(exc)) from exc
    return {"seconds": seconds, "mechanism": _dr.COMFY_LAST_START_MECHANISM, "cold": True}


def comfy_stop(wait: float = 30.0) -> bool:
    """Stop ComfyUI (systemd unit when active, else the launcher-process path)."""
    try:
        return bool(_dr.shutdown_comfy(wait=wait))
    except Exception:
        return not _dr.comfy_is_up()


def verify_png(path: str) -> Tuple[int, int]:
    """Return (width, height) for a valid, non-empty PNG; raise otherwise."""
    if not path or not os.path.exists(path):
        raise RuntimeError("render output missing")
    size = os.path.getsize(path)
    if size <= 0:
        raise RuntimeError("render output is empty")
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("render output is not a PNG")
    from PIL import Image
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        w, h = im.size
    if w <= 0 or h <= 0:
        raise RuntimeError("render output has zero dimensions")
    return w, h


def run_flux_generation(
    prompt: str, *, steps: int = _dr.IMAGE_GEN_STEPS, width: int = _dr.IMAGE_GEN_WIDTH,
    height: int = _dr.IMAGE_GEN_HEIGHT, seed: Optional[int] = None, timeout: float = 600.0,
) -> dict:
    """Start ComfyUI on demand, queue the fixed reviewed workflow, verify the
    PNG, copy it under /tmp/hermes-renders/.  Raises with the real ComfyUI
    error; never claims an image exists.  Lock/unload are the caller's job."""
    import shutil

    was_up = _dr.comfy_is_up()
    # run_image_generation starts ComfyUI itself when it is down (start_comfy:
    # systemd unit preferred, launcher fallback, Ollama unload interlock).
    _kwargs = {"steps": steps, "width": width, "height": height}
    if seed is not None:
        _kwargs["seed"] = seed
    if timeout != 600.0:
        _kwargs["timeout"] = timeout
    res = _dr.run_image_generation(prompt, **_kwargs)
    start = {
        "seconds": res.get("startup_seconds", 0.0),
        "mechanism": "already_up" if was_up else _dr.COMFY_LAST_START_MECHANISM,
        "cold": not was_up,
    }
    w, h = verify_png(res["path"])
    os.makedirs(RENDER_DIR, exist_ok=True)
    dest = os.path.join(RENDER_DIR, f"flux_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.png")
    shutil.copy2(res["path"], dest)
    verify_png(dest)
    res.update({
        "path": dest, "source_path": res["path"], "width": w, "height": h,
        "startup_seconds": start["seconds"], "start_mechanism": start["mechanism"],
        "cold": start["cold"], "bytes": os.path.getsize(dest),
    })
    return res


# ---------------------------------------------------------------------------
# Shared local-accelerator lock
# ---------------------------------------------------------------------------

class AcceleratorLock:
    """Single shared lock for every heavy local accelerator use.

    * flock on a file (shared by every gateway lane; the aider wrapper and
      the MCP coder server do NOT take it yet — noted limitation); metadata
      (owner, route, pid, ts) is written inside the file for /route status
      and stale diagnosis.
    * ``acquire`` is bounded by ``timeout`` and raises LockUnavailable —
      never blocks the gateway indefinitely.
    * Stale recovery: a dead owner pid releases flock automatically; we
      detect + record it.  A live holder is never killed.
    * Non-reentrant by design: a second acquire from the same process gets a
      fresh fd and simply times out (bounded), so it cannot deadlock.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or LOCK_PATH
        self._fd: Optional[int] = None
        self.meta: dict = {}
        self.result: str = "not_acquired"

    def _read_meta(self) -> dict:
        try:
            with open(self.path, "r") as fh:
                raw = fh.read().strip()
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    @staticmethod
    def _pid_alive(pid: Any) -> bool:
        try:
            pid = int(pid)
        except Exception:
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def acquire(self, owner: str, route: str, timeout: float = LOCK_TIMEOUT_DEFAULT,
                poll: float = 0.25) -> "AcceleratorLock":
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        deadline = time.time() + max(0.0, timeout)
        stale_seen: Optional[dict] = None
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                        raise
                prev = self._read_meta()
                if prev and not self._pid_alive(prev.get("pid")):
                    # Owner is dead: its flock is already gone by definition,
                    # so the next iteration acquires.  Record for telemetry.
                    stale_seen = prev
                if time.time() >= deadline:
                    os.close(fd)
                    self.result = "timeout"
                    holder = prev.get("owner") if prev else "unknown"
                    raise LockUnavailable(
                        f"accelerator lock held by {holder} (route={prev.get('route') if prev else '?'}, "
                        f"pid={prev.get('pid') if prev else '?'}) — waited {timeout:.0f}s"
                    )
                time.sleep(poll)
        except LockUnavailable:
            raise
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            raise
        self._fd = fd
        self.meta = {"owner": owner, "route": route, "pid": os.getpid(), "ts": time.time(),
                     "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if stale_seen and not self._pid_alive(stale_seen.get("pid")):
            self.meta["recovered_from"] = {k: stale_seen.get(k) for k in ("owner", "route", "pid", "ts_iso")}
            self.result = "stale_recovered"
        else:
            self.result = "acquired"
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, json.dumps(self.meta).encode())
            os.fsync(fd)
        except Exception:
            pass
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            os.ftruncate(self._fd, 0)
        except Exception:
            pass
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self._fd)
            finally:
                self._fd = None

    def __enter__(self) -> "AcceleratorLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def lock_status(path: Optional[str] = None) -> dict:
    lock = AcceleratorLock(path or LOCK_PATH)
    meta = lock._read_meta()
    if not meta:
        return {"held": False}
    alive = lock._pid_alive(meta.get("pid"))
    return {"held": alive, "stale": not alive, **{k: meta.get(k) for k in ("owner", "route", "pid", "ts_iso")}}


# ---------------------------------------------------------------------------
# Resource preflight (runs INSIDE the lock; the gateway calls it in an executor)
# ---------------------------------------------------------------------------

def prepare_local_target(
    target_model: str, *, route: str, keep_alive: Any, load_timeout: Optional[float] = None,
    exclusive: bool = True, stop_comfy: bool = True, preload: bool = True,
) -> dict:
    """Make ``target_model`` resident with the accelerator to itself.

    1. inspect Ollama (/api/ps) and ComfyUI;
    2. stop conflicting local models via keep_alive:0 (== ``ollama stop``);
    3. stop ComfyUI (systemd unit / launcher path) if it is up;
    4. load the target with the route's keep_alive, bounded by load_timeout;
    Never touches the ollama service, never kills processes, never writes
    any pin/mode state.  Raises LocalLoadError on failure.
    """
    _dr.assert_not_abliterated(target_model)
    t0 = time.time()
    previous = ollama_loaded_names()
    comfy_was_up = _dr.comfy_is_up()
    unloaded: List[str] = []
    for name in previous:
        if not name or name == target_model:
            continue
        base_same = name.split(":")[0] == target_model.split(":")[0] and (
            {name, target_model} <= {"qwen3.6:35b-a3b", DEEP_LOCAL_MODEL})
        if base_same:
            # Shared runner (derived tag) — `ollama ps` shows the base tag; leave it.
            continue
        if exclusive or name in HEAVY_LOCAL_MODELS or target_model in HEAVY_LOCAL_MODELS:
            if ollama_unload(name):
                unloaded.append(name)
    comfy_stopped = False
    if stop_comfy and comfy_was_up:
        comfy_stopped = comfy_stop()
        if not comfy_stopped:
            raise LocalLoadError("ComfyUI is holding the accelerator and could not be stopped")
    load_seconds = 0.0
    if preload:
        load_seconds = ollama_load(
            target_model, keep_alive,
            load_timeout if load_timeout is not None else LOAD_TIMEOUT.get(target_model, 180.0),
        )
    return {
        "target": target_model, "route": route, "previous_loaded": previous,
        "unloaded": unloaded, "comfy_was_up": comfy_was_up, "comfy_stopped": comfy_stopped,
        "load_seconds": round(load_seconds, 2), "preflight_seconds": round(time.time() - t0, 2),
    }


def prepare_for_flux() -> dict:
    """Unload every Ollama resident before Flux (16 GB card)."""
    previous = ollama_loaded_names()
    unloaded = [n for n in previous if n and ollama_unload(n)]
    return {"previous_loaded": previous, "unloaded": unloaded}


# ---------------------------------------------------------------------------
# Telemetry (bounded; never prompt text / credentials / QFB content)
# ---------------------------------------------------------------------------

_TELEMETRY_KEYS = (
    "route", "reason_code", "signals", "confidence", "model", "provider", "dispatch",
    "previous_loaded", "unloaded", "load_seconds", "inference_seconds", "fallback",
    "lock", "outcome", "error", "prompt_sha8", "prompt_chars", "est_tokens",
    "pin_preserved", "mode", "overrides_pin", "explicit", "pack_tokens", "pack_turns",
    "pack_truncated", "comfy", "keep_alive", "session_sha8", "history_tokens",
)


def log_route_event(**fields: Any) -> dict:
    rec: Dict[str, Any] = {"v": 2, "ts": time.time()}
    for key in _TELEMETRY_KEYS:
        if key in fields and fields[key] is not None:
            val = fields[key]
            if isinstance(val, float):
                val = round(val, 3)
            if isinstance(val, str) and len(val) > 300:
                val = val[:300] + "…"
            rec[key] = val
    try:
        os.makedirs(os.path.dirname(TELEMETRY_PATH), exist_ok=True)
        with open(TELEMETRY_PATH, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass
    return rec


def decision_telemetry(decision: RouteDecision, **extra: Any) -> dict:
    r = decision.reason
    return log_route_event(
        route=decision.route, reason_code=r.reason_code, signals=r.signals,
        confidence=r.confidence, model=decision.model, provider=decision.provider,
        dispatch=decision.dispatch, prompt_sha8=r.prompt_sha8, prompt_chars=r.prompt_chars,
        est_tokens=r.est_tokens, overrides_pin=r.overrides_pin, explicit=decision.explicit,
        keep_alive=decision.keep_alive, **extra,
    )


def recent_route_events(n: int = 5) -> List[dict]:
    try:
        with open(TELEMETRY_PATH, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 64_000))
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except Exception:
        return []
    out = []
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("v") == 2:
            out.append(rec)
        if len(out) >= n:
            break
    return out


def route_status_text(*, mode: dict, armed: Optional[str], pinned_model: Optional[str]) -> str:
    loaded = ollama_ps()
    lines = ["🧭 Route status (read-only)"]
    lines.append(f"mode: {mode.get('mode', 'home')}" + (f" (pin: {pinned_model})" if pinned_model else ""))
    lines.append(f"armed override: {armed or 'none'}")
    if loaded:
        for m in loaded:
            name = m.get("name") or m.get("model")
            vram = m.get("size_vram") or 0
            total = m.get("size") or 0
            pct = int(100 * vram / total) if total else 0
            lines.append(f"ollama: {name} ctx={m.get('context_length')} gpu={pct}% until={str(m.get('expires_at'))[11:19]}")
    else:
        lines.append("ollama: nothing loaded")
    lines.append(f"comfyui: {'up' if _dr.comfy_is_up() else 'down'} (unit {comfy_unit_state()})")
    ls = lock_status()
    if ls.get("held"):
        lines.append(f"accelerator lock: held by {ls.get('owner')} route={ls.get('route')} since {ls.get('ts_iso')}")
    elif ls.get("stale"):
        lines.append(f"accelerator lock: stale metadata from {ls.get('owner')} (owner gone; recoverable)")
    else:
        lines.append("accelerator lock: free")
    lines.append("models: HOME_FAST=%s DEEP_LOCAL=%s CODE_FAST=%s CODE_HEAVY=%s VISION=%s IMAGE=%s CLOUD_SAFE=%s" % (
        HOME_FAST_MODEL, DEEP_LOCAL_MODEL, CODE_FAST_MODEL, CODE_HEAVY_MODEL, VISION_MODEL,
        IMAGE_GEN_LABEL, CLOUD_SAFE_MODEL))
    for rec in recent_route_events(3):
        lines.append(
            f"recent: {rec.get('route')} {rec.get('reason_code')} -> {rec.get('model')} "
            f"[{rec.get('outcome', 'decided')}]"
        )
    return "\n".join(lines)


def route_signature(decision: RouteDecision) -> str:
    """Optional response suffix; OFF unless DUMBLEDORE_ROUTE_SIGNATURE=1."""
    if os.environ.get(ROUTE_SIGNATURE_ENV) != "1":
        return ""
    return f"\n\n[route {decision.route} · {decision.model} · {decision.reason.reason_code}]"
