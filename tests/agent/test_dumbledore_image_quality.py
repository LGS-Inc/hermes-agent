from __future__ import annotations

import json

import pytest

from agent import dumbledore_router as router


CASES = [
    ("Create me a futuristic image for Quantum Web Studios Inc",
     "a futuristic image for Quantum Web Studios Inc"),
    ("Create me an image of a futuristic city at sunset",
     "a futuristic city at sunset"),
    ("Make me a picture of an elegant owl in watercolor",
     "an elegant owl in watercolor"),
    ("Generate a cinematic portrait of a wizard",
     "a cinematic portrait of a wizard"),
    ("Draw an intricate steampunk airship",
     "an intricate steampunk airship"),
    ("Render a photorealistic forest with mist",
     "a photorealistic forest with mist"),
    ("Design a minimalist logo for a bakery",
     "a minimalist logo for a bakery"),
    ("Produce an atmospheric image showing moonlit ruins",
     "an atmospheric image showing moonlit ruins"),
    ("Paint a vibrant illustration depicting tropical birds",
     "a vibrant illustration depicting tropical birds"),
    ("Create a dark surreal artwork with floating islands",
     "a dark surreal artwork with floating islands"),
    ("Please create me a futuristic image for a quantum computing studio",
     "a futuristic image for a quantum computing studio"),
    ("Generate image: neon cyberpunk alley, rainy night",
     "neon cyberpunk alley, rainy night"),
]


def test_stripper_removes_only_command_wrapper():
    for source, expected in CASES:
        assert router.extract_image_subject(source) == expected
        assert router.is_image_generation(source)


def test_questions_and_bare_commands_do_not_generate():
    assert not router.is_image_generation("Can you make images?")
    assert not router.is_image_generation("Generate")


def test_enrichment_requires_complete_verbatim_original(monkeypatch):
    original = "girls playing volleyball on the beach"

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "cinematic volleyball at sunset"}}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return Response()

    import httpx
    monkeypatch.setattr(httpx, "Client", Client)
    result = router.enrich_image_prompt(original)
    assert result["prompt"] == original
    assert result["enriched"] is False
    assert result["reason"] == "verbatim_guard"


def test_brand_enrichment_uses_exact_object_first_typography_structure(monkeypatch):
    original = "Quantum Web Studios Inc headquarters at night"
    result = router.enrich_image_prompt(original)
    prompt = result["prompt"]
    assert prompt.startswith("A large front-facing architectural wall sign")
    assert '"Quantum Web Studios Inc"' in prompt
    assert prompt.count('"') == 2
    assert prompt.count("Quantum Web Studios Inc") == 1
    assert " headquarters at night" in prompt
    assert "professionally fabricated dimensional lettering" in prompt
    assert "refined brushed-metal or crisp matte finish" in prompt
    assert "occupying a substantial fraction of the frame" in prompt
    assert "and no other text, letters, numbers, or symbols anywhere" in prompt
    assert result["enriched"] is True
    assert result["reason"] == "brand_structure"


@pytest.mark.parametrize(
    "direction",
    [
        "engraved serif",
        "brushed gold lettering",
        "military stencil",
        "elegant thin uppercase",
        "dimensional brushed-metal letters",
    ],
)
def test_paid_brand_authored_typography_replaces_premium_default(direction):
    original = f'"Heroes United Foundation" on a lobby sign, {direction}, at dusk'
    prompt = router.build_paid_brand_prompt(original)
    assert direction in prompt
    assert "the Chairman's exact typography direction" in prompt
    assert "professionally fabricated dimensional lettering" not in prompt
    assert prompt.count("Heroes United Foundation") == 1
    assert "large front-facing" in prompt
    assert "no other text, letters, numbers, or symbols anywhere" in prompt


def test_non_text_enrichment_rejects_added_lettering_triggers(monkeypatch):
    original = "a cinematic portrait of a wizard"

    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": original + ", ancient glowing runes"}}

    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def post(self, *args, **kwargs): return Response()

    import httpx
    monkeypatch.setattr(httpx, "Client", Client)
    result = router.enrich_image_prompt(original)
    assert result["reason"] == "lettering_trigger_guard"
    assert "runes" not in result["prompt"]
    assert "ancient" not in result["prompt"]
    assert "blank smooth unmarked visible surfaces" in result["prompt"]


def test_fast_and_quality_step_constants_are_distinct():
    assert router.IMAGE_GEN_STEPS == 4
    assert router.IMAGE_GEN_QUALITY_STEPS == 16


def test_quality_step_count_reaches_flux2_scheduler(tmp_path, monkeypatch):
    output = tmp_path / "quality.png"
    output.write_bytes(b"png")
    submitted = {}

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):
            submitted.update(json)
            return Response({"prompt_id": "pid"})

        def get(self, url):
            return Response({
                "pid": {
                    "status": {"status_str": "success"},
                    "outputs": {"7": {"images": [{"filename": output.name}]}},
                }
            })

    import httpx
    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(router, "COMFY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(router, "comfy_is_up", lambda: True)
    monkeypatch.setattr(router, "_unload_ollama_models", lambda: None)
    result = router.run_image_generation(
        "a detailed scene", steps=router.IMAGE_GEN_QUALITY_STEPS
    )
    assert result["path"] == str(output)
    assert submitted["prompt"]["13"]["inputs"]["steps"] == 16


def test_resolution_and_seed_reach_workflow(tmp_path, monkeypatch):
    output = tmp_path / "resolution.png"
    output.write_bytes(b"png")
    submitted = {}

    class Response:
        def __init__(self, payload): self.payload = payload
        def raise_for_status(self): pass
        def json(self): return self.payload

    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def post(self, url, json):
            submitted.update(json); return Response({"prompt_id": "pid"})
        def get(self, url):
            return Response({"pid": {"status": {"status_str": "success"},
                "outputs": {"7": {"images": [{"filename": output.name}]}}}})

    import httpx
    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(router, "COMFY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(router, "comfy_is_up", lambda: True)
    result = router.run_image_generation(
        "brand sign", steps=4, width=1280, height=1280, seed=424242
    )
    assert submitted["prompt"]["4"]["inputs"]["width"] == 1280
    assert submitted["prompt"]["4"]["inputs"]["height"] == 1280
    assert submitted["prompt"]["11"]["inputs"]["noise_seed"] == 424242
    assert result["width"] == 1280 and result["height"] == 1280


def test_router_log_records_both_prompts(tmp_path, monkeypatch):
    telemetry = tmp_path / "router.jsonl"
    monkeypatch.setattr(router, "TELEMETRY_PATH", str(telemetry))
    router.log_decision(
        mode="home",
        rule_fired="image_gen",
        model=router.IMAGE_GEN_MODEL,
        original_prompt="owl in a library",
        enriched_prompt="owl in a library, cinematic moonlight",
        enrichment_seconds=1.25,
        preset="quality",
    )
    record = json.loads(telemetry.read_text())
    assert record["original_prompt"] == "owl in a library"
    assert record["enriched_prompt"].startswith(record["original_prompt"])
    assert record["enrichment_seconds"] == 1.25
    assert record["preset"] == "quality"
