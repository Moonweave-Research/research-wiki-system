import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legacy.graphrag.wiki_entity_alias import (  # noqa: E402
    ALIAS_MAP,
    canonicalize_entity,
    reset_alias_cache,
)


def test_canonicalize_known_alias_lce():
    assert canonicalize_entity("LCE") == "liquid crystal elastomer"
    assert canonicalize_entity("lces") == "liquid crystal elastomer"
    assert canonicalize_entity("Liquid Crystalline Elastomer") == "liquid crystal elastomer"


def test_canonicalize_pedot_pss_variants():
    canonical = "pedot:pss"
    assert canonicalize_entity("PEDOT:PSS") == canonical
    assert canonicalize_entity("PEDOT-PSS") == canonical
    assert canonicalize_entity("pedot pss") == canonical
    assert (
        canonicalize_entity("Poly(3,4-ethylenedioxythiophene):poly(styrene sulfonate)") == canonical
    )


def test_canonicalize_carbon_nanotube_family():
    assert canonicalize_entity("CNT") == "carbon nanotube"
    assert canonicalize_entity("MWCNTs") == "carbon nanotube"
    # SWCNT is separate (different material)
    assert canonicalize_entity("SWCNT") == "single-walled carbon nanotube"
    assert canonicalize_entity("SWCNTs") == "single-walled carbon nanotube"


def test_canonicalize_pdms_acronym():
    assert canonicalize_entity("PDMS") == "polydimethylsiloxane"
    assert canonicalize_entity("Poly(dimethylsiloxane)") == "polydimethylsiloxane"


def test_canonicalize_falls_through_unknown_entity():
    # Unknown entity stays as lowercase + whitespace-collapsed form
    assert canonicalize_entity("Novel material XYZ-42") == "novel material xyz-42"
    assert canonicalize_entity("  spaces   collapsed   ") == "spaces collapsed"


def test_canonicalize_preserves_chemistry_punctuation():
    # Even when not in the alias map, colon/hyphen/dot/slash survive
    assert canonicalize_entity("4-cyano-4'-pentylbiphenyl") == "4-cyano-4'-pentylbiphenyl"
    assert canonicalize_entity("cs.AI/0501001") == "cs.ai/0501001"


def test_canonicalize_case_insensitive():
    assert canonicalize_entity("DEA") == "dielectric elastomer actuator"
    assert canonicalize_entity("dea") == "dielectric elastomer actuator"
    assert canonicalize_entity("Dea") == "dielectric elastomer actuator"


def test_canonicalize_idempotent():
    # Canonical form should map to itself
    for canonical in ALIAS_MAP.keys():
        assert canonicalize_entity(canonical) == canonical


def test_alias_cache_resets_for_test_isolation():
    # Validates the reset hook used by extending the map at test time.
    reset_alias_cache()
    assert canonicalize_entity("LCE") == "liquid crystal elastomer"


def test_ppr_normalize_entity_uses_alias_map():
    """The PPR module's normalize_entity should now route through alias map."""
    from legacy.graphrag.wiki_ppr import normalize_entity

    assert normalize_entity("LCE") == "liquid crystal elastomer"
    assert normalize_entity("PEDOT-PSS") == "pedot:pss"
    # Falls through to lowercase for unknown
    assert normalize_entity("Some Novel Polymer") == "some novel polymer"
