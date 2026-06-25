from unittest.mock import MagicMock, patch

import yaml

from scripts.ingest.arxiv_ingest import (
    ATOM_NS,
    ARXIV_NS,
    build_frontmatter,
    build_note_body,
    is_mdpi,
    main,
    parse_atom_xml,
    safe_arxiv_id,
    write_note,
)
from legacy.graphrag.wiki_graph import infer_kind


MINIMAL_ATOM_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="{ATOM_NS}" xmlns:arxiv="{ARXIV_NS}">
  <entry>
    <title>Attention Is All You Need</title>
    <summary>We propose a new network architecture.</summary>
    <published>2017-06-12T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:doi>10.1234/test.001</arxiv:doi>
    <link title="pdf" href="https://arxiv.org/pdf/1706.03762" rel="related"/>
  </entry>
</feed>""".encode()


def _make_fake_urlopen(xml_bytes: bytes):
    def fake_urlopen(req, timeout=None):
        resp = MagicMock()
        resp.read.return_value = xml_bytes
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    return fake_urlopen


def test_safe_id_basic():
    safe_id, version = safe_arxiv_id("2401.04088")
    assert safe_id == "arxiv_2401_04088"
    assert version == ""


def test_safe_id_versioned():
    safe_id, version = safe_arxiv_id("2401.04088v2")
    assert safe_id == "arxiv_2401_04088"
    assert version == "v2"


def test_safe_id_legacy_prefix():
    safe_id, version = safe_arxiv_id("cs.AI/0501001")
    assert "/" not in safe_id
    assert safe_id == "arxiv_cs_AI_0501001"
    assert version == ""


def test_parse_atom_minimal():
    meta = parse_atom_xml(MINIMAL_ATOM_XML)
    assert meta["title"] == "Attention Is All You Need"
    assert "Ashish Vaswani" in meta["authors"]
    assert "Noam Shazeer" in meta["authors"]
    assert "We propose" in meta["abstract"]
    assert meta["year"] == "2017"
    assert meta["primary_category"] == "cs.CL"
    assert meta["doi"] == "10.1234/test.001"
    assert meta["pdf_url"] == "https://arxiv.org/pdf/1706.03762"


def test_frontmatter_kind_paper(tmp_path):
    vault = tmp_path
    research_dir = vault / "wiki" / "research"
    research_dir.mkdir(parents=True)

    meta = {
        "title": "Test Paper Title",
        "authors": ["Author One"],
        "abstract": "This is a test abstract.",
        "year": "2024",
        "primary_category": "cond-mat.soft",
        "doi": "",
        "pdf_url": "",
    }
    safe_id = "arxiv_2401_00001"
    fm = build_frontmatter(
        raw_id="2401.00001",
        safe_id=safe_id,
        version="",
        meta=meta,
        pdf_path="",
        domain="research",
        category="arxiv",
    )
    body = build_note_body(meta)
    note_path = research_dir / f"{safe_id}.md"
    write_note(note_path, fm, body)

    wiki_root = vault / "wiki"
    rel_path = note_path.relative_to(wiki_root)

    kind = infer_kind(rel_path, fm, body)
    assert kind == "Paper"


def test_idempotent_merge(tmp_path):
    vault = tmp_path
    research_dir = vault / "wiki" / "research"
    research_dir.mkdir(parents=True)
    log_path = vault / "wiki" / "log.md"
    log_path.write_text("# Log\n", encoding="utf-8")

    with patch("urllib.request.urlopen", side_effect=_make_fake_urlopen(MINIMAL_ATOM_XML)):
        with patch("time.sleep"):
            rc = main(
                [
                    "--ids",
                    "1706.03762",
                    "--vault",
                    str(vault),
                    "--no-download",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                ]
            )
    assert rc == 0

    note_path = research_dir / "arxiv_1706_03762.md"
    assert note_path.exists()

    # Manually set status to "reading" to verify preserve-status behaviour
    text = note_path.read_text(encoding="utf-8")
    note_path.write_text(text.replace("status: unread", "status: reading"), encoding="utf-8")

    with patch("urllib.request.urlopen", side_effect=_make_fake_urlopen(MINIMAL_ATOM_XML)):
        with patch("time.sleep"):
            rc2 = main(
                [
                    "--ids",
                    "1706.03762",
                    "--vault",
                    str(vault),
                    "--no-download",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--merge",
                ]
            )
    assert rc2 == 0

    final_text = note_path.read_text(encoding="utf-8")
    import re

    fm_match = re.match(r"^---\n(.*?)\n---", final_text, re.DOTALL)
    assert fm_match
    final_fm = yaml.safe_load(fm_match.group(1))
    assert final_fm["status"] == "reading"

    # Tags should not be duplicated
    tags = final_fm["tags"]
    assert len(tags) == len(set(tags))


def test_log_appended(tmp_path):
    vault = tmp_path
    research_dir = vault / "wiki" / "research"
    research_dir.mkdir(parents=True)
    log_path = vault / "wiki" / "log.md"
    log_path.write_text("# Log\n", encoding="utf-8")

    initial_content = log_path.read_text(encoding="utf-8")

    with patch("urllib.request.urlopen", side_effect=_make_fake_urlopen(MINIMAL_ATOM_XML)):
        with patch("time.sleep"):
            rc = main(
                [
                    "--ids",
                    "1706.03762",
                    "--vault",
                    str(vault),
                    "--no-download",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                ]
            )
    assert rc == 0

    final_content = log_path.read_text(encoding="utf-8")
    assert len(final_content) > len(initial_content)
    assert "arxiv_ingest" in final_content


def test_is_mdpi_doi_prefix():
    assert is_mdpi({"doi": "10.3390/polym15040897"}) is True
    assert is_mdpi({"doi": "10.3390/SENSORS22125000"}) is True


def test_is_mdpi_journal_ref_ignored():
    """journal_ref alone is no longer a signal — too many false positives.
    See test_is_mdpi_false_positive_substrings for the motivation.
    """
    # Even strings containing 'MDPI' or known MDPI journal names return False
    # without a 10.3390/ DOI, because journal_ref text is unreliable.
    assert is_mdpi({"doi": "", "journal_ref": "Polymers 2024, 16, 123"}) is False
    assert is_mdpi({"doi": "", "journal_ref": "MDPI Materials, vol 17"}) is False


def test_is_mdpi_false_positive_substrings():
    """Critical regression guard: major non-MDPI journals share substrings
    with MDPI titles ('Materials', 'Sensors', 'Polymer'). These must NOT
    be flagged as MDPI, or a materials-science vault silently loses papers.
    """
    # Elsevier, Wiley, ACS, Nature Publishing Group, IEEE — all non-MDPI:
    assert is_mdpi({"doi": "", "journal_ref": "Materials Today 50, 100-110"}) is False
    assert is_mdpi({"doi": "", "journal_ref": "Advanced Materials 35, e2300123"}) is False
    assert is_mdpi({"doi": "", "journal_ref": "Chemistry of Materials 32, 12345"}) is False
    assert is_mdpi({"doi": "", "journal_ref": "Nature Materials 22, 567-575"}) is False
    assert is_mdpi({"doi": "", "journal_ref": "Sensors and Actuators A 350, 114115"}) is False
    assert is_mdpi({"doi": "", "journal_ref": "Acta Materialia 215, 117121"}) is False
    assert is_mdpi({"doi": "", "journal_ref": "Polymer 280, 126102"}) is False


def test_is_mdpi_non_mdpi():
    assert is_mdpi({"doi": "10.1038/nature12345", "journal_ref": "Nature 600, 1-5"}) is False
    assert is_mdpi({"doi": "10.1126/science.abc1234", "journal_ref": ""}) is False
    assert is_mdpi({"doi": "", "journal_ref": ""}) is False
    assert is_mdpi({"doi": "", "journal_ref": "Phys. Rev. B 108, 045123"}) is False


MDPI_ATOM_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="{ATOM_NS}" xmlns:arxiv="{ARXIV_NS}">
  <entry>
    <title>An MDPI Polymer Paper</title>
    <summary>Polymer abstract.</summary>
    <published>2024-03-15T00:00:00Z</published>
    <author><name>Author One</name></author>
    <arxiv:primary_category term="cond-mat.soft" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:doi>10.3390/polym16060897</arxiv:doi>
    <arxiv:journal_ref>Polymers 2024, 16, 897</arxiv:journal_ref>
    <link title="pdf" href="https://arxiv.org/pdf/2403.00001" rel="related"/>
  </entry>
</feed>""".encode()


def test_filter_mdpi_default_skips(tmp_path):
    vault = tmp_path
    (vault / "wiki" / "research").mkdir(parents=True)
    (vault / "wiki" / "log.md").write_text("# Log\n", encoding="utf-8")

    with patch("urllib.request.urlopen", side_effect=_make_fake_urlopen(MDPI_ATOM_XML)):
        with patch("time.sleep"):
            rc = main(
                [
                    "--ids",
                    "2403.00001",
                    "--vault",
                    str(vault),
                    "--no-download",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                ]
            )
    assert rc == 0
    # Note must NOT be created.
    assert not (vault / "wiki" / "research" / "arxiv_2403_00001.md").exists()


def test_filter_mdpi_disabled_creates(tmp_path):
    vault = tmp_path
    (vault / "wiki" / "research").mkdir(parents=True)
    (vault / "wiki" / "log.md").write_text("# Log\n", encoding="utf-8")

    with patch("urllib.request.urlopen", side_effect=_make_fake_urlopen(MDPI_ATOM_XML)):
        with patch("time.sleep"):
            rc = main(
                [
                    "--ids",
                    "2403.00001",
                    "--vault",
                    str(vault),
                    "--no-download",
                    "--no-filter-mdpi",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                ]
            )
    assert rc == 0
    # Note IS created when filter disabled.
    assert (vault / "wiki" / "research" / "arxiv_2403_00001.md").exists()


def test_parse_atom_rejects_arxiv_error_entry():
    """arXiv signals 'ID not found' with a 200 + entry whose id points to
    api/errors. We must NOT return that as valid metadata."""
    import pytest

    bad = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="{ATOM_NS}" xmlns:arxiv="{ARXIV_NS}">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_9999.99999</id>
    <title>Error</title>
    <summary>incorrect id format</summary>
  </entry>
</feed>""".encode()
    with pytest.raises(ValueError, match="error entry"):
        parse_atom_xml(bad)


def test_pdf_download_failure_clears_pdf_path_and_downgrades(tmp_path):
    """Regression guard for the PDF cascade bug: when PDF download fails,
    the note must NOT advertise a phantom pdf_path. Downstream wiki_pdf_cache
    would otherwise record 'missing pdf' every refresh and the orchestrator
    would mis-classify this as a successful ingest.
    """
    vault = tmp_path
    research_dir = vault / "wiki" / "research"
    research_dir.mkdir(parents=True)
    (vault / "wiki" / "log.md").write_text("# Log\n", encoding="utf-8")

    call_count = {"n": 0}

    def selective_urlopen(req, timeout=None):
        # First call (metadata) → return Atom XML successfully.
        # Second call (PDF) → simulate network error.
        call_count["n"] += 1
        if call_count["n"] == 1:
            resp = MagicMock()
            resp.read.return_value = MINIMAL_ATOM_XML
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp
        raise TimeoutError("simulated PDF timeout")

    with patch("urllib.request.urlopen", side_effect=selective_urlopen):
        with patch("time.sleep"):
            rc = main(
                [
                    "--ids",
                    "1706.03762",
                    "--vault",
                    str(vault),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                ]
            )
    # Single ID, hard failure (PDF) ⇒ rc=1 (all-failed exit branch).
    assert rc == 1
    note_path = research_dir / "arxiv_1706_03762.md"
    assert note_path.exists()
    fm_text = note_path.read_text(encoding="utf-8")
    # pdf_path must NOT point to a non-existent file
    assert "pdf_path: ''" in fm_text or "pdf_path: ''\n" in fm_text or "pdf_path:\n" in fm_text


def test_merge_refuses_malformed_existing_frontmatter(tmp_path):
    """If the existing note has malformed YAML, --merge must NOT silently
    clobber it with fresh metadata. The user may have hand-edited the file.
    """
    vault = tmp_path
    research_dir = vault / "wiki" / "research"
    research_dir.mkdir(parents=True)
    (vault / "wiki" / "log.md").write_text("# Log\n", encoding="utf-8")
    bad_note = research_dir / "arxiv_1706_03762.md"
    bad_note.write_text(
        "---\nstatus: reading\n  malformed:\n: this is bad yaml\n---\nbody\n",
        encoding="utf-8",
    )
    original = bad_note.read_text(encoding="utf-8")

    with patch("urllib.request.urlopen", side_effect=_make_fake_urlopen(MINIMAL_ATOM_XML)):
        with patch("time.sleep"):
            main(
                [
                    "--ids",
                    "1706.03762",
                    "--vault",
                    str(vault),
                    "--no-download",
                    "--merge",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                ]
            )
    # File must be untouched.
    assert bad_note.read_text(encoding="utf-8") == original
