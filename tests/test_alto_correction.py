"""ALTO correction must not pass a failed request off as a corrected file.

Every model error used to be caught per block and the original tokens kept, so
the file was written to ALTO_Corrected/, counted as a success, and the run
reported "All N files processed successfully" with output identical to its
input.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "alto_correction", REPO_ROOT / "AI_ocr_correction" / "02_correct_alto_xml.py"
)
alto = importlib.util.module_from_spec(_spec)
sys.modules["alto_correction"] = alto
_spec.loader.exec_module(alto)

from common.llm_provider import TruncatedOutputError  # noqa: E402

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
ALTO_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="{ALTO_NS}">
  <Layout><Page ID="P1"><PrintSpace>
    <TextBlock ID="B1">
      <TextLine ID="L1">
        <String ID="S1" CONTENT="Ouagadougou," HPOS="10" VPOS="10" WIDTH="80" HEIGHT="12"/>
        <String ID="S2" CONTENT="Ie" HPOS="95" VPOS="10" WIDTH="12" HEIGHT="12"/>
        <String ID="S3" CONTENT="imam" HPOS="110" VPOS="10" WIDTH="40" HEIGHT="12"/>
      </TextLine>
    </TextBlock>
  </PrintSpace></Page></Layout>
</alto>
"""


class _Client:
    def __init__(self, answer=None, error=None):
        self.answer, self.error = answer, error

    def generate_structured(self, system_prompt, user_prompt, schema):
        if self.error is not None:
            raise self.error
        return self.answer


class _QuotaError(Exception):
    status_code = 402


def _alto_dir(tmp_path):
    source = tmp_path / "ALTO"
    source.mkdir()
    (source / "page.xml").write_text(ALTO_XML, encoding="utf-8")
    return source, tmp_path / "ALTO_Corrected"


def test_a_failed_request_fails_the_file_and_writes_nothing(tmp_path):
    source, out = _alto_dir(tmp_path)
    client = _Client(error=TruncatedOutputError("Gemini", "gemini-3.7-flash", "max_tokens"))

    success, errors, *_ = alto.process_alto_files(client, source, out, "prompt")

    assert (success, errors) == (0, 1)
    assert not (out / "page.xml").exists()


def test_quota_exhaustion_stops_the_run(tmp_path):
    source, out = _alto_dir(tmp_path)
    (source / "page2.xml").write_text(ALTO_XML, encoding="utf-8")
    client = _Client(error=_QuotaError("402 Insufficient credits"))

    success, errors, *_ = alto.process_alto_files(client, source, out, "prompt")

    assert (success, errors) == (0, 2)
    assert not out.exists() or not any(out.iterdir())


def test_a_corrected_file_keeps_coordinates_and_the_default_namespace(tmp_path):
    source, out = _alto_dir(tmp_path)
    answer = alto.LineCorrectionBatch(lines=[alto.CorrectedLine(
        line_index=0,
        original_tokens=["Ouagadougou,", "Ie", "imam"],
        corrected_tokens=["Ouagadougou,", "le", "imam"],
    )])

    success, errors, _, _, updated = alto.process_alto_files(_Client(answer=answer), source, out, "prompt")

    assert (success, errors, updated) == (1, 0, 1)
    written = (out / "page.xml").read_text(encoding="utf-8")
    assert 'CONTENT="le"' in written and 'HPOS="95"' in written
    assert "ns0:" not in written
