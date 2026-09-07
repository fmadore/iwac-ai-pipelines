"""Generate synthetic text/PDF/audio inputs and preview a validated text write.

No credentials, network calls or inference. Outputs stay in the chosen directory.
"""
import argparse
import json
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pypdf import PdfWriter
from common.artifacts import commit_artifact, read_artifact
from common.checkpoint import sha256_file, atomic_write_text
from common.omeka_text_updater import PropertyTarget, TextUpdate, run_text_updates
from common.write_guard import WriteGuard


class PreviewArchive:
    base_url = "https://example.invalid/api"

    def get_item(self, item_id):
        return {"o:id": item_id, "dcterms:title": [{"type": "literal", "@value": "Synthetic example"}]}

    def update_item(self, *args):
        raise AssertionError("The example must never write to an archive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source.txt"
    atomic_write_text(source, "Un document synthétique pour vérifier le transfert des résultats.\n")
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with (out / "blank-page.pdf").open("wb") as handle:
        writer.write(handle)
    with wave.open(str(out / "silence.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 16000)
    artifact = out / "1.txt"
    atomic_write_text(artifact, "Résultat synthétique, sans appel à un modèle.")
    commit_artifact(artifact, context={"pipeline": "offline-example-v1", "model_key": "synthetic-no-inference"},
                    source_sha256=sha256_file(source))
    read_artifact(artifact)
    stats = run_text_updates(PreviewArchive(), [TextUpdate("1", 1, artifact.read_text(encoding="utf-8"))],
                             PropertyTarget("bibo:content", 91), guard=WriteGuard(dry_run=True))
    atomic_write_text(out / "preview.json", json.dumps(stats, indent=2) + "\n")
    assert stats["would_update"] == 1 and stats["updated"] == 0


if __name__ == "__main__":
    main()
