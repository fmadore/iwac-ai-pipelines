"""Prompt discovery and the menu that a `--prompt N` flag answers."""

import io

import pytest
from rich.console import Console

from common.prompt_loader import discover_prompts, load_prompt_md, select_prompt_interactive

DEFAULT = "default prompt"


@pytest.fixture
def prompts(tmp_path):
    (tmp_path / "2_full_audio_translation.md").write_text("# Translation\n\nTranslate.\n", encoding="utf-8")
    (tmp_path / "1_full_audio_transcription.md").write_text(
        "# Transcription\n\nTranscribe every word.\n\nKeep speakers apart.\n", encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text("no header here", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("not a prompt", encoding="utf-8")
    return tmp_path


def quiet(answers=()):
    console = Console(file=io.StringIO())
    replies = iter(answers)
    console.input = lambda *_args, **_kwargs: next(replies)
    return console


def test_prompts_are_ordered_by_number_and_described_from_the_filename(prompts):
    options = discover_prompts(prompts)
    assert [(o.number, o.description) for o in options] == [
        (0, "Notes"),
        (1, "Full Audio Transcription"),
        (2, "Full Audio Translation"),
    ]


def test_a_missing_folder_yields_no_prompts(tmp_path):
    assert discover_prompts(tmp_path / "absent") == []


def test_the_title_line_is_not_part_of_the_prompt(prompts):
    assert load_prompt_md(prompts / "1_full_audio_transcription.md") == (
        "Transcribe every word.\nKeep speakers apart."
    )
    assert load_prompt_md(prompts / "notes.md") == "no header here"


def test_preselected_number_answers_the_menu_without_stdin(prompts):
    content, number = select_prompt_interactive(
        prompts, quiet(), default_prompt=DEFAULT, preselected=2,
    )
    assert (content, number) == ("Translate.", 2)


def test_preselected_zero_means_the_default_prompt(prompts):
    assert select_prompt_interactive(
        prompts, quiet(), default_prompt=DEFAULT, preselected=0,
    ) == (DEFAULT, None)


def test_a_preselected_number_that_does_not_exist_is_an_error(prompts):
    with pytest.raises(ValueError, match="--prompt 9"):
        select_prompt_interactive(prompts, quiet(), default_prompt=DEFAULT, preselected=9)


def test_the_menu_reasks_until_the_answer_names_a_prompt(prompts):
    content, number = select_prompt_interactive(
        prompts, quiet(["abc", "7", "1"]), default_prompt=DEFAULT,
    )
    assert number == 1 and content.startswith("Transcribe")


def test_enter_and_an_empty_folder_both_fall_back_to_the_default(prompts, tmp_path):
    assert select_prompt_interactive(prompts, quiet([""]), default_prompt=DEFAULT) == (DEFAULT, None)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert select_prompt_interactive(empty, quiet(), default_prompt=DEFAULT) == (DEFAULT, None)
