"""Tests for the language module: SentenceBuilder + gloss_to_text."""

from __future__ import annotations

from app.language.gloss_to_text import gloss_sequence_to_text
from app.language.sentence_builder import SentenceBuilder


def test_add_gloss_appends_and_suppresses_immediate_duplicates():
    builder = SentenceBuilder()
    assert builder.add_gloss("HELLO") is True
    assert builder.add_gloss("HELLO") is False  # duplicate suppression
    assert builder.add_gloss("YOU") is True
    assert builder.gloss_sequence == ["HELLO", "YOU"]


def test_add_gloss_ignores_unknown_label():
    builder = SentenceBuilder(unknown_label="UNKNOWN")
    assert builder.add_gloss("UNKNOWN") is False
    assert builder.gloss_sequence == []


def test_clear_resets_sequence():
    builder = SentenceBuilder()
    builder.add_gloss("HELLO")
    builder.clear()
    assert builder.gloss_sequence == []
    assert builder.get_text() == ""


def test_known_phrase_template_hello_how_you():
    assert gloss_sequence_to_text(["HELLO", "HOW", "YOU"]) == "Hello, how are you?"


def test_known_phrase_template_thank_you():
    assert gloss_sequence_to_text(["THANK_YOU"]) == "Thank you."


def test_pronoun_object_verb_reorders_to_svo():
    # ISL-typical Subject-Object-Verb -> English Subject-Verb-Object
    text = gloss_sequence_to_text(["I", "WATER", "WANT"])
    assert text == "I want water."


def test_unrecognized_sequence_falls_back_to_literal_join():
    text = gloss_sequence_to_text(["GOOD", "FOOD"])
    assert text == "Good food."


def test_empty_sequence_returns_empty_string():
    assert gloss_sequence_to_text([]) == ""
    assert gloss_sequence_to_text(["UNKNOWN"]) == ""


def test_sentence_builder_end_to_end_via_add_gloss():
    builder = SentenceBuilder()
    for token in ["HELLO", "HOW", "YOU"]:
        builder.add_gloss(token)
    assert builder.get_text() == "Hello, how are you?"
