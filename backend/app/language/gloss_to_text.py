"""Gloss sequence -> natural-language text.

IMPORTANT (see docs/architecture.md "Limitations"): this is a small,
rule-based/template convenience layer, NOT a linguistic machine-translation
system. ISL grammar (topic-comment structure, non-manual markers such as
facial expression/eyebrow raises, spatial referencing) is far richer than
what a fixed template set can capture. This module only handles the
demo-scale vocabulary well; anything outside it falls back to a literal,
punctuated join of the gloss words.
"""

from __future__ import annotations

from typing import List, Tuple

from app.language.vocabulary import WordClass, surface_form, word_class

# Exact-match templates for common short gloss sequences. Checked before any
# grammar heuristic. Keys are upper-case gloss tuples.
_PHRASE_TEMPLATES: dict[Tuple[str, ...], str] = {
    ("HELLO",): "Hello.",
    ("HELLO", "HOW", "YOU"): "Hello, how are you?",
    ("HOW", "YOU"): "How are you?",
    ("THANK_YOU",): "Thank you.",
    ("THANK", "YOU"): "Thank you.",
    ("PLEASE",): "Please.",
    ("SORRY",): "Sorry.",
    ("YES",): "Yes.",
    ("NO",): "No.",
    ("GOOD",): "Good.",
    ("BAD",): "Bad.",
    ("HELP",): "Help.",
    ("HELP", "ME"): "Help me.",
    ("STOP",): "Stop.",
    ("WELCOME",): "You're welcome.",
    ("I", "NAME"): "My name is...",
}


def _apply_grammar_heuristics(tokens: List[str]) -> str:
    """Best-effort reordering for the common ISL Subject-Object-Verb
    pattern (e.g. "I WATER WANT") into an English Subject-Verb-Object
    sentence ("I want water."). Falls back to a literal join when the
    pattern doesn't match a recognized [pronoun, noun, verb] shape.
    """
    classes = [word_class(t) for t in tokens]

    if len(tokens) == 3 and classes == [WordClass.PRONOUN, WordClass.NOUN, WordClass.VERB]:
        pronoun, noun, verb = tokens
        return f"{surface_form(pronoun).capitalize()} {surface_form(verb)} {surface_form(noun)}."

    if len(tokens) == 2 and classes == [WordClass.PRONOUN, WordClass.VERB]:
        pronoun, verb = tokens
        return f"{surface_form(pronoun).capitalize()} {surface_form(verb)}."

    # Literal fallback: join with spaces, capitalize, punctuate.
    words = [surface_form(t) for t in tokens]
    sentence = " ".join(words)
    sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
    if any(word_class(t) == WordClass.QUESTION for t in tokens):
        return sentence + "?"
    return sentence + "."


def gloss_sequence_to_text(gloss_tokens: List[str], unknown_label: str = "UNKNOWN") -> str:
    """Convert an ordered list of committed gloss tokens into a natural
    sentence. UNKNOWN tokens are dropped before conversion — they represent
    "no confident sign detected", not part of the message.
    """
    tokens = [t.upper() for t in gloss_tokens if t.upper() != unknown_label.upper()]
    if not tokens:
        return ""

    key = tuple(tokens)
    if key in _PHRASE_TEMPLATES:
        return _PHRASE_TEMPLATES[key]

    return _apply_grammar_heuristics(tokens)
