"""Vocabulary and lightweight word-class metadata used by the sentence
builder's grammar rules.

The class list itself is fully configurable (``settings.vocabulary`` or a
trained model's ``classes.json``). The word-class tags below are only used
to make gloss -> text conversion a bit more natural (e.g. reordering an
ISL-typical Subject-Object-Verb gloss into an English Subject-Verb-Object
sentence). They are best-effort linguistic hints, not a claim of full
grammatical analysis — see docs/architecture.md limitations section.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List

from app.config import settings


class WordClass(str, Enum):
    PRONOUN = "pronoun"
    VERB = "verb"
    NOUN = "noun"
    GREETING = "greeting"
    POLITE = "polite"
    RESPONSE = "response"
    DESCRIPTOR = "descriptor"
    QUESTION = "question"
    OTHER = "other"


# Best-effort classification of the default demo vocabulary (Section 13).
# Unrecognized tokens (e.g. from a custom/trained vocabulary) default to
# WordClass.OTHER and are simply passed through untouched.
_DEFAULT_WORD_CLASSES: Dict[str, WordClass] = {
    "I": WordClass.PRONOUN,
    "YOU": WordClass.PRONOUN,
    "WE": WordClass.PRONOUN,
    "GO": WordClass.VERB,
    "COME": WordClass.VERB,
    "STOP": WordClass.VERB,
    "HELP": WordClass.VERB,
    "WANT": WordClass.VERB,
    "WATER": WordClass.NOUN,
    "FOOD": WordClass.NOUN,
    "NAME": WordClass.NOUN,
    "HELLO": WordClass.GREETING,
    "WELCOME": WordClass.GREETING,
    "THANK_YOU": WordClass.POLITE,
    "THANK YOU": WordClass.POLITE,
    "PLEASE": WordClass.POLITE,
    "SORRY": WordClass.POLITE,
    "YES": WordClass.RESPONSE,
    "NO": WordClass.RESPONSE,
    "GOOD": WordClass.DESCRIPTOR,
    "BAD": WordClass.DESCRIPTOR,
    "HOW": WordClass.QUESTION,
}

# Human-readable surface form for glosses that use underscores/uppercase
# internally (e.g. model class "THANK_YOU" -> displayed/spoken "thank you").
_DISPLAY_OVERRIDES: Dict[str, str] = {
    "THANK_YOU": "thank you",
    "I": "I",
}


def get_vocabulary() -> List[str]:
    return list(settings.vocabulary)


def word_class(token: str) -> WordClass:
    return _DEFAULT_WORD_CLASSES.get(token.upper(), WordClass.OTHER)


def surface_form(token: str) -> str:
    """Lowercase, human-readable form of a gloss token, e.g. THANK_YOU -> 'thank you'."""
    if token in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[token]
    return token.replace("_", " ").lower()
