"""The spoken consent step for cloned voices: a sentence with a random code, checked by our own STT.

``start`` makes a challenge (speaker name, basis, a 4-digit code, the sentence to read) that lives in
memory for 15 minutes and can be used once. ``verify`` reads a transcript of the speaker's recording
and passes only when it has the code (digits or number words, any spacing) and the words "agree" and
"voice". Names are not compared: speech-to-text often spells them differently, and the code already
proves the sentence was read for this challenge.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from typing import Any

CHALLENGE_TTL_S = 15 * 60
_MAX_CHALLENGES = 32
BASES = ("own", "permission")
_CHALLENGES: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()

_UNITS = {
    "zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "for": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def sentence_for(name: str, code: str) -> str:
    spaced = " ".join(code)
    return f"I, {name}, agree that my voice may be copied by AI on this computer. My code is {spaced}."


def _clean_name(name: str) -> str:
    return " ".join(str(name).split())[:60]


def start(speaker_name: str, basis: str) -> dict[str, Any]:
    name = _clean_name(speaker_name)
    if not name:
        raise ValueError("Type the speaker's name.")
    if basis not in BASES:
        raise ValueError("Choose whose voice this is.")
    code = f"{secrets.randbelow(9000) + 1000}"
    challenge = {
        "id": secrets.token_hex(8),
        "speakerName": name,
        "basis": basis,
        "code": code,
        "sentence": sentence_for(name, code),
        "createdAt": time.time(),
    }
    with _LOCK:
        _expire()
        while len(_CHALLENGES) >= _MAX_CHALLENGES:
            _CHALLENGES.pop(next(iter(_CHALLENGES)))
        _CHALLENGES[challenge["id"]] = challenge
    return dict(challenge)


def _expire() -> None:
    now = time.time()
    for key in [k for k, v in _CHALLENGES.items() if now - v["createdAt"] > CHALLENGE_TTL_S]:
        _CHALLENGES.pop(key, None)


def peek(challenge_id: str) -> dict[str, Any] | None:
    with _LOCK:
        _expire()
        found = _CHALLENGES.get(str(challenge_id))
        return dict(found) if found else None


def consume(challenge_id: str) -> None:
    with _LOCK:
        _CHALLENGES.pop(str(challenge_id), None)


def spoken_digits(text: str) -> str:
    """Every number in ``text`` as one digit string: "4 7 two one", "forty seven twenty one" -> "4721"."""
    out: list[str] = []
    tokens = re.findall(r"[a-z]+|\d+", text.lower().replace("-", " "))
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isdigit():
            out.append(token)
        elif token in _TENS:
            value = _TENS[token]
            if i + 1 < len(tokens) and tokens[i + 1] in _UNITS and 0 < _UNITS[tokens[i + 1]] < 10:
                value += _UNITS[tokens[i + 1]]
                i += 1
            out.append(str(value))
        elif token in _UNITS and token not in ("o", "oh", "for") or (token in ("o", "oh", "for") and out):
            out.append(str(_UNITS[token]))
        i += 1
    return "".join(out)


def verify(transcript: str, code: str) -> str | None:
    """``None`` when the recording reads the sentence; otherwise a plain reason to show."""
    words = set(re.findall(r"[a-z]+", transcript.lower()))
    if code not in spoken_digits(transcript):
        return (
            f"The code {' '.join(code)} was not heard in the recording. "
            "Record the sentence again, reading the code one digit at a time."
        )
    if not ({"agree", "agreed", "agrees"} & words) or not ({"voice", "voices"} & words):
        return "The words of the consent sentence were not heard. Record the whole sentence again, clearly."
    return None
