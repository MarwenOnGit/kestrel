"""Structural guarantees (Doc 08 §7).

These do not test behaviour - they test that the ARCHITECTURE cannot be
subverted by a future change. Both are build-breaking.

  1. Airborne code has no capability to decrypt its own recordings.
  2. There is exactly one write path from IQ to the vault, and it goes
     through the warrant gate.
"""
from __future__ import annotations

import ast
import os
import pathlib

AIR = pathlib.Path(__file__).resolve().parent.parent / "kestrel" / "air"


def _air_sources() -> list[tuple[pathlib.Path, str]]:
    return [(p, p.read_text()) for p in AIR.rglob("*.py")]


def _strip_comments_and_docstrings(src: str) -> str:
    """Only executable code counts - a comment mentioning a forbidden name is
    documentation, not capability."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            node.value.value = ""
    return ast.unparse(tree)


def test_air_code_cannot_decrypt_its_own_recordings():
    forbidden = ("unseal_with", "SealedBox", "PrivateKey", "seal_sk",
                 "decrypt", "generate_seal_keypair")
    offenders = []
    for path, src in _air_sources():
        code = _strip_comments_and_docstrings(src)
        for name in forbidden:
            if name in code:
                offenders.append(f"{path.name}: {name}")
    assert not offenders, (
        "Airborne code must not be able to read its own take (Doc 08 §4). "
        f"Found: {offenders}")


def test_only_sealer_writes_to_the_vault():
    """Any other module opening a file for writing under air/ is suspect."""
    offenders = []
    for path, src in _air_sources():
        if path.name == "sealer.py":
            continue
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open":
                mode = ""
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = str(node.args[1].value)
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = str(kw.value.value)
                if any(m in mode for m in ("w", "a", "x")):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "Only kestrel/air/sealer.py may write recordings, so that every write "
        f"passes the warrant gate. Found writers at: {offenders}")


def test_collection_controller_is_the_only_caller_of_seal_recording():
    offenders = []
    for path, src in _air_sources():
        if path.name == "sealer.py":
            continue
        if "seal_recording" in _strip_comments_and_docstrings(src):
            offenders.append(path.name)
    assert not offenders, (
        f"seal_recording() must only be called via CollectionController: {offenders}")


def test_warrant_gate_is_deterministic():
    """The gate must stay deterministic and auditable line by line: no models,
    no randomness. Checked over the IMPORT GRAPH, not the text, so that prose
    describing the rule does not trip it."""
    banned = {"torch", "tensorrt", "sklearn", "random", "numpy", "np"}
    tree = ast.parse((AIR / "warrant_gate.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    offenders = imported & banned
    assert not offenders, f"warrant_gate.py must not import {offenders}"


def test_air_code_never_imports_lab_tooling():
    """kestrel/lab/ decodes protocol content from public recordings. That is
    validation tooling, and it must never become a capability of the aircraft:
    content handling in the air is the warrant's business, not a library's."""
    offenders = []
    for path, src in _air_sources():
        for node in ast.walk(ast.parse(src)):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            if any(m.startswith("kestrel.lab") for m in mods):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"kestrel/air must not import kestrel.lab: {offenders}"
