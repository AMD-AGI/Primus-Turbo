"""Vendored copy of aiter's gfx1250 FlyDSL forward prefill kernel (m32x8).

Five files copied verbatim from /home/lihuzhan/code/aiter-src on 2026-09-23, with
only their import lines rewritten so the package is self-contained: it depends on
flydsl, torch and the stdlib, and on nothing in the aiter package. See
../PROVENANCE.md for the file list and the exact edits.

Import this package through impl.py's `_sibling_pkg()`, never with a plain
`import flydsl_fwd` -- see the docstring there.
"""
