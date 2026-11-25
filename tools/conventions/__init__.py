###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Repository conventions as executable rules (``python tools/conventions``).

A convention that only a reviewer enforces is enforced now and then. Each rule here is a
function that reads the tree and reports what breaks it, with an ID, the reason for the rule
and how to fix a finding (``--explain <ID>``); pre-commit runs them all, locally and in CI.

  * ``core.py``: the tree, findings, the baseline, the runner and the command line.
  * ``rules/``: one module per group of rules; a module in that directory is picked up by
    itself. Add a rule with ``@rule("<ID>", "<title>")`` on a function ``(tree) -> findings``
    whose docstring says why the rule exists and how to fix a finding.
  * ``baseline.txt``: findings that predate a rule. A rule declared ``baseline=True`` passes on
    the files listed there and fails on any other; a listed file that no longer breaks the rule
    must be taken off the list, so the list only shrinks.
"""
