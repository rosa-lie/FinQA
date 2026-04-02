#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible DPO entrypoint.

This wrapper keeps the historical CLI name and options while delegating
execution to the new unified router.
"""

from __future__ import annotations

import sys

from financial_data_processors.router import run_cli


def main() -> None:
    print("[compat] fin_to_dpo_pairs.py is routed to financial_data_router.py --task dpo", file=sys.stderr)
    run_cli(default_task="dpo")


if __name__ == "__main__":
    main()
