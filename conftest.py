"""Root pytest configuration for miniLLM-engine.

Sandbox workaround
------------------
When running under the DSH file sandbox on Windows, directories created with
POSIX ``mode=0o700`` are translated into a Windows ACL that denies directory
listing *even to the creating process*. pytest creates every temporary
directory with ``mode=0o700`` (``tmp_path``, ``tmp_path_factory``), which makes
those fixtures unusable.

This module patches ``pathlib.Path.mkdir`` so that ``mode=0o700`` is rewritten
to ``mode=0o777`` (sandbox-friendly) while the DSH sandbox is active. The patch
is disabled in any other environment, where ``0o700`` keeps its normal meaning.
"""

from __future__ import annotations

import os
import pathlib
import sys


if sys.platform == "win32" and os.environ.get("DSH_SESSION_ID"):
    _orig_mkdir = pathlib.Path.mkdir

    def _sandbox_safe_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if mode == 0o700:
            mode = 0o777
        return _orig_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    pathlib.Path.mkdir = _sandbox_safe_mkdir
