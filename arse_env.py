"""
Accept the pre-rename environment variables.

The project was renamed from ARES to ARSE, and every ``ARES_*`` setting
became ``ARSE_*``.  Servers still carry ``ARES_*`` lines in their ``.env``;
importing this module copies each of them to its ``ARSE_*`` name unless
that is already set, so an old ``.env`` keeps working.  Imported first by
every module that reads an ``ARSE_*`` setting.
"""

import os

for _k, _v in list(os.environ.items()):
    if _k.startswith("ARES_"):
        os.environ.setdefault("ARSE_" + _k[len("ARES_"):], _v)
