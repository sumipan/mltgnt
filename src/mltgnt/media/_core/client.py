"""Re-export of the MediaClient contract defined in ``mltgnt.interfaces.media``."""

from __future__ import annotations

from mltgnt.interfaces.media import MediaClient, Status, adapt_client

__all__ = ["MediaClient", "Status", "adapt_client"]
