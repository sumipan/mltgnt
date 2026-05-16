import importlib.metadata
from unittest.mock import patch

import mltgnt


def test_version_matches_metadata():
    assert mltgnt.__version__ == importlib.metadata.version("mltgnt")


def test_version_is_string():
    assert isinstance(mltgnt.__version__, str)


def test_version_fallback_when_package_not_found():
    with patch(
        "importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError("mltgnt"),
    ):
        import importlib as _il

        _il.reload(mltgnt)
        assert mltgnt.__version__ == "0.0.0"
    _il.reload(mltgnt)
