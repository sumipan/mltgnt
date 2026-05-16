import importlib.metadata

import mltgnt


def test_version_matches_metadata():
    assert mltgnt.__version__ == importlib.metadata.version("mltgnt")


def test_version_is_string():
    assert isinstance(mltgnt.__version__, str)
