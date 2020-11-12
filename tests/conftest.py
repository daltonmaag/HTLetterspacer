from pathlib import Path

import pytest


@pytest.fixture
def datadir(request) -> Path:
    return Path(__file__).parent / "data"
