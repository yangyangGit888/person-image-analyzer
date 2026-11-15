"""pytest 公共 fixture。"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.analysis_service import AnalysisService
from tests.helpers import FakeProvider, single_person_result


@contextmanager
def _build_client(
    provider: FakeProvider | None = None,
    **overrides: Any,
) -> Iterator[tuple[TestClient, FakeProvider]]:
    config: dict[str, Any] = {
        "ai_api_key": "test-key",
        "ai_enable_thinking": False,
        "ai_timeout": 5,
        "ai_max_retries": 1,
        "ai_retry_backoff": 0,
    }
    config.update(overrides)
    settings = Settings(_env_file=None, **config)
    fake_provider = provider or FakeProvider(single_person_result())

    app = create_app(settings)
    with TestClient(app) as client:
        app.state.ai_provider = fake_provider
        app.state.analysis_service = AnalysisService(fake_provider, settings)
        yield client, fake_provider


@pytest.fixture
def make_client() -> Any:
    """返回一个可自定义 Settings / FakeProvider 的客户端工厂。"""

    return _build_client


@pytest.fixture
def client() -> Iterator[TestClient]:
    with _build_client() as (test_client, _):
        yield test_client
