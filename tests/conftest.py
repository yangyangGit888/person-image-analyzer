"""pytest 公共 fixture。"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.analysis_service import AnalysisService
from tests.helpers import FakeProvider, RoutingFakeProvider, single_person_result


@contextmanager
def _build_client(
    provider: FakeProvider | None = None,
    **overrides: Any,
) -> Iterator[tuple[TestClient, FakeProvider]]:
    with tempfile.TemporaryDirectory() as storage_dir:
        config: dict[str, Any] = {
            "ai_api_key": "test-key",
            "ai_enable_thinking": False,
            "ai_timeout": 5,
            "ai_max_retries": 1,
            "ai_retry_backoff": 0,
            "storage_root": storage_dir,
            "segmentation_provider": "mock",
            "image_edit_provider": "mock",
            "image_edit_poll_initial": 0,
            "image_edit_poll_max": 0,
            "image_edit_timeout": 5,
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


@contextmanager
def _build_edit_client(
    *,
    person_result: dict[str, Any] | None = None,
    region_result: dict[str, Any] | None = None,
    **overrides: Any,
) -> Iterator[tuple[TestClient, RoutingFakeProvider]]:
    from app.services.edit_analysis_service import EditAnalysisService
    from app.services.region_detection_service import RegionDetectionService
    from app.services.segmentation_service import SegmentationService
    with tempfile.TemporaryDirectory() as storage_dir:
        config: dict[str, Any] = {
            "ai_api_key": "test-key",
            "ai_enable_thinking": False,
            "ai_timeout": 5,
            "ai_max_retries": 1,
            "ai_retry_backoff": 0,
            "storage_root": storage_dir,
            "segmentation_provider": "mock",
            "image_edit_provider": "mock",
            "image_edit_poll_initial": 0,
            "image_edit_poll_max": 0,
            "image_edit_timeout": 5,
        }
        config.update(overrides)
        settings = Settings(_env_file=None, **config)
        fake_provider = RoutingFakeProvider(
            person_result=person_result,
            region_result=region_result,
        )
        app = create_app(settings)
        with TestClient(app) as client:
            analysis_service = AnalysisService(fake_provider, settings)
            app.state.ai_provider = fake_provider
            app.state.analysis_service = analysis_service
            region_detection = RegionDetectionService(fake_provider)
            segmentation = SegmentationService(app.state.segmentation_provider)
            app.state.edit_analysis_service = EditAnalysisService(
                settings,
                app.state.edit_repository,
                app.state.storage_service,
                app.state.edit_image_service,
                analysis_service,
                region_detection,
                segmentation,
                app.state.mask_service,
            )
            yield client, fake_provider


@pytest.fixture
def make_edit_client() -> Any:
    return _build_edit_client


@pytest.fixture
def edit_client() -> Iterator[TestClient]:
    with _build_edit_client() as (test_client, _):
        yield test_client
