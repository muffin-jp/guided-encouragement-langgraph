"""Startup wiring: retrieval fails open, and /healthz says so.

pgvector is the only retrieval store, so a missing DATABASE_URL or a failed
embedder load must neither crash startup nor hide: the app serves ungrounded
replies and /healthz reports ``"retriever": "failed"``. No DB, no model — the
embedder is the offline stub and PgVectorRetriever is never reached.
"""

from __future__ import annotations

import logging

import pytest

import app.main as main
import app.rag.embedder as embedder_mod
from tests.test_retriever import StubEmbedder

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def quiet_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the classifier dark and the stub embedder in place, so only the
    # retrieval wiring under test varies.
    monkeypatch.setattr(main, "CLASSIFIER_ENABLED", False)
    monkeypatch.setattr(embedder_mod, "load_embedder", lambda: StubEmbedder())
    monkeypatch.delenv("RAG_BACKEND", raising=False)


async def _startup_health() -> dict[str, bool | str]:
    async with main.lifespan(main.app):
        return await main.healthz()


async def test_rag_off_reports_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "RAG_ENABLED", False)
    health = await _startup_health()
    assert health["retriever"] == "disabled"
    assert main.app.state.retriever is None


async def test_missing_database_url_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "RAG_ENABLED", True)
    monkeypatch.setattr(main, "DATABASE_URL", "")
    health = await _startup_health()
    assert health["ok"] is True
    assert health["retriever"] == "failed"
    assert main.app.state.retriever is None


async def test_embedder_failure_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> StubEmbedder:
        raise FileNotFoundError("weights not vendored")

    monkeypatch.setattr(main, "RAG_ENABLED", True)
    monkeypatch.setattr(main, "DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(embedder_mod, "load_embedder", _boom)
    health = await _startup_health()
    assert health["retriever"] == "failed"
    assert main.app.state.retriever is None


async def test_legacy_rag_backend_is_warned_about(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(main, "RAG_ENABLED", False)
    monkeypatch.setenv("RAG_BACKEND", "memory")
    with caplog.at_level(logging.WARNING, logger="bloom"):
        await _startup_health()
    assert any("RAG_BACKEND=memory is no longer supported" in r.message for r in caplog.records)
