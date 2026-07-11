"""Regression test for the Supabase client retry wrapper (app/store/supabase.py).

Production incident (2026-07-11, live Render deploy): a single /run job
against Mall of America produced a 500 on /status --

    httpx.RemoteProtocolError: Server disconnected

raised from Store.get_job()'s self._client...execute() call. The
Supabase client's httpx.Client holds one persistent keep-alive
connection for the process lifetime; Supabase's edge proxy can close it
server-side without httpx noticing until the next reuse. _execute()
retries once on the same builder to recover from exactly this.
"""
from __future__ import annotations

import httpx
import pytest

from app.store.supabase import _execute


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FlakyBuilder:
    """Raises once (simulating a stale reused connection), then succeeds --
    execute() must be safely callable more than once on the same builder."""

    def __init__(self, fail_times: int, error: Exception, result=None):
        self.fail_times = fail_times
        self.error = error
        self.result = result
        self.call_count = 0

    def execute(self):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise self.error
        return FakeResponse(self.result)


def test_retries_once_on_remote_protocol_error():
    builder = FlakyBuilder(fail_times=1, error=httpx.RemoteProtocolError("Server disconnected"), result=[{"job_id": "abc"}])
    result = _execute(builder)
    assert result.data == [{"job_id": "abc"}]
    assert builder.call_count == 2


def test_succeeds_immediately_when_no_error():
    builder = FlakyBuilder(fail_times=0, error=httpx.RemoteProtocolError("unused"), result=[{"ok": True}])
    result = _execute(builder)
    assert result.data == [{"ok": True}]
    assert builder.call_count == 1


def test_does_not_retry_more_than_once():
    # a second consecutive failure is a real problem, not a stale-connection
    # blip -- it should propagate rather than retry forever
    builder = FlakyBuilder(fail_times=2, error=httpx.RemoteProtocolError("Server disconnected"))
    with pytest.raises(httpx.RemoteProtocolError):
        _execute(builder)
    assert builder.call_count == 2


@pytest.mark.parametrize("error", [
    httpx.RemoteProtocolError("Server disconnected"),
    httpx.ConnectError("connection refused"),
    httpx.ReadError("read failed"),
    httpx.WriteError("write failed"),
])
def test_retries_on_all_covered_transient_errors(error):
    builder = FlakyBuilder(fail_times=1, error=error, result=[{"ok": True}])
    result = _execute(builder)
    assert result.data == [{"ok": True}]


def test_does_not_retry_on_non_transient_error():
    # a non-connection error (e.g. a real API/validation error) should
    # propagate immediately, not be swallowed by a pointless retry
    class BoomBuilder:
        def __init__(self):
            self.call_count = 0

        def execute(self):
            self.call_count += 1
            raise ValueError("not a connection issue")

    builder = BoomBuilder()
    with pytest.raises(ValueError):
        _execute(builder)
    assert builder.call_count == 1
