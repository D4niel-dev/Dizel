"""Regression tests for the ChatManager backend lifecycle.

These use a fake backend so they exercise the swap/close rules without loading
gigabytes of weights. The invariants they protect:

* loading B closes A only after B is ready;
* a failed load leaves A loaded and closes the half-built B;
* load and generation are mutually exclusive;
* unload releases the backend.
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dizel_core import model_backend
from dizel_core.chat_manager import ChatManager


class FakeBackend:
    instances = []
    fail_names = set()

    def __init__(self):
        self.path = ""
        self.closed = False
        self.ready = False
        self._device = "cpu"
        FakeBackend.instances.append(self)

    def load(self, checkpoint_path, device, on_progress, **kwargs):
        if os.path.basename(checkpoint_path) in FakeBackend.fail_names:
            raise RuntimeError("synthetic load failure")
        self.path = checkpoint_path
        self.ready = True
        self._device = device
        on_progress("fake loaded")

    @property
    def is_ready(self):
        return self.ready

    def close(self):
        self.closed = True
        self.ready = False

    def stop(self):
        pass

    def count_tokens(self, text):
        return len(text.split())

    @property
    def context_length(self):
        return 2048

    @property
    def model_info(self):
        return {"device": self._device}


def _manager():
    manager = ChatManager()
    # The tests must not depend on whatever provider is saved on this machine.
    manager._provider_slug = "local"
    manager._api_provider = None
    return manager


def _checkpoints(directory):
    a = os.path.join(directory, "a.gguf")
    b = os.path.join(directory, "b.gguf")
    broken = os.path.join(directory, "broken.gguf")
    for path in (a, b, broken):
        with open(path, "wb") as handle:
            handle.write(b"fake")
    return a, b, broken


def test_load_a_then_b_closes_a():
    with tempfile.TemporaryDirectory() as directory:
        a, b, _ = _checkpoints(directory)
        manager = _manager()
        manager.load_model(a, "cpu")
        first = manager._backend
        assert first is not None and first.path == a
        assert manager.is_ready and manager.checkpoint_path == a

        manager.load_model(b, "cpu")
        second = manager._backend
        assert second is not first, "a new backend must be swapped in"
        assert second.path == b and not second.closed
        assert first.closed, "the replaced backend must be released"
        assert manager.checkpoint_path == b


def test_failed_load_keeps_previous_model():
    with tempfile.TemporaryDirectory() as directory:
        a, b, broken = _checkpoints(directory)
        manager = _manager()
        manager.load_model(b, "cpu")
        previous = manager._backend

        FakeBackend.fail_names = {os.path.basename(broken)}
        try:
            raised = False
            try:
                manager.load_model(broken, "cpu")
            except RuntimeError as error:
                raised = "synthetic" in str(error)
            assert raised, "a failed load must raise"
        finally:
            FakeBackend.fail_names = set()

        assert manager._backend is previous, "the old backend must stay installed"
        assert not previous.closed
        assert manager.checkpoint_path == b
        assert manager.is_ready
        failed = FakeBackend.instances[-1]
        assert failed.closed, "the half-built backend must be released"


def test_failed_first_load_leaves_manager_empty():
    with tempfile.TemporaryDirectory() as directory:
        _, _, broken = _checkpoints(directory)
        manager = _manager()
        FakeBackend.fail_names = {os.path.basename(broken)}
        try:
            try:
                manager.load_model(broken, "cpu")
            except RuntimeError:
                pass
            else:
                raise AssertionError("a failed load must raise")
        finally:
            FakeBackend.fail_names = set()

        assert manager._backend is None
        assert manager.checkpoint_path == ""
        assert not manager.is_ready
        assert FakeBackend.instances[-1].closed


def test_load_refused_while_generating():
    with tempfile.TemporaryDirectory() as directory:
        a, b, _ = _checkpoints(directory)
        manager = _manager()
        manager.load_model(a, "cpu")
        loaded = manager._backend

        manager._is_generating = True
        try:
            try:
                manager.load_model(b, "cpu")
            except RuntimeError as error:
                assert "generat" in str(error).lower()
            else:
                raise AssertionError("load during generation must be refused")
        finally:
            manager._is_generating = False

        assert manager._backend is loaded
        assert not loaded.closed
        assert manager.checkpoint_path == a


def test_generation_refused_while_loading():
    with tempfile.TemporaryDirectory() as directory:
        a, _, _ = _checkpoints(directory)
        manager = _manager()
        manager.load_model(a, "cpu")

        errors = []
        manager._load_in_progress = True
        try:
            manager.send_message(
                "hello",
                [],
                on_token=lambda token: None,
                on_done=lambda text: None,
                on_error=errors.append,
            )
        finally:
            manager._load_in_progress = False

        assert errors, "generation must report an error while a load is running"
        assert "loading" in errors[0].lower()
        assert not manager.is_generating, "no generation thread may be started"


def test_unload_releases_backend():
    with tempfile.TemporaryDirectory() as directory:
        a, _, _ = _checkpoints(directory)
        manager = _manager()
        manager.load_model(a, "cpu")
        loaded = manager._backend

        manager.unload_model()
        assert manager._backend is None
        assert loaded.closed
        assert manager.checkpoint_path == ""
        assert not manager.is_ready


def test_unload_refused_while_generating():
    with tempfile.TemporaryDirectory() as directory:
        a, _, _ = _checkpoints(directory)
        manager = _manager()
        manager.load_model(a, "cpu")
        loaded = manager._backend

        manager._is_generating = True
        try:
            try:
                manager.unload_model()
            except RuntimeError as error:
                assert "generat" in str(error).lower()
            else:
                raise AssertionError("unload during generation must be refused")
        finally:
            manager._is_generating = False

        assert manager._backend is loaded
        assert not loaded.closed


def main():
    original = model_backend.GGUFBackend
    model_backend.GGUFBackend = FakeBackend
    try:
        test_load_a_then_b_closes_a()
        test_failed_load_keeps_previous_model()
        test_failed_first_load_leaves_manager_empty()
        test_load_refused_while_generating()
        test_generation_refused_while_loading()
        test_unload_releases_backend()
        test_unload_refused_while_generating()
    finally:
        model_backend.GGUFBackend = original
    print("model lifecycle tests passed")


if __name__ == "__main__":
    main()
