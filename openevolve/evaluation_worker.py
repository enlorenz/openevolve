"""Private subprocess entry point for isolated evaluator calls.

This module intentionally uses only the standard library.  The supervising
OpenEvolve process owns timeout enforcement and process-tree cleanup; this
worker only imports one evaluator function, invokes it, and writes a pickle
envelope to a private result file.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import pickle
from pathlib import Path
import signal
import sys
import traceback
from typing import Any


PROTOCOL_VERSION = 1
PARENT_SYS_PATH_ENV = "OPENEVOLVE_EVALUATION_PARENT_SYS_PATH"
_PR_SET_PDEATHSIG = 1


def _arm_linux_parent_death_signal(expected_parent_pid: int) -> None:
    """Ensure Linux kills this worker if its supervising process disappears."""

    if sys.platform != "linux":
        return

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))

    # The parent may have exited between spawning us and the prctl call.
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def _inherit_parent_sys_path() -> None:
    """Restore runtime import paths that an exec boundary cannot inherit."""

    serialized_path = os.environ.get(PARENT_SYS_PATH_ENV)
    if not serialized_path:
        return
    try:
        parent_path = json.loads(serialized_path)
    except (TypeError, ValueError):
        return
    if not isinstance(parent_path, list) or not all(
        isinstance(entry, str) for entry in parent_path
    ):
        return

    sys.path[:] = parent_path + [entry for entry in sys.path if entry not in parent_path]


def _load_evaluator_function(evaluation_file: str, function_name: str) -> Any:
    """Load a named evaluator function from an absolute source-file path."""

    evaluation_path = Path(evaluation_file).resolve()
    evaluation_dir = str(evaluation_path.parent)
    if evaluation_dir not in sys.path:
        sys.path.insert(0, evaluation_dir)

    spec = importlib.util.spec_from_file_location("evaluation_module", evaluation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {evaluation_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluation_module"] = module
    spec.loader.exec_module(module)

    function = getattr(module, function_name, None)
    if not callable(function):
        raise AttributeError(
            f"Evaluation file {evaluation_path} does not contain callable {function_name!r}"
        )
    return function


def _error_envelope(exc: BaseException) -> dict[str, Any]:
    """Build an always-pickleable description of a remote exception."""

    try:
        serialized_exception = pickle.dumps(exc, protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException:
        serialized_exception = None

    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "error",
        "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "serialized_exception": serialized_exception,
    }


def _serialize_result(result: Any) -> bytes:
    """Serialize an exact evaluator return value or a safe protocol error."""

    envelope = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "ok",
        "result": result,
    }
    try:
        return pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException as exc:
        serialization_error = RuntimeError(
            f"Evaluator result could not cross the process boundary: {exc}"
        )
        return pickle.dumps(
            _error_envelope(serialization_error),
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _write_envelope(result_path: str, envelope: dict[str, Any] | bytes) -> None:
    """Write one complete envelope after all potentially failing serialization."""

    data = (
        envelope
        if isinstance(envelope, bytes)
        else pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    )
    with open(result_path, "wb") as result_file:
        result_file.write(data)
        result_file.flush()
        os.fsync(result_file.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("evaluation_file")
    parser.add_argument("function_name")
    parser.add_argument("program_path")
    parser.add_argument("result_path")
    parser.add_argument("parent_pid", type=int)
    args = parser.parse_args()

    try:
        _arm_linux_parent_death_signal(args.parent_pid)
        _inherit_parent_sys_path()
        evaluator_function = _load_evaluator_function(
            args.evaluation_file,
            args.function_name,
        )
        result = evaluator_function(args.program_path)
    except BaseException as exc:
        envelope: dict[str, Any] | bytes = _error_envelope(exc)
    else:
        envelope = _serialize_result(result)

    _write_envelope(args.result_path, envelope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
