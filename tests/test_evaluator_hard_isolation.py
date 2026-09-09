"""Hard wall-clock isolation tests for candidate evaluator subprocesses."""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator


TEST_TIMEOUT = 1.5


EVALUATOR_SOURCE = r'''
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

from openevolve.evaluation_result import EvaluationResult


def _value(code, key):
    prefix = f"# {key}:"
    for line in code.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise ValueError(f"Missing {prefix}")


def _record_pid(code):
    Path(_value(code, "PID_FILE")).write_text(str(os.getpid()), encoding="utf-8")


def evaluate(program_path):
    code = Path(program_path).read_text(encoding="utf-8")

    if "# BEHAVIOR: infinite" in code:
        _record_pid(code)
        while True:
            pass

    if "# BEHAVIOR: sleep" in code:
        _record_pid(code)
        time.sleep(60)

    if "# BEHAVIOR: ignore-term" in code:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _record_pid(code)
        while True:
            pass

    if "# BEHAVIOR: process-tree" in code:
        child_pid_file = _value(code, "CHILD_PID_FILE")
        child_code = (
            "from pathlib import Path\n"
            "import os\n"
            "import time\n"
            f"Path({child_pid_file!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
            "while True:\n"
            "    pass\n"
        )
        subprocess.Popen([sys.executable, "-c", child_code])
        deadline = time.monotonic() + 5
        while not Path(child_pid_file).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        _record_pid(code)
        while True:
            pass

    if "# BEHAVIOR: handled-failure" in code:
        return EvaluationResult(
            metrics={"combined_score": -1.0, "runs_successfully": 0.0},
            artifacts={"failure.json": '{"error_type": "candidate"}\n'},
        )

    if "# BEHAVIOR: runtime-sys-path" in code:
        from isolation_runtime_dependency import VALUE
        return {"combined_score": VALUE, "runs_successfully": 1.0}

    if "# BEHAVIOR: artifacts" in code:
        return EvaluationResult(
            metrics={"combined_score": 0.75, "runs_successfully": 1.0},
            artifacts={
                "phenotype.json": '{\n  "lines": [1, 2, 3]\n}\n',
                "multiline.txt": "first line\nsecond line\n",
                "raw.bin": b"\x00\x01artifact\xff",
                "worker.pid": str(os.getpid()),
            },
        )

    return {"combined_score": 0.5, "runs_successfully": 1.0}
'''


def _run_timeout_then_success_in_pool(evaluation_file: str, pid_file: str):
    """Exercise nested subprocess isolation inside a process-pool worker."""

    async def run():
        config = EvaluatorConfig(
            timeout=TEST_TIMEOUT,
            max_retries=1,
            cascade_evaluation=False,
        )
        evaluator = Evaluator(config, evaluation_file)
        timed_out = await evaluator.evaluate_program(
            f"# BEHAVIOR: infinite\n# PID_FILE: {pid_file}\n",
            "pool-timeout",
        )
        success = await evaluator.evaluate_program("# BEHAVIOR: success\n", "pool-success")
        return timed_out, success

    return asyncio.run(run())


class EvaluatorHardIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="openevolve-hard-isolation-"))
        self.evaluation_file = self.temp_dir / "evaluator.py"
        self.evaluation_file.write_text(EVALUATOR_SOURCE, encoding="utf-8")
        self.config = EvaluatorConfig(
            timeout=TEST_TIMEOUT,
            max_retries=1,
            cascade_evaluation=False,
        )
        self.evaluator = Evaluator(self.config, str(self.evaluation_file))
        self.recorded_pids: set[int] = set()

    async def asyncTearDown(self) -> None:
        for pid in self.recorded_pids:
            if not self._pid_is_gone(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def _pid_is_gone(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    async def _read_pid(self, path: Path) -> int:
        deadline = time.monotonic() + 5.0
        while not path.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        self.assertTrue(path.exists(), f"PID marker was not written: {path}")
        pid = int(path.read_text(encoding="utf-8"))
        self.recorded_pids.add(pid)
        return pid

    async def _assert_pid_gone(self, pid: int) -> None:
        deadline = time.monotonic() + 2.0
        while not self._pid_is_gone(pid) and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        self.assertTrue(self._pid_is_gone(pid), f"process {pid} survived isolation cleanup")

    async def _timeout(self, behavior: str, pid_file: Path, program_id: str):
        started = time.monotonic()
        result = await self.evaluator.evaluate_program(
            f"# BEHAVIOR: {behavior}\n# PID_FILE: {pid_file}\n",
            program_id,
        )
        elapsed = time.monotonic() - started
        self.assertEqual(result, {"error": 0.0, "timeout": True})
        self.assertLess(elapsed, TEST_TIMEOUT + 1.0)
        pid = await self._read_pid(pid_file)
        await self._assert_pid_gone(pid)
        return pid

    async def test_success_preserves_metrics_and_multiformat_artifacts(self) -> None:
        metrics = await self.evaluator.evaluate_program(
            "# BEHAVIOR: artifacts\n",
            "artifact-success",
        )
        artifacts = self.evaluator.get_pending_artifacts("artifact-success")
        worker_pid = int(artifacts.pop("worker.pid"))
        self.recorded_pids.add(worker_pid)

        self.assertEqual(
            metrics,
            {"combined_score": 0.75, "runs_successfully": 1.0},
        )
        self.assertEqual(
            artifacts,
            {
                "phenotype.json": '{\n  "lines": [1, 2, 3]\n}\n',
                "multiline.txt": "first line\nsecond line\n",
                "raw.bin": b"\x00\x01artifact\xff",
            },
        )
        self.assertEqual(json.loads(artifacts["phenotype.json"])["lines"], [1, 2, 3])
        await self._assert_pid_gone(worker_pid)

    async def test_handled_failure_result_crosses_boundary_unchanged(self) -> None:
        metrics = await self.evaluator.evaluate_program(
            "# BEHAVIOR: handled-failure\n",
            "handled-failure",
        )
        artifacts = self.evaluator.get_pending_artifacts("handled-failure")

        self.assertEqual(
            metrics,
            {"combined_score": -1.0, "runs_successfully": 0.0},
        )
        self.assertEqual(
            artifacts,
            {"failure.json": '{"error_type": "candidate"}\n'},
        )

    async def test_runtime_parent_sys_path_is_inherited(self) -> None:
        import_dir = self.temp_dir / "runtime-import"
        import_dir.mkdir()
        (import_dir / "isolation_runtime_dependency.py").write_text(
            "VALUE = 0.625\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(import_dir))
        try:
            result = await self.evaluator.evaluate_program(
                "# BEHAVIOR: runtime-sys-path\n",
                "runtime-sys-path",
            )
        finally:
            sys.path.remove(str(import_dir))

        self.assertEqual(
            result,
            {"combined_score": 0.625, "runs_successfully": 1.0},
        )

    async def test_infinite_loop_is_killed_at_timeout(self) -> None:
        await self._timeout(
            "infinite",
            self.temp_dir / "infinite.pid",
            "infinite-loop",
        )

    async def test_sleeping_evaluator_is_killed_at_timeout(self) -> None:
        await self._timeout(
            "sleep",
            self.temp_dir / "sleep.pid",
            "sleeping-evaluator",
        )

    @unittest.skipUnless(os.name == "posix", "SIGKILL escalation test requires POSIX")
    async def test_sigkill_escalation_stops_process_that_ignores_sigterm(self) -> None:
        await self._timeout(
            "ignore-term",
            self.temp_dir / "ignore-term.pid",
            "ignore-term",
        )

    @unittest.skipUnless(sys.platform == "linux", "parent-death signal requires Linux")
    async def test_supervisor_death_kills_inflight_evaluator(self) -> None:
        pid_file = self.temp_dir / "parent-death.pid"
        candidate = f"# BEHAVIOR: infinite\n# PID_FILE: {pid_file}\n"
        supervisor_script = f"""
import asyncio
from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator

evaluator = Evaluator(
    EvaluatorConfig(timeout=60, max_retries=0, cascade_evaluation=False),
    {str(self.evaluation_file)!r},
)
asyncio.run(evaluator.evaluate_program(
    {candidate!r},
    "parent-death",
))
"""
        supervisor = subprocess.Popen(
            [sys.executable, "-c", supervisor_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "TMPDIR": str(self.temp_dir)},
        )
        try:
            evaluator_pid = await self._read_pid(pid_file)
            supervisor.kill()
            supervisor.wait(timeout=3)
            await self._assert_pid_gone(evaluator_pid)
        finally:
            if supervisor.poll() is None:
                supervisor.kill()
                supervisor.wait(timeout=3)

    async def test_repeated_timeouts_leave_no_processes_and_allow_later_success(self) -> None:
        pids = []
        for index in range(3):
            pids.append(
                await self._timeout(
                    "infinite",
                    self.temp_dir / f"repeat-{index}.pid",
                    f"repeat-{index}",
                )
            )

        self.assertEqual(len(set(pids)), 3)
        success = await self.evaluator.evaluate_program(
            "# BEHAVIOR: success\n",
            "after-timeouts",
        )
        self.assertEqual(
            success,
            {"combined_score": 0.5, "runs_successfully": 1.0},
        )

    @unittest.skipUnless(os.name == "posix", "process-group test requires POSIX")
    async def test_timeout_kills_spawned_descendant_process(self) -> None:
        parent_pid_file = self.temp_dir / "tree-parent.pid"
        child_pid_file = self.temp_dir / "tree-child.pid"
        started = time.monotonic()
        result = await self.evaluator.evaluate_program(
            "\n".join(
                (
                    "# BEHAVIOR: process-tree",
                    f"# PID_FILE: {parent_pid_file}",
                    f"# CHILD_PID_FILE: {child_pid_file}",
                    "",
                )
            ),
            "process-tree",
        )

        self.assertEqual(result, {"error": 0.0, "timeout": True})
        self.assertLess(time.monotonic() - started, TEST_TIMEOUT + 1.0)
        parent_pid = await self._read_pid(parent_pid_file)
        child_pid = await self._read_pid(child_pid_file)
        await self._assert_pid_gone(parent_pid)
        await self._assert_pid_gone(child_pid)

    async def test_timeout_artifacts_and_logging_remain_unambiguous(self) -> None:
        pid_file = self.temp_dir / "logging.pid"
        with self.assertLogs("openevolve.evaluator", level="WARNING") as captured:
            await self._timeout("infinite", pid_file, "logged-candidate")

        artifacts = self.evaluator.get_pending_artifacts("logged-candidate")
        self.assertEqual(
            artifacts,
            {
                "timeout": True,
                "timeout_duration": TEST_TIMEOUT,
                "failure_stage": "evaluation",
                "error_type": "timeout",
            },
        )
        message = "\n".join(captured.output)
        self.assertIn("Evaluator timeout", message)
        self.assertIn("program_id=logged-candidate", message)
        self.assertIn(f"configured timeout={TEST_TIMEOUT} s", message)
        self.assertIn("evaluator timeouts are not retried", message)

    async def test_process_pool_worker_survives_timeout_and_evaluates_next_candidate(self) -> None:
        pid_file = self.temp_dir / "pool-timeout.pid"
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=1) as executor:
            timed_out, success = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    _run_timeout_then_success_in_pool,
                    str(self.evaluation_file),
                    str(pid_file),
                ),
                timeout=10,
            )

        self.assertEqual(timed_out, {"error": 0.0, "timeout": True})
        self.assertEqual(
            success,
            {"combined_score": 0.5, "runs_successfully": 1.0},
        )
        pid = await self._read_pid(pid_file)
        await self._assert_pid_gone(pid)


if __name__ == "__main__":
    unittest.main()
