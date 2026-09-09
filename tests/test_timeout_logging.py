"""Focused tests for domain-specific timeout logging."""

from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

import openai

from openevolve.config import EvaluatorConfig, LLMModelConfig
from openevolve.evaluator import Evaluator
from openevolve.llm.openai import OpenAILLM


class LLMTimeoutLoggingTests(IsolatedAsyncioTestCase):
    @staticmethod
    def _make_llm(*, retries: int) -> OpenAILLM:
        """Construct the request wrapper without creating a network client."""

        with patch("openevolve.llm.openai.openai.OpenAI"):
            return OpenAILLM(
                LLMModelConfig(
                    name="test-model",
                    provider="openai",
                    api_base="https://example.invalid/v1",
                    api_key="unused",
                    system_message="test system message",
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=32,
                    timeout=0.01,
                    retries=retries,
                    retry_delay=0,
                )
            )

    async def test_async_llm_timeout_logs_domain_duration_model_and_retries(self) -> None:
        llm = self._make_llm(retries=1)
        llm._call_api = AsyncMock(side_effect=asyncio.TimeoutError)

        with self.assertLogs("openevolve.llm.openai", level="WARNING") as captured:
            with self.assertRaises(asyncio.TimeoutError):
                await llm.generate("test prompt")

        self.assertEqual(llm._call_api.await_count, 2)
        messages = "\n".join(captured.output)
        self.assertIn("LLM timeout", messages)
        self.assertIn("model='test-model'", messages)
        self.assertIn("provider='openai'", messages)
        self.assertIn("configured timeout=0.01 s", messages)
        self.assertIn("attempt=1/2; retrying after 0 s", messages)
        self.assertIn("attempt=2/2; no retries remaining", messages)

    async def test_provider_timeout_uses_the_same_unambiguous_log(self) -> None:
        llm = self._make_llm(retries=0)
        timeout = openai.APITimeoutError(request=Mock())
        llm._call_api = AsyncMock(side_effect=timeout)

        with self.assertLogs("openevolve.llm.openai", level="ERROR") as captured:
            with self.assertRaises(openai.APITimeoutError):
                await llm.generate("test prompt")

        self.assertEqual(llm._call_api.await_count, 1)
        self.assertIn("LLM timeout", captured.output[0])
        self.assertIn("configured timeout=0.01 s", captured.output[0])
        self.assertIn("attempt=1/1; no retries remaining", captured.output[0])


class EvaluatorTimeoutLoggingTests(IsolatedAsyncioTestCase):
    @staticmethod
    def _make_evaluator() -> Evaluator:
        """Construct the evaluator shell without importing a fixture module."""

        evaluator = Evaluator.__new__(Evaluator)
        evaluator.config = EvaluatorConfig(
            timeout=0.02,
            max_retries=1,
            cascade_evaluation=False,
        )
        evaluator.program_suffix = ".py"
        evaluator.llm_ensemble = None
        evaluator._pending_artifacts = {}
        return evaluator

    async def test_evaluator_timeout_logs_identity_duration_and_no_retry(self) -> None:
        evaluator = self._make_evaluator()
        evaluator._direct_evaluate = AsyncMock(side_effect=asyncio.TimeoutError)

        with self.assertLogs("openevolve.evaluator", level="WARNING") as captured:
            result = await evaluator.evaluate_program("pass", "candidate-123")

        self.assertEqual(result, {"error": 0.0, "timeout": True})
        self.assertEqual(evaluator._direct_evaluate.await_count, 1)
        message = captured.output[0]
        self.assertIn("Evaluator timeout", message)
        self.assertIn("program_id=candidate-123", message)
        self.assertIn("candidate_path=", message)
        self.assertIn("configured timeout=0.02 s", message)
        self.assertIn("attempt=1/2", message)
        self.assertIn("evaluator timeouts are not retried", message)

    async def test_successful_evaluation_is_unchanged(self) -> None:
        evaluator = self._make_evaluator()
        evaluator._direct_evaluate = AsyncMock(return_value={"combined_score": 0.75})

        result = await evaluator.evaluate_program("pass", "candidate-ok")

        self.assertEqual(result, {"combined_score": 0.75})
        self.assertEqual(evaluator._direct_evaluate.await_count, 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
