"""Tests for provider-specific OpenAI request body extensions."""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import yaml

from openevolve.config import Config, LLMConfig, LLMModelConfig
from openevolve.llm.openai import OpenAILLM


class TestExtraBodyConfig(unittest.TestCase):
    @staticmethod
    async def _run_inline(_executor, callback):
        """Run the executor callback inline so this unit test does not need worker threads."""
        return callback()

    def test_yaml_global_value_is_inherited_and_model_value_overrides_it(self):
        global_extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        model_extra_body = {"provider_option": "model-specific"}

        config = Config.from_dict(
            yaml.safe_load(
                """
llm:
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
  models:
    - name: inherits-global
    - name: overrides-global
      extra_body:
        provider_option: model-specific
  evaluator_models:
    - name: evaluator
"""
            )
        )

        self.assertEqual(config.llm.models[0].extra_body, global_extra_body)
        self.assertEqual(config.llm.models[1].extra_body, model_extra_body)
        self.assertEqual(config.llm.evaluator_models[0].extra_body, global_extra_body)

    def test_global_value_survives_model_rebuild(self):
        extra_body = {"provider_option": True}
        config = LLMConfig(primary_model="test-model", extra_body=extra_body)

        config.rebuild_models()

        self.assertEqual(config.models[0].extra_body, extra_body)
        self.assertEqual(config.evaluator_models[0].extra_body, extra_body)

    @patch("openevolve.llm.openai.openai.OpenAI")
    def test_configured_value_is_forwarded_to_openai(self, mock_openai):
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        llm = OpenAILLM(
            LLMModelConfig(name="test-model", extra_body=extra_body, retries=0)
        )
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "response"
        llm.client.chat.completions.create.return_value = response
        loop = Mock()
        loop.run_in_executor = AsyncMock(side_effect=self._run_inline)

        with patch("openevolve.llm.openai.asyncio.get_event_loop", return_value=loop):
            result = asyncio.run(llm.generate("prompt"))

        self.assertEqual(result, "response")
        params = mock_openai.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(params["extra_body"], extra_body)

    @patch("openevolve.llm.openai.openai.OpenAI")
    def test_unconfigured_value_is_omitted_from_openai_request(self, mock_openai):
        llm = OpenAILLM(LLMModelConfig(name="test-model", retries=0))
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "response"
        llm.client.chat.completions.create.return_value = response
        loop = Mock()
        loop.run_in_executor = AsyncMock(side_effect=self._run_inline)

        with patch("openevolve.llm.openai.asyncio.get_event_loop", return_value=loop):
            asyncio.run(llm.generate("prompt"))

        params = mock_openai.return_value.chat.completions.create.call_args.kwargs
        self.assertNotIn("extra_body", params)


if __name__ == "__main__":
    unittest.main()
