"""Focused feature-gate tests for true-diversity prompt wiring."""

import asyncio
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from openevolve import process_parallel as process_parallel_module
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.evolution_trace import EvolutionTracer
from openevolve.process_parallel import ProcessParallelController, SerializableResult
from openevolve.prompt.sampler import PromptSampler


class TestTrueDiversityConfigurationAndPrompt(unittest.TestCase):
    @staticmethod
    def _program(program_id, score, code=None):
        return {
            "id": program_id,
            "code": code or f"def {program_id}(): return {score}",
            "metrics": {"combined_score": score, "latency": score * 10},
        }

    def test_missing_and_explicit_false_use_identical_legacy_prompt(self):
        missing = Config.from_dict({})
        explicit_false = Config.from_dict({"prompt": {"true_diversity": False}})
        self.assertFalse(missing.prompt.true_diversity)
        self.assertFalse(explicit_false.prompt.true_diversity)

        for config in (missing, explicit_false):
            config.prompt.num_top_programs = 1
            config.prompt.num_diverse_programs = 2
            config.prompt.include_artifacts = False

        top_programs = [
            self._program("top", 0.9),
            self._program("legacy_rank_2", 0.8),
            self._program("legacy_rank_3", 0.7),
        ]
        inspiration = self._program("legacy_inspiration", 0.4)
        build_args = {
            "current_program": "def current(): return 0",
            "parent_program": "def current(): return 0",
            "program_metrics": {"combined_score": 0.5},
            "top_programs": top_programs,
            "inspirations": [inspiration],
        }

        random.seed(17)
        missing_prompt = PromptSampler(missing.prompt).build_prompt(**build_args)["user"]
        random.seed(17)
        false_prompt = PromptSampler(explicit_false.prompt).build_prompt(**build_args)["user"]

        self.assertEqual(missing_prompt, false_prompt)
        self.assertIn("## Diverse Programs", false_prompt)
        self.assertIn("### Program 1 (Score: 0.9000)", false_prompt)
        self.assertIn("### Program D1 (Score:", false_prompt)
        self.assertIn("### Program D2 (Score:", false_prompt)
        self.assertIn("def legacy_rank_2()", false_prompt)
        self.assertIn("def legacy_rank_3()", false_prompt)
        self.assertIn("def legacy_inspiration()", false_prompt)
        legacy_inspiration_text = (
            "These programs represent diverse approaches and creative solutions "
            "that may inspire new ideas:"
        )
        self.assertIn(legacy_inspiration_text, missing_prompt)
        self.assertIn(legacy_inspiration_text, false_prompt)
        self.assertNotIn("not necessarily successful solutions", missing_prompt)
        self.assertNotIn("not necessarily successful solutions", false_prompt)

    def test_true_prompt_has_one_central_diverse_section_and_map_context(self):
        config = Config()
        config.prompt.true_diversity = True
        config.prompt.num_top_programs = 1
        config.prompt.num_diverse_programs = 2
        config.prompt.include_artifacts = False
        sampler = PromptSampler(config.prompt)

        top_programs = [
            self._program("top", 0.9),
            self._program("pseudo_diverse_1", 0.8),
            self._program("pseudo_diverse_2", 0.7),
        ]
        inspiration = self._program("map_owner", 0.4)
        inspiration["true_diversity_niche"] = {
            "map_cell": [2, 4],
            "descriptor_values": {"latency": 4.0},
            "source_island": 1,
        }

        user_prompt = sampler.build_prompt(
            current_program="def current(): return 0",
            parent_program="def current(): return 0",
            program_metrics={"combined_score": 0.5},
            top_programs=top_programs,
            inspirations=[inspiration],
            feature_dimensions=["latency"],
        )["user"]

        self.assertIn("def top()", user_prompt)
        self.assertNotIn("def pseudo_diverse_1()", user_prompt)
        self.assertNotIn("def pseudo_diverse_2()", user_prompt)
        self.assertNotIn("## Diverse Programs", user_prompt)
        self.assertEqual(user_prompt.count("## Inspiration Programs"), 1)
        self.assertEqual(user_prompt.count("def map_owner()"), 1)
        self.assertIn("### Program 1 (Score: 0.9000)", user_prompt)
        self.assertIn("### Diverse Inspiration 1 (Score: 0.4000)", user_prompt)
        clarification = (
            "These programs represent different behavioral niches, not necessarily "
            "successful solutions. Their overall performance may be strong, weak, or "
            "poor. Treat them as sources of potentially useful mechanisms or ideas "
            "rather than solutions to imitate wholesale."
        )
        self.assertEqual(user_prompt.count(clarification), 1)
        self.assertNotIn(
            "These programs represent diverse approaches and creative solutions",
            user_prompt,
        )
        self.assertIn(
            "MAP niche: cell=[2, 4]; descriptors: latency=4; source island=1",
            user_prompt,
        )


class TestTrueDiversityControllerWiring(unittest.TestCase):
    @staticmethod
    def _controller_config(true_diversity):
        config = Config()
        config.prompt.true_diversity = true_diversity
        config.prompt.num_top_programs = 1
        config.prompt.num_diverse_programs = 2
        config.database.num_islands = 2
        config.evaluator.parallel_evaluations = 1
        return config

    @staticmethod
    def _program(program_id, score, island):
        return Program(
            id=program_id,
            code=f"def {program_id}(): return {score}",
            metrics={"combined_score": score},
            metadata={"island": island, "map_elites_cell": [int(score * 10)]},
        )

    def test_false_submission_keeps_legacy_sampler_and_worker_call_shape(self):
        config = self._controller_config(False)
        parent = self._program("parent", 0.5, 0)
        inspiration = self._program("legacy_inspiration", 0.4, 0)
        database = MagicMock()
        database.current_island = 0
        database.sample_from_island.return_value = (
            parent,
            [inspiration],
            "island_random",
        )
        controller = ProcessParallelController(config, "evaluator.py", database)
        controller.executor = Mock()
        controller.executor.submit.return_value = MagicMock()

        with patch.object(controller, "_create_database_snapshot", return_value={}):
            controller._submit_iteration(7, island_id=0)

        database.sample_from_island.assert_called_once_with(
            island_id=0,
            num_inspirations=2,
            include_selection_source=True,
        )
        database.sample_true_diverse_inspirations.assert_not_called()
        worker_args = controller.executor.submit.call_args.args
        self.assertEqual(len(worker_args), 6)
        self.assertEqual(worker_args[4], [inspiration.id])
        self.assertEqual(worker_args[5], "island_random")

    def test_true_submission_freezes_parent_island_top_before_target_selection(self):
        config = self._controller_config(True)
        # The parent came from an archive fallback on island 1 while scheduling
        # the child and true-diverse MAP inspirations on target island 0.
        parent = self._program("foreign_parent", 0.5, 1)
        foreign_top = self._program("foreign_top", 0.9, 1)
        foreign_rank_2 = self._program("foreign_rank_2", 0.8, 1)
        selected = self._program("target_map_owner", 0.4, 0)
        audit = {
            "true_diversity": True,
            "requested_count": 2,
            "returned_count": 1,
            "target_island": 0,
            "selected_program_ids": [selected.id],
            "selected_map_cells": {selected.id: [4]},
            "selected_descriptor_values": {selected.id: {}},
            "selected_source_islands": {selected.id: 0},
            "shortfall_count": 1,
            "shortfall_reason": "insufficient_eligible_distinct_cells",
        }
        database = MagicMock()
        database.current_island = 0
        database.programs = {
            program.id: program
            for program in (parent, foreign_top, foreign_rank_2, selected)
        }
        database.islands = [{selected.id}, {parent.id, foreign_top.id, foreign_rank_2.id}]
        database.sample_from_island.return_value = (
            parent,
            [],
            "global_fallback_archive",
        )
        database.sample_true_diverse_inspirations.return_value = ([selected], audit)
        controller = ProcessParallelController(config, "evaluator.py", database)
        controller.executor = Mock()
        controller.executor.submit.return_value = MagicMock()

        with patch.object(controller, "_create_database_snapshot", return_value={}):
            controller._submit_iteration(8, island_id=0)

        database.sample_from_island.assert_called_once_with(
            island_id=0,
            num_inspirations=0,
            include_selection_source=True,
        )
        database.sample_true_diverse_inspirations.assert_called_once_with(
            parent=parent,
            target_island=0,
            n=2,
            excluded_program_ids=[foreign_top.id],
            lineage_depth=3,
        )
        worker_args = controller.executor.submit.call_args.args
        self.assertEqual(worker_args[4], [selected.id])
        self.assertEqual(worker_args[5], "global_fallback_archive")
        self.assertEqual(worker_args[6], [foreign_top.id])
        self.assertIs(worker_args[7], audit)

    def test_worker_passes_only_fixed_top_and_authoritative_niche_to_prompt(self):
        config = self._controller_config(True)
        config.diff_based_evolution = False
        config.language = "python"
        parent = self._program("parent", 0.5, 0)
        top = self._program("top", 0.9, 0)
        pseudo = self._program("pseudo", 0.8, 0)
        inspiration = self._program("map_owner", 0.4, 0)
        programs = {
            program.id: program.to_dict()
            for program in (parent, top, pseudo, inspiration)
        }
        snapshot = {
            "programs": programs,
            "islands": [{parent.id, top.id, pseudo.id, inspiration.id}, set()],
            "current_island": 0,
            "sampling_island": 0,
            "feature_dimensions": ["combined_score"],
            "artifacts": {},
        }
        audit = {
            "true_diversity": True,
            "target_island": 0,
            "selected_map_cells": {inspiration.id: [3, 1]},
            "selected_descriptor_values": {
                inspiration.id: {"combined_score": 0.4}
            },
            "selected_source_islands": {inspiration.id: 0},
        }
        prompt_sampler = Mock()
        prompt_sampler.build_prompt.return_value = {"system": "system", "user": "user"}
        llm = Mock()
        llm.generate_with_context = AsyncMock(
            return_value="```python\ndef evolved(): return 1\n```"
        )
        evaluator = Mock()
        evaluator.evaluate_program = AsyncMock(return_value={"combined_score": 0.6})
        evaluator.get_pending_artifacts.return_value = None

        with (
            patch.object(process_parallel_module, "_lazy_init_worker_components"),
            patch.object(process_parallel_module, "_worker_config", config, create=True),
            patch.object(
                process_parallel_module,
                "_worker_prompt_sampler",
                prompt_sampler,
                create=True,
            ),
            patch.object(process_parallel_module, "_worker_llm_ensemble", llm, create=True),
            patch.object(process_parallel_module, "_worker_evaluator", evaluator, create=True),
        ):
            result = process_parallel_module._run_iteration_worker(
                1,
                snapshot,
                parent.id,
                [inspiration.id],
                "island_random",
                [top.id],
                audit,
            )

        self.assertIsNone(result.error)
        prompt_call = prompt_sampler.build_prompt.call_args.kwargs
        self.assertEqual([p["id"] for p in prompt_call["top_programs"]], [top.id])
        self.assertEqual([p["id"] for p in prompt_call["previous_programs"]], [top.id])
        self.assertEqual([p["id"] for p in prompt_call["inspirations"]], [inspiration.id])
        self.assertEqual(
            prompt_call["inspirations"][0]["true_diversity_niche"],
            {
                "map_cell": [3, 1],
                "descriptor_values": {"combined_score": 0.4},
                "source_island": 0,
            },
        )
        self.assertIs(result.true_diversity_provenance, audit)


class TestTrueDiversityTraceWiring(unittest.TestCase):
    def _trace_metadata(self, provenance, trace_path=None):
        config = Config()
        config.database.num_islands = 1
        config.evaluator.parallel_evaluations = 1
        config.checkpoint_interval = 100
        database = ProgramDatabase(config.database)
        parent = Program(
            id="parent",
            code="def parent(): return 0",
            metrics={"combined_score": 0.5},
        )
        database.add(parent, target_island=0)
        tracer = (
            EvolutionTracer(
                output_path=str(trace_path),
                format="jsonl",
                include_prompts=False,
                buffer_size=1,
            )
            if trace_path is not None
            else Mock()
        )
        controller = ProcessParallelController(
            config, "evaluator.py", database, evolution_tracer=tracer
        )
        controller.executor = Mock()
        result = SerializableResult(
            child_program_dict={
                "id": "child",
                "code": "def child(): return 1",
                "parent_id": parent.id,
                "generation": 1,
                "metrics": {"combined_score": 0.6},
                "iteration_found": 1,
                "metadata": {"changes": "test", "island": 0},
            },
            parent_program_dict=parent.to_dict(),
            parent_id=parent.id,
            iteration=1,
            target_island=0,
            true_diversity_provenance=provenance,
        )
        future = MagicMock()
        future.done.return_value = True
        future.result.return_value = result

        async def complete():
            with patch.object(controller, "_submit_iteration", return_value=future):
                await controller.run_evolution(1, 1)

        asyncio.run(complete())
        if trace_path is not None:
            tracer.close()
            trace = json.loads(Path(trace_path).read_text(encoding="utf-8").splitlines()[0])
            return trace["metadata"]
        return tracer.log_trace.call_args.kwargs["metadata"]

    def test_true_selection_provenance_is_added_to_trace(self):
        provenance = {
            "true_diversity": True,
            "requested_count": 3,
            "returned_count": 1,
            "selected_map_cells": {"map_owner": [1, 2]},
            "shortfall_count": 2,
            "shortfall_reason": "insufficient_eligible_distinct_cells",
        }
        metadata = self._trace_metadata(provenance)
        self.assertTrue(metadata["true_diversity"])
        self.assertEqual(metadata["true_diversity_selection"], provenance)

    def test_false_branch_adds_no_trace_fields(self):
        metadata = self._trace_metadata(None)
        self.assertNotIn("true_diversity", metadata)
        self.assertNotIn("true_diversity_selection", metadata)

    def test_true_selection_provenance_flushes_as_jsonl(self):
        provenance = {
            "true_diversity": True,
            "requested_count": 2,
            "returned_count": 1,
            "target_island": 0,
            "selected_program_ids": ["map_owner"],
            "selected_map_cells": {"map_owner": [1, 2]},
            "selected_descriptor_values": {"map_owner": {"latency": 4.5}},
            "excluded_parent_ids": ["parent"],
            "excluded_top_ids": ["top"],
            "identical_code_exclusions": [],
            "close_relative_exclusions": ["sibling"],
            "shortfall_count": 1,
            "shortfall_reason": "insufficient_distinct_eligible_cells_after_exclusions",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = self._trace_metadata(
                provenance, Path(temp_dir) / "evolution_trace.jsonl"
            )

        self.assertTrue(metadata["true_diversity"])
        self.assertEqual(metadata["true_diversity_selection"], provenance)


if __name__ == "__main__":
    unittest.main()
