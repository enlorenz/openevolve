"""Focused tests for commentary-insensitive exact-code prompt Top deduplication."""

import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from openevolve import process_parallel as process_parallel_module
from openevolve.config import Config
from openevolve.database import Program
from openevolve.process_parallel import ProcessParallelController
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import select_distinct_code_programs


COMMENTARY_VARIANT_A = (
    "def shared():\n"
    '    """Strategy A."""\n'
    "    value = 1  # rank-one explanation\n"
    "    return value\n"
)
COMMENTARY_VARIANT_B = (
    "def shared():\n"
    '    """A completely different strategy.\n'
    "    More commentary follows.\n"
    '    """\n'
    "    value = 1  # rank-two explanation\n"
    "    return value\n"
)
COMMENTARY_VARIANT_C = (
    "def shared():\n"
    "    # A comment-only line absent from the other variants.\n"
    "    value = 1\n"
    "    return value\n"
)
PLAIN_SHARED_CODE = "def shared():\n    value = 1\n    return value\n"
DISTINCT_EXECUTABLE_CODE = "def distinct():\n    value = 2\n    return value\n"


def _program(
    program_id,
    score,
    code=None,
    *,
    parent_id=None,
    metadata=None,
):
    program_metadata = {"island": 0}
    program_metadata.update(metadata or {})
    return Program(
        id=program_id,
        code=code if code is not None else f"def {program_id}(): return {score}",
        parent_id=parent_id,
        metrics={"combined_score": score},
        metadata=program_metadata,
    )


def _select(programs, limit):
    return select_distinct_code_programs(
        programs,
        limit,
        lambda program: program.code,
    )


class TestDistinctCodeTopSelection(unittest.TestCase):
    def test_t1_is_unchanged(self):
        first = _program("first", 0.9)
        second = _program("second", 0.8)

        selected, skipped = _select([first, second], 1)

        self.assertEqual([program.id for program in selected], [first.id])
        self.assertEqual(skipped, [])

    def test_t2_distinct_code_preserves_rank_order(self):
        first = _program("first", 0.9)
        second = _program("second", 0.8)

        selected, skipped = _select([first, second], 2)

        self.assertEqual([program.id for program in selected], [first.id, second.id])
        self.assertEqual(skipped, [])

    def test_duplicate_rank_two_backfills_from_rank_three(self):
        first = _program("first", 0.9, "def shared(): return 1")
        duplicate = _program("duplicate", 0.8, first.code)
        third = _program("third", 0.7, "def distinct(): return 2")

        selected, skipped = _select([first, duplicate, third], 2)

        self.assertEqual([program.id for program in selected], [first.id, third.id])
        self.assertEqual([program.id for program in skipped], [duplicate.id])

    def test_commentary_only_rank_two_backfills_from_rank_three(self):
        first = _program("first", 0.9, COMMENTARY_VARIANT_A)
        commentary_duplicate = _program(
            "commentary-duplicate",
            0.8,
            COMMENTARY_VARIANT_B,
        )
        third = _program("third", 0.7, DISTINCT_EXECUTABLE_CODE)

        selected, skipped = _select([first, commentary_duplicate, third], 2)

        self.assertEqual([program.id for program in selected], [first.id, third.id])
        self.assertEqual(
            [program.id for program in skipped],
            [commentary_duplicate.id],
        )

    def test_t3_interleaved_duplicates_keep_highest_ranked_representatives(self):
        first_x = _program("first-x", 0.9, "code X")
        second_x = _program("second-x", 0.8, "code X")
        first_y = _program("first-y", 0.7, "code Y")
        third_x = _program("third-x", 0.6, "code X")
        first_z = _program("first-z", 0.5, "code Z")

        selected, skipped = _select(
            [first_x, second_x, first_y, third_x, first_z], 3
        )

        self.assertEqual(
            [program.id for program in selected],
            [first_x.id, first_y.id, first_z.id],
        )
        self.assertEqual(
            [program.id for program in skipped],
            [second_x.id, third_x.id],
        )

    def test_fewer_distinct_codes_returns_honest_shortfall(self):
        first = _program("first", 0.9, "same code")
        second = _program("second", 0.8, "same code")
        third = _program("third", 0.7, "same code")

        selected, skipped = _select([first, second, third], 3)

        self.assertEqual([program.id for program in selected], [first.id])
        self.assertEqual(
            [program.id for program in skipped], [second.id, third.id]
        )

    def test_commentary_only_candidates_return_honest_shortfall(self):
        first = _program("first", 0.9, COMMENTARY_VARIANT_A)
        second = _program("second", 0.8, COMMENTARY_VARIANT_B)
        third = _program("third", 0.7, COMMENTARY_VARIANT_C)
        without_commentary = _program("plain", 0.6, PLAIN_SHARED_CODE)

        selected, skipped = _select(
            [first, second, third, without_commentary],
            3,
        )

        self.assertEqual([program.id for program in selected], [first.id])
        self.assertEqual(
            [program.id for program in skipped],
            [second.id, third.id, without_commentary.id],
        )

    def test_equal_fitness_distinct_code_preserves_existing_tie_order(self):
        tied_second = _program("tied-second", 0.5, "code second")
        tied_first = _program("tied-first", 0.5, "code first")

        selected, skipped = _select([tied_second, tied_first], 2)

        self.assertEqual(
            [program.id for program in selected],
            [tied_second.id, tied_first.id],
        )
        self.assertEqual(skipped, [])

    def test_whitespace_differences_are_not_normalized(self):
        plain = _program("plain", 0.9, "return 1")
        trailing_space = _program("trailing-space", 0.8, "return 1 ")

        selected, skipped = _select([plain, trailing_space], 2)

        self.assertEqual(
            [program.id for program in selected],
            [plain.id, trailing_space.id],
        )
        self.assertEqual(skipped, [])

    def test_relatives_and_migrants_with_different_code_remain_eligible(self):
        parent = _program("lineage-parent", 0.9, "code parent")
        child = _program(
            "child",
            0.8,
            "code child",
            parent_id=parent.id,
        )
        sibling = _program(
            "sibling",
            0.7,
            "code sibling",
            parent_id=parent.id,
        )
        migrant = _program(
            "migrant",
            0.6,
            "code migrant",
            parent_id=parent.id,
            metadata={"migrant": True, "migration_source_island": 1},
        )

        selected, skipped = _select([parent, child, sibling, migrant], 4)

        self.assertEqual(
            [program.id for program in selected],
            [parent.id, child.id, sibling.id, migrant.id],
        )
        self.assertEqual(skipped, [])


class TestTopPromptRendering(unittest.TestCase):
    @staticmethod
    def _prompt(true_diversity, top_programs):
        config = Config()
        config.prompt.true_diversity = true_diversity
        config.prompt.num_top_programs = 2
        config.prompt.num_diverse_programs = 0
        config.prompt.include_artifacts = False
        return PromptSampler(config.prompt).build_prompt(
            current_program="def current(): return 0",
            parent_program="def current(): return 0",
            program_metrics={"combined_score": 0.1},
            top_programs=[program.to_dict() for program in top_programs],
            inspirations=[],
        )["user"]

    def test_true_and_legacy_render_rank_one_and_rank_three(self):
        first = _program("first", 0.9, "def shared(): return 1")
        duplicate = _program("duplicate", 0.8, first.code)
        third = _program("third", 0.7, "def distinct(): return 2")

        for true_diversity in (False, True):
            with self.subTest(true_diversity=true_diversity):
                prompt = self._prompt(true_diversity, [first, duplicate, third])

                self.assertEqual(prompt.count(first.code), 1)
                self.assertEqual(prompt.count(third.code), 1)
                self.assertIn("### Program 1 (Score: 0.9000)", prompt)
                self.assertIn("### Program 2 (Score: 0.7000)", prompt)
                self.assertNotIn("### Program 2 (Score: 0.8000)", prompt)

    def test_true_and_legacy_render_past_commentary_only_duplicate(self):
        first = _program("first", 0.9, COMMENTARY_VARIANT_A)
        commentary_duplicate = _program(
            "commentary-duplicate",
            0.8,
            COMMENTARY_VARIANT_B,
        )
        third = _program("third", 0.7, DISTINCT_EXECUTABLE_CODE)

        for true_diversity in (False, True):
            with self.subTest(true_diversity=true_diversity):
                prompt = self._prompt(
                    true_diversity,
                    [first, commentary_duplicate, third],
                )

                self.assertIn(first.code, prompt)
                self.assertNotIn(commentary_duplicate.code, prompt)
                self.assertIn(third.code, prompt)
                self.assertIn("### Program 1 (Score: 0.9000)", prompt)
                self.assertIn("### Program 2 (Score: 0.7000)", prompt)
                self.assertNotIn("### Program 2 (Score: 0.8000)", prompt)

    def test_all_duplicate_code_renders_only_one_top_program(self):
        first = _program("first", 0.9, "def shared(): return 1")
        duplicate = _program("duplicate", 0.8, first.code)

        prompt = self._prompt(False, [first, duplicate])

        self.assertEqual(prompt.count(first.code), 1)
        self.assertIn("### Program 1 (Score: 0.9000)", prompt)
        self.assertNotIn("### Program 2", prompt)

    def test_commentary_only_duplicates_render_an_honest_shortfall(self):
        first = _program("first", 0.9, COMMENTARY_VARIANT_A)
        commentary_duplicate = _program(
            "commentary-duplicate",
            0.8,
            COMMENTARY_VARIANT_B,
        )
        without_commentary = _program("plain", 0.7, PLAIN_SHARED_CODE)

        for true_diversity in (False, True):
            with self.subTest(true_diversity=true_diversity):
                prompt = self._prompt(
                    true_diversity,
                    [first, commentary_duplicate, without_commentary],
                )

                self.assertIn(first.code, prompt)
                self.assertNotIn(commentary_duplicate.code, prompt)
                self.assertNotIn(without_commentary.code, prompt)
                self.assertIn("### Program 1 (Score: 0.9000)", prompt)
                self.assertNotIn("### Program 2", prompt)


class TestTopSelectionPipeline(unittest.TestCase):
    @staticmethod
    def _config(true_diversity):
        config = Config()
        config.prompt.true_diversity = true_diversity
        config.prompt.num_top_programs = 2
        config.prompt.num_diverse_programs = 0
        config.database.num_islands = 1
        config.evaluator.parallel_evaluations = 1
        config.diff_based_evolution = False
        config.language = "python"
        return config

    @staticmethod
    def _run_worker_and_capture_prompt(
        config,
        parent,
        ranked_programs,
        *,
        top_program_ids=None,
    ):
        all_programs = [parent, *ranked_programs]
        snapshot = {
            "programs": {
                program.id: program.to_dict()
                for program in all_programs
            },
            "islands": [{program.id for program in all_programs}],
            "current_island": 0,
            "sampling_island": 0,
            "feature_dimensions": [],
            "artifacts": {},
        }
        prompt_sampler = Mock()
        prompt_sampler.build_prompt.return_value = {
            "system": "system",
            "user": "user",
        }
        llm = Mock()
        llm.generate_with_context = AsyncMock(
            return_value="```python\ndef evolved(): return 1\n```"
        )
        evaluator = Mock()
        evaluator.evaluate_program = AsyncMock(
            return_value={"combined_score": 0.2}
        )
        evaluator.get_pending_artifacts.return_value = None

        with (
            patch.object(process_parallel_module, "_lazy_init_worker_components"),
            patch.object(
                process_parallel_module,
                "_worker_config",
                config,
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_prompt_sampler",
                prompt_sampler,
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_llm_ensemble",
                llm,
                create=True,
            ),
            patch.object(
                process_parallel_module,
                "_worker_evaluator",
                evaluator,
                create=True,
            ),
        ):
            result = process_parallel_module._run_iteration_worker(
                1,
                snapshot,
                parent.id,
                [],
                "island_random",
                top_program_ids,
            )

        return result, prompt_sampler.build_prompt.call_args.kwargs

    @staticmethod
    def _submit_true_diversity_iteration(parent, ranked_programs):
        config = TestTopSelectionPipeline._config(True)
        all_programs = [parent, *ranked_programs]
        audit = {"true_diversity": True, "selected_program_ids": []}
        database = MagicMock()
        database.current_island = 0
        database.programs = {
            program.id: program
            for program in all_programs
        }
        database.islands = [{program.id for program in all_programs}]
        database.sample_from_island.return_value = (
            parent,
            [],
            "island_random",
        )
        database.sample_true_diverse_inspirations.return_value = ([], audit)
        controller = ProcessParallelController(config, "evaluator.py", database)
        controller.executor = Mock()
        controller.executor.submit.return_value = MagicMock()

        with patch.object(controller, "_create_database_snapshot", return_value={}):
            controller._submit_iteration(1, island_id=0)

        return database, controller.executor.submit.call_args.args, audit

    def test_legacy_worker_scans_beyond_raw_top_t(self):
        config = self._config(False)
        parent = _program("parent", 0.1)
        first = _program("first", 0.9, "def shared(): return 1")
        duplicate = _program("duplicate", 0.8, first.code)
        third = _program("third", 0.7, "def distinct(): return 2")

        result, prompt_call = self._run_worker_and_capture_prompt(
            config,
            parent,
            [first, duplicate, third],
        )

        self.assertIsNone(result.error)
        self.assertEqual(
            [program["id"] for program in prompt_call["top_programs"]],
            [first.id, third.id],
        )
        self.assertEqual(
            [program["id"] for program in prompt_call["previous_programs"]],
            [first.id, third.id],
        )

    def test_legacy_worker_scans_past_commentary_only_duplicate(self):
        config = self._config(False)
        parent = _program("parent", 0.1)
        first = _program("first", 0.9, COMMENTARY_VARIANT_A)
        commentary_duplicate = _program(
            "commentary-duplicate",
            0.8,
            COMMENTARY_VARIANT_B,
        )
        third = _program("third", 0.7, DISTINCT_EXECUTABLE_CODE)

        result, prompt_call = self._run_worker_and_capture_prompt(
            config,
            parent,
            [first, commentary_duplicate, third],
        )

        self.assertIsNone(result.error)
        self.assertEqual(
            [program["id"] for program in prompt_call["top_programs"]],
            [first.id, third.id],
        )
        self.assertEqual(
            [program["id"] for program in prompt_call["previous_programs"]],
            [first.id, third.id],
        )

    def test_true_worker_fallback_and_frozen_ids_skip_commentary_duplicate(self):
        config = self._config(True)
        parent = _program("parent", 0.1)
        first = _program("first", 0.9, COMMENTARY_VARIANT_A)
        commentary_duplicate = _program(
            "commentary-duplicate",
            0.8,
            COMMENTARY_VARIANT_B,
        )
        third = _program("third", 0.7, DISTINCT_EXECUTABLE_CODE)

        cases = (
            ("fallback", None),
            (
                "frozen",
                [first.id, commentary_duplicate.id, third.id],
            ),
        )
        for name, top_program_ids in cases:
            with self.subTest(path=name):
                result, prompt_call = self._run_worker_and_capture_prompt(
                    config,
                    parent,
                    [first, commentary_duplicate, third],
                    top_program_ids=top_program_ids,
                )

                self.assertIsNone(result.error)
                self.assertEqual(
                    [program["id"] for program in prompt_call["top_programs"]],
                    [first.id, third.id],
                )
                self.assertEqual(
                    [program["id"] for program in prompt_call["previous_programs"]],
                    [first.id, third.id],
                )

    def test_true_controller_freezes_distinct_code_top_ids(self):
        parent = _program("parent", 0.1)
        first = _program("first", 0.9, "def shared(): return 1")
        duplicate = _program("duplicate", 0.8, first.code)
        third = _program("third", 0.7, "def distinct(): return 2")

        database, worker_args, audit = self._submit_true_diversity_iteration(
            parent,
            [first, duplicate, third],
        )

        database.sample_true_diverse_inspirations.assert_called_once_with(
            parent=parent,
            target_island=0,
            n=0,
            excluded_program_ids=[first.id, third.id],
            lineage_depth=3,
        )
        self.assertEqual(worker_args[6], [first.id, third.id])
        self.assertIs(worker_args[7], audit)

    def test_true_controller_skips_commentary_duplicate_in_frozen_top_ids(self):
        parent = _program("parent", 0.1)
        first = _program("first", 0.9, COMMENTARY_VARIANT_A)
        commentary_duplicate = _program(
            "commentary-duplicate",
            0.8,
            COMMENTARY_VARIANT_B,
        )
        third = _program("third", 0.7, DISTINCT_EXECUTABLE_CODE)

        database, worker_args, audit = self._submit_true_diversity_iteration(
            parent,
            [first, commentary_duplicate, third],
        )

        database.sample_true_diverse_inspirations.assert_called_once_with(
            parent=parent,
            target_island=0,
            n=0,
            excluded_program_ids=[first.id, third.id],
            lineage_depth=3,
        )
        self.assertEqual(worker_args[6], [first.id, third.id])
        self.assertIs(worker_args[7], audit)


if __name__ == "__main__":
    unittest.main()
