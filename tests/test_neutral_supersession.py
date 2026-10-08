"""Focused regression tests for exact-fitness MAP-cell supersession.

Neutral supersession is deliberately narrower than a general preference for
newer programs: it applies only to evidence-backed, finite, exactly
equal-fitness programs with different IDs and different source code that
collide in one island's MAP cell.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


def _database(
    *,
    population_size: int = 100,
    archive_size: int = 20,
    num_islands: int = 1,
    migration_rate: float = 1.0,
) -> ProgramDatabase:
    """Build a database whose two physical cells never move during a test."""

    config = Config()
    config.database.in_memory = True
    config.database.population_size = population_size
    config.database.archive_size = archive_size
    config.database.num_islands = num_islands
    config.database.feature_dimensions = ["cell"]
    config.database.feature_bins = 2
    config.database.feature_bin_edges = {"cell": [0.0]}
    config.database.migration_rate = migration_rate
    config.database.migration_interval = 1
    database = ProgramDatabase(config.database)
    # Prompt caching is lazy in production; initialize it explicitly so these
    # tests can verify retirement cleanup without constructing an LLM prompt.
    database.prompts_by_program = {}
    return database


def _fitness_evidence_database() -> ProgramDatabase:
    """Build a one-cell database that accepts arbitrary fitness mappings."""

    config = Config()
    config.database.in_memory = True
    config.database.num_islands = 1
    config.database.feature_dimensions = ["complexity"]
    config.database.feature_bins = 2
    config.database.feature_bin_edges = {"complexity": [100.0]}
    database = ProgramDatabase(config.database)
    database.prompts_by_program = {}
    return database


def _program_with_metrics(
    program_id: str,
    metrics: dict,
    *,
    timestamp: float = 0.0,
    code: str | None = None,
) -> Program:
    return Program(
        id=program_id,
        code=code if code is not None else f"# {program_id}",
        metrics=dict(metrics),
        timestamp=timestamp,
    )


def _program(
    program_id: str,
    fitness: float,
    *,
    cell: float = 1.0,
    code: str | None = None,
    parent_id: str | None = None,
    generation: int = 0,
    iteration_found: int = 0,
    timestamp: float | None = None,
    changes_description: str = "",
    metadata: dict | None = None,
) -> Program:
    values = {
        "id": program_id,
        "code": code if code is not None else f"def program_{program_id}(): return {fitness!r}",
        "changes_description": changes_description,
        "parent_id": parent_id,
        "generation": generation,
        "iteration_found": iteration_found,
        "metrics": {"combined_score": fitness, "cell": cell},
        "metadata": dict(metadata or {}),
    }
    if timestamp is not None:
        values["timestamp"] = timestamp
    return Program(**values)


def _cell_owner(database: ProgramDatabase, island: int, program: Program) -> str:
    coords = program.metadata["map_elites_cell"]
    key = database._feature_coords_to_key(coords)
    return database.island_feature_maps[island][key]


def _active_state(database: ProgramDatabase) -> dict:
    """Snapshot all active ID-bearing structures used by replacement preflight."""

    return {
        "programs": set(database.programs),
        "maps": copy.deepcopy(database.island_feature_maps),
        "islands": [set(island) for island in database.islands],
        "archive": set(database.archive),
        "best": database.best_program_id,
        "island_bests": list(database.island_best_programs),
        "prompts": copy.deepcopy(database.prompts_by_program),
        "feature_stats": copy.deepcopy(database.feature_stats),
    }


class NeutralSupersessionTests(unittest.TestCase):
    def test_worse_same_cell_candidate_does_not_displace_incumbent(self) -> None:
        database = _database()
        incumbent = _program("old", 0.8, code="def old(): return 1")
        worse = _program("worse", 0.2, code="def worse(): return 2")

        database.add(incumbent, target_island=0)
        database.add(worse, target_island=0)

        self.assertEqual(_cell_owner(database, 0, incumbent), incumbent.id)
        self.assertIn(incumbent.id, database.programs)
        self.assertIn(worse.id, database.programs)
        self.assertIn(worse.id, database.islands[0])
        self.assertNotIn("map_elites_replacement", worse.metadata)

    def test_strict_better_replacement_uses_complete_bookkeeping(self) -> None:
        database = _database()
        incumbent = _program("old", 0.4, code="def old(): return 1")
        better = _program("better", 0.9, code="def better(): return 2")
        database.add(incumbent, target_island=0)
        database.prompts_by_program = {
            incumbent.id: {"evolve": {"user": "old prompt"}}
        }

        database.add(better, target_island=0)

        self.assertEqual(_cell_owner(database, 0, better), better.id)
        self.assertNotIn(incumbent.id, database.programs)
        self.assertNotIn(incumbent.id, database.islands[0])
        self.assertNotIn(incumbent.id, database.archive)
        self.assertNotIn(incumbent.id, database.prompts_by_program)
        self.assertEqual(database.best_program_id, better.id)
        self.assertEqual(database.island_best_programs[0], better.id)
        event = better.metadata["map_elites_replacement"]
        self.assertEqual(event["replacement_kind"], "strict")
        self.assertEqual(event["old_id"], incumbent.id)
        self.assertEqual(event["new_id"], better.id)

    def test_equal_different_code_performs_full_neutral_supersession(self) -> None:
        database = _database()
        incumbent = _program("old", 0.75, code="def old(): return 1")
        newcomer = _program(
            "new",
            0.75,
            code="def new(): return 2",
            parent_id="actual-parent",
            generation=7,
            iteration_found=19,
            timestamp=1234.5,
            changes_description="change only this genotype",
            metadata={"migrant": True, "migration_source_island": 3, "custom": "kept"},
        )
        database.add(incumbent, iteration=11, target_island=0)
        database.prompts_by_program = {
            incumbent.id: {"evolve": {"user": "old prompt"}}
        }

        database.add(newcomer, iteration=23, target_island=0)

        self.assertEqual(set(database.programs), {newcomer.id})
        self.assertEqual(database.islands[0], {newcomer.id})
        self.assertEqual(set(database.island_feature_maps[0].values()), {newcomer.id})
        self.assertEqual(database.archive, {newcomer.id})
        self.assertEqual(database.island_best_programs[0], newcomer.id)
        self.assertEqual(database.best_program_id, newcomer.id)
        self.assertNotIn(incumbent.id, database.prompts_by_program)
        self.assertNotIn(newcomer.id, database.prompts_by_program)

        # Supersession is a representation event, not a rewritten lineage edge.
        self.assertEqual(newcomer.parent_id, "actual-parent")
        self.assertEqual(newcomer.generation, 7)
        self.assertEqual(newcomer.iteration_found, 23)
        self.assertEqual(newcomer.timestamp, 1234.5)
        self.assertEqual(newcomer.changes_description, "change only this genotype")
        self.assertIs(newcomer.metadata["migrant"], True)
        self.assertEqual(newcomer.metadata["migration_source_island"], 3)
        self.assertEqual(newcomer.metadata["custom"], "kept")

        event = newcomer.metadata["map_elites_replacement"]
        expected_event = {
            "old_id": incumbent.id,
            "new_id": newcomer.id,
            "island": 0,
            "cell": [1],
            "exact_fitness": 0.75,
            "iteration": 23,
            "archive_transferred": True,
            "island_best_transferred": True,
            "global_best_transferred": True,
            "replacement_kind": "neutral",
        }
        for key, value in expected_event.items():
            self.assertEqual(event[key], value, key)

    def test_equal_fitness_in_different_cells_coexists(self) -> None:
        database = _database()
        left = _program("left", 0.5, cell=-1.0, code="def left(): return 1")
        right = _program("right", 0.5, cell=1.0, code="def right(): return 2")

        database.add(left, target_island=0)
        database.add(right, target_island=0)

        self.assertEqual(set(database.programs), {left.id, right.id})
        self.assertEqual(set(database.island_feature_maps[0].values()), {left.id, right.id})
        self.assertNotIn("map_elites_replacement", right.metadata)

    def test_identical_code_equal_fitness_does_not_churn_cell_owner(self) -> None:
        database = _database()
        shared_code = "def same(): return 1"
        incumbent = _program("old", 0.5, code=shared_code)
        duplicate = _program("duplicate", 0.5, code=shared_code)

        database.add(incumbent, target_island=0)
        database.add(duplicate, target_island=0)

        self.assertEqual(_cell_owner(database, 0, incumbent), incumbent.id)
        self.assertEqual(database.best_program_id, incumbent.id)
        self.assertIn(incumbent.id, database.programs)
        self.assertIn(duplicate.id, database.programs)
        self.assertNotIn("map_elites_replacement", duplicate.metadata)

    def test_valid_numeric_evidence_allows_exact_neutral_supersession(self) -> None:
        cases = (
            ("combined-zero", {"combined_score": 0.0}, 0.0),
            ("numeric-fallback", {"quality": 0.5567}, 0.5567),
            (
                "invalid-combined-with-fallback",
                {"combined_score": "invalid", "quality": 0.5567},
                0.5567,
            ),
        )
        for label, metrics, expected_fitness in cases:
            with self.subTest(label=label):
                database = _fitness_evidence_database()
                incumbent = _program_with_metrics(f"{label}-old", metrics)
                newcomer = _program_with_metrics(f"{label}-new", metrics)

                database.add(incumbent, target_island=0)
                database.add(newcomer, target_island=0)

                self.assertEqual(_cell_owner(database, 0, newcomer), newcomer.id)
                event = newcomer.metadata["map_elites_replacement"]
                self.assertEqual(event["replacement_kind"], "neutral")
                self.assertEqual(event["exact_fitness"], expected_fitness)

    def test_empty_metrics_preserve_legacy_timestamp_fallback(self) -> None:
        shared_code = "value = 1"

        newer_database = _fitness_evidence_database()
        older_incumbent = _program_with_metrics(
            "empty-old", {}, timestamp=1.0, code=shared_code
        )
        newer_candidate = _program_with_metrics(
            "empty-new", {}, timestamp=2.0, code=shared_code
        )
        newer_database.add(older_incumbent, target_island=0)
        newer_database.add(newer_candidate, target_island=0)

        self.assertEqual(
            _cell_owner(newer_database, 0, newer_candidate), newer_candidate.id
        )
        self.assertEqual(
            newer_candidate.metadata["map_elites_replacement"]["replacement_kind"],
            "strict",
        )

        older_database = _fitness_evidence_database()
        newer_incumbent = _program_with_metrics(
            "newer-incumbent", {}, timestamp=2.0, code=shared_code
        )
        older_candidate = _program_with_metrics(
            "older-candidate", {}, timestamp=1.0, code=shared_code
        )
        older_database.add(newer_incumbent, target_island=0)
        older_database.add(older_candidate, target_island=0)

        self.assertEqual(
            _cell_owner(older_database, 0, newer_incumbent), newer_incumbent.id
        )
        self.assertNotIn("map_elites_replacement", older_candidate.metadata)

    def test_missing_and_valid_zero_preserve_metrics_presence_fallback(self) -> None:
        valid_zero = {"combined_score": 0.0}

        valid_newcomer_database = _fitness_evidence_database()
        empty_incumbent = _program_with_metrics("empty-old", {}, timestamp=2.0)
        valid_newcomer = _program_with_metrics(
            "valid-new", valid_zero, timestamp=1.0
        )
        valid_newcomer_database.add(empty_incumbent, target_island=0)
        valid_newcomer_database.add(valid_newcomer, target_island=0)

        self.assertEqual(
            _cell_owner(valid_newcomer_database, 0, valid_newcomer),
            valid_newcomer.id,
        )
        self.assertEqual(
            valid_newcomer.metadata["map_elites_replacement"]["replacement_kind"],
            "strict",
        )

        empty_newcomer_database = _fitness_evidence_database()
        valid_incumbent = _program_with_metrics(
            "valid-old", valid_zero, timestamp=1.0
        )
        empty_newcomer = _program_with_metrics("empty-new", {}, timestamp=2.0)
        empty_newcomer_database.add(valid_incumbent, target_island=0)
        empty_newcomer_database.add(empty_newcomer, target_island=0)

        self.assertEqual(
            _cell_owner(empty_newcomer_database, 0, valid_incumbent),
            valid_incumbent.id,
        )
        self.assertNotIn("map_elites_replacement", empty_newcomer.metadata)

    def test_unusable_fitness_evidence_does_not_neutral_supersede(self) -> None:
        cases = (
            ("malformed-fallback", {"quality": "not-a-number"}),
            ("boolean-only", {"timeout": True}),
            ("boolean-combined", {"combined_score": True}),
            ("invalid-combined", {"combined_score": "not-a-number"}),
            ("feature-only", {"complexity": 7.0}),
        )
        for label, metrics in cases:
            with self.subTest(label=label):
                database = _fitness_evidence_database()
                incumbent = _program_with_metrics(f"{label}-old", metrics)
                newcomer = _program_with_metrics(f"{label}-new", metrics)

                database.add(incumbent, target_island=0)
                database.add(newcomer, target_island=0)

                self.assertEqual(_cell_owner(database, 0, incumbent), incumbent.id)
                self.assertIn(incumbent.id, database.programs)
                self.assertIn(newcomer.id, database.programs)
                self.assertNotIn("map_elites_replacement", newcomer.metadata)

    def test_nonfinite_equal_scores_do_not_neutral_supersede(self) -> None:
        for index, score in enumerate((math.nan, math.inf, -math.inf)):
            with self.subTest(score=score):
                database = _database()
                incumbent = _program(f"old-{index}", score, code=f"def old_{index}(): pass")
                newcomer = _program(f"new-{index}", score, code=f"def new_{index}(): pass")

                database.add(incumbent, target_island=0)
                database.add(newcomer, target_island=0)

                self.assertEqual(_cell_owner(database, 0, incumbent), incumbent.id)
                self.assertIn(incumbent.id, database.programs)
                self.assertNotIn("map_elites_replacement", newcomer.metadata)

    def test_adjacent_floats_are_not_treated_as_neutral_ties(self) -> None:
        lower_database = _database()
        incumbent = _program("old", 0.5, code="def old(): return 0")
        lower = _program(
            "lower",
            math.nextafter(0.5, -math.inf),
            code="def lower(): return -1",
        )
        lower_database.add(incumbent, target_island=0)
        lower_database.add(lower, target_island=0)
        self.assertEqual(_cell_owner(lower_database, 0, incumbent), incumbent.id)
        self.assertNotIn("map_elites_replacement", lower.metadata)

        upper_database = _database()
        incumbent = _program("old", 0.5, code="def old(): return 0")
        upper = _program(
            "upper",
            math.nextafter(0.5, math.inf),
            code="def upper(): return 1",
        )
        upper_database.add(incumbent, target_island=0)
        upper_database.add(upper, target_island=0)
        self.assertEqual(_cell_owner(upper_database, 0, upper), upper.id)
        self.assertEqual(
            upper.metadata["map_elites_replacement"]["replacement_kind"], "strict"
        )

    def test_full_archive_transfer_does_not_drop_an_unrelated_member(self) -> None:
        database = _database(archive_size=2)
        incumbent = _program("old", 0.8, cell=1.0, code="def old(): return 1")
        unrelated = _program("third", 0.2, cell=-1.0, code="def third(): return 3")
        newcomer = _program("new", 0.8, cell=1.0, code="def new(): return 2")
        database.add(incumbent, target_island=0)
        database.add(unrelated, target_island=0)
        self.assertEqual(database.archive, {incumbent.id, unrelated.id})

        database.add(newcomer, target_island=0)

        self.assertEqual(database.archive, {newcomer.id, unrelated.id})
        self.assertEqual(len(database.archive), 2)
        self.assertTrue(
            newcomer.metadata["map_elites_replacement"]["archive_transferred"]
        )

    def test_selection_and_inspiration_roles_follow_newcomer(self) -> None:
        database = _database()
        incumbent = _program("old", 0.8, code="def old(): return 1")
        newcomer = _program("new", 0.8, code="def new(): return 2")
        database.add(incumbent, target_island=0)
        database.add(newcomer, target_island=0)

        self.assertEqual(database._sample_from_island_random(0).id, newcomer.id)
        self.assertEqual(database._sample_from_island_weighted(0).id, newcomer.id)
        self.assertEqual(database._sample_from_archive_for_island(0).id, newcomer.id)
        self.assertEqual(database._sample_random_parent().id, newcomer.id)

        context = _program("context", 0.1, cell=-1.0, code="def context(): return 3")
        context.metadata["island"] = 0
        database.programs[context.id] = context
        database.islands[0].add(context.id)
        inspirations = database._sample_inspirations(context, n=1, island_id=0)
        self.assertEqual([program.id for program in inspirations], [newcomer.id])
        self.assertEqual(database.island_best_programs[0], newcomer.id)
        self.assertEqual(_cell_owner(database, 0, newcomer), newcomer.id)

    def test_population_count_is_stable_below_at_and_across_repeated_replacements(self) -> None:
        below_capacity = _database(population_size=10)
        current = _program("p0", 0.5, code="def p0(): return 0")
        below_capacity.add(current, target_island=0)
        for index in range(1, 11):
            current = _program(
                f"p{index}",
                0.5,
                code=f"def p{index}(): return {index}",
            )
            below_capacity.add(current, target_island=0)
            self.assertEqual(len(below_capacity.programs), 1)
            self.assertEqual(_cell_owner(below_capacity, 0, current), current.id)

        at_capacity = _database(population_size=2)
        incumbent = _program("old", 0.8, cell=1.0, code="def old(): return 1")
        unrelated = _program("unrelated", 0.2, cell=-1.0, code="def unrelated(): return 3")
        newcomer = _program("new", 0.8, cell=1.0, code="def new(): return 2")
        at_capacity.add(incumbent, target_island=0)
        at_capacity.add(unrelated, target_island=0)
        at_capacity.add(newcomer, target_island=0)

        self.assertEqual(set(at_capacity.programs), {newcomer.id, unrelated.id})
        self.assertEqual(len(at_capacity.programs), 2)

    def test_checkpoint_manifest_excludes_stale_retired_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = _database()
            incumbent = _program("old", 0.5, code="def old(): return 1")
            newcomer = _program("new", 0.5, code="def new(): return 2")
            database.add(incumbent, iteration=1, target_island=0)
            database.save(temp_dir, iteration=1)
            old_path = Path(temp_dir, "programs", "old.json")
            self.assertTrue(old_path.exists())

            database.add(newcomer, iteration=2, target_island=0)
            database.save(temp_dir, iteration=2)
            metadata = json.loads(Path(temp_dir, "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(set(metadata["active_program_ids"]), {newcomer.id})
            self.assertTrue(old_path.exists(), "retired JSON may remain as historical data")

            resumed = ProgramDatabase(database.config)
            resumed.load(temp_dir)

            self.assertEqual(set(resumed.programs), {newcomer.id})
            self.assertNotIn(incumbent.id, resumed.programs)
            self.assertEqual(resumed.islands[0], {newcomer.id})
            self.assertEqual(set(resumed.island_feature_maps[0].values()), {newcomer.id})
            self.assertEqual(resumed.archive, {newcomer.id})
            self.assertEqual(resumed.best_program_id, newcomer.id)
            self.assertEqual(resumed.island_best_programs[0], newcomer.id)
            self.assertEqual(
                resumed.get(newcomer.id).metadata["map_elites_replacement"]["old_id"],
                incumbent.id,
            )

    def test_checkpoint_manifest_preserves_legacy_and_empty_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = _database()
            program = _program("active", 0.5, code="def active(): return 1")
            database.add(program, target_island=0)
            database.save(temp_dir, iteration=1)

            metadata_path = Path(temp_dir, "metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("active_program_ids")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            legacy_resume = ProgramDatabase(database.config)
            legacy_resume.load(temp_dir)
            self.assertEqual(set(legacy_resume.programs), {program.id})

            metadata["active_program_ids"] = []
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            empty_resume = ProgramDatabase(database.config)
            empty_resume.load(temp_dir)
            self.assertEqual(empty_resume.programs, {})
            self.assertEqual(empty_resume.archive, set())
            self.assertTrue(all(not island for island in empty_resume.islands))
            self.assertTrue(
                all(not feature_map for feature_map in empty_resume.island_feature_maps)
            )

    def test_reused_active_id_is_rejected_without_corrupting_roles(self) -> None:
        database = _database()
        incumbent = _program("same-id", 0.5, code="def incumbent(): return 1")
        database.add(incumbent, target_island=0)
        before = _active_state(database)

        with self.assertRaises(ValueError):
            database.add(
                _program("same-id", 0.5, code="def accidental_overwrite(): return 2"),
                target_island=0,
            )

        self.assertEqual(_active_state(database), before)
        self.assertIs(database.get(incumbent.id), incumbent)

    def test_equal_fitness_migrant_supersedes_target_and_keeps_migration_lineage(self) -> None:
        database = _database(num_islands=2, migration_rate=1.0)
        source_seed = _program("source-seed", 0.8, code="def source_seed(): return 0")
        source = _program(
            "source",
            0.8,
            code="def source(): return 1",
            parent_id="source-parent",
            generation=4,
            changes_description="source change",
        )
        target = _program("target", 0.8, code="def target(): return 2")
        database.add(source_seed, target_island=0)
        database.add(source, target_island=0)
        self.assertEqual(
            source.metadata["map_elites_replacement"]["old_id"], source_seed.id
        )
        database.add(target, target_island=1)

        database.migrate_programs()

        migrants = [
            program
            for program in database.programs.values()
            if program.parent_id == source.id
            and program.metadata.get("migrant") is True
            and program.metadata.get("island") == 1
        ]
        self.assertEqual(len(migrants), 1)
        migrant = migrants[0]
        self.assertEqual(migrant.code, source.code)
        self.assertEqual(migrant.generation, source.generation)
        self.assertEqual(migrant.changes_description, source.changes_description)
        self.assertEqual(migrant.metadata["migration_source_island"], 0)
        self.assertNotIn(target.id, database.programs)
        self.assertEqual(_cell_owner(database, 1, migrant), migrant.id)

        event = migrant.metadata["map_elites_replacement"]
        self.assertEqual(event["replacement_kind"], "neutral")
        self.assertEqual(event["old_id"], target.id)
        self.assertEqual(event["new_id"], migrant.id)
        self.assertNotEqual(event["old_id"], source_seed.id)

    def test_malformed_incumbent_roles_fail_without_partial_mutation(self) -> None:
        corruptions = ("second_island", "second_cell")
        for corruption in corruptions:
            with self.subTest(corruption=corruption):
                database = _database(num_islands=2)
                incumbent = _program("old", 0.5, code="def old(): return 1")
                newcomer = _program("new", 0.5, code="def new(): return 2")
                database.add(incumbent, target_island=0)
                database.prompts_by_program = {
                    incumbent.id: {"evolve": {"user": "old prompt"}}
                }
                if corruption == "second_island":
                    database.islands[1].add(incumbent.id)
                else:
                    database.island_feature_maps[1]["0"] = incumbent.id
                before = _active_state(database)

                with self.assertRaises((ValueError, RuntimeError)):
                    database.add(newcomer, target_island=0)

                self.assertEqual(_active_state(database), before)
                self.assertNotIn(newcomer.id, database.programs)

    def test_malformed_replacement_restores_dynamic_feature_statistics(self) -> None:
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 2
        config.database.feature_dimensions = ["cell"]
        config.database.feature_bins = 2
        config.database.feature_bin_edges = {}
        database = ProgramDatabase(config.database)
        database.prompts_by_program = {}

        incumbent = _program("old", 0.5, cell=1.0, code="def old(): return 1")
        newcomer = _program("new", 0.5, cell=1.0, code="def new(): return 2")
        database.add(incumbent, target_island=0)
        database.islands[1].add(incumbent.id)
        before = _active_state(database)

        with self.assertRaises(ValueError):
            database.add(newcomer, target_island=0)

        self.assertEqual(_active_state(database), before)


if __name__ == "__main__":
    unittest.main()
