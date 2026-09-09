"""Tests for opt-in physical MAP-Elites bin edges."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from openevolve.config import Config, DatabaseConfig
from openevolve.database import Program, ProgramDatabase


WAIT_EDGES = [0, 2, 4, 8, 12, 18, 24, 36, 72]
PRICE_EDGES = [-40, -30, -20, -10, -5, 0, 5, 10, 20]


def fixed_config(**overrides) -> DatabaseConfig:
    values = {
        "population_size": 100,
        "archive_size": 100,
        "num_islands": 1,
        "feature_dimensions": [
            "additional_wait_hours",
            "mean_price_delta_pct_vs_baseline_off",
        ],
        "feature_bin_edges": {
            "additional_wait_hours": WAIT_EDGES,
            "mean_price_delta_pct_vs_baseline_off": PRICE_EDGES,
        },
    }
    values.update(overrides)
    return DatabaseConfig(**values)


def program(program_id: str, wait: float, price: float, score: float = 0.5) -> Program:
    return Program(
        id=program_id,
        code=f"# {program_id}",
        metrics={
            "combined_score": score,
            "additional_wait_hours": wait,
            "mean_price_delta_pct_vs_baseline_off": price,
        },
    )


class FixedFeatureBinTests(unittest.TestCase):
    def test_boundary_semantics_match_right_insertion(self) -> None:
        config = DatabaseConfig(
            population_size=50,
            archive_size=10,
            num_islands=1,
            feature_dimensions=["x"],
            feature_bins=2,
            feature_bin_edges={"x": [-10, 0, 5]},
        )
        database = ProgramDatabase(config)
        cases = [
            (-100, 0),
            (-10, 1),
            (-9.5, 1),
            (0, 2),
            (2.5, 2),
            (5, 3),
            (100, 3),
        ]

        for index, (value, expected_bin) in enumerate(cases):
            with self.subTest(value=value):
                candidate = Program(
                    id=f"boundary-{index}",
                    code="# boundary",
                    metrics={"combined_score": 0.5, "x": value},
                )
                self.assertEqual(database._calculate_feature_coords(candidate), [expected_bin])

    def test_every_powersched_edge_maps_to_the_bin_on_its_right(self) -> None:
        database = ProgramDatabase(fixed_config())

        for index, edge in enumerate(WAIT_EDGES):
            with self.subTest(dimension="wait", edge=edge):
                candidate = program(f"wait-{index}", edge, -15)
                self.assertEqual(database._calculate_feature_coords(candidate)[0], index + 1)
        for index, edge in enumerate(PRICE_EDGES):
            with self.subTest(dimension="price", edge=edge):
                candidate = program(f"price-{index}", 1, edge)
                self.assertEqual(database._calculate_feature_coords(candidate)[1], index + 1)

    def test_fixed_edges_override_legacy_bin_count_and_support_mixed_mode(self) -> None:
        config = DatabaseConfig(
            feature_dimensions=["fixed", "dynamic"],
            feature_bins={"fixed": 99, "dynamic": 4},
            feature_bin_edges={"fixed": [0, 10]},
        )
        database = ProgramDatabase(config)
        candidate = Program(
            id="mixed",
            code="# mixed",
            metrics={"combined_score": 0.5, "fixed": 5, "dynamic": 100},
        )

        self.assertEqual(database._calculate_feature_coords(candidate), [1, 2])
        self.assertEqual(database.feature_bins_per_dim, {"fixed": 3, "dynamic": 4})
        self.assertNotIn("fixed", database.feature_stats)
        self.assertIn("dynamic", database.feature_stats)

    def test_fixed_coordinate_does_not_change_after_population_extrema(self) -> None:
        database = ProgramDatabase(fixed_config())
        reference = program("reference", 15, -17)
        expected = database._calculate_feature_coords(reference)

        database.add(program("low-extreme", -10_000, -10_000))
        database.add(program("high-extreme", 10_000, 10_000))

        self.assertEqual(expected, [5, 3])
        self.assertEqual(database._calculate_feature_coords(reference), expected)
        self.assertEqual(database.feature_stats, {})

    def test_legacy_dynamic_binning_still_tracks_observed_extrema(self) -> None:
        config = DatabaseConfig(
            feature_dimensions=["x"],
            feature_bins=10,
            archive_size=10,
        )
        database = ProgramDatabase(config)
        reference = Program(
            id="reference",
            code="# reference",
            metrics={"combined_score": 0.5, "x": 5.0},
        )

        self.assertEqual(database._calculate_feature_coords(reference), [5])
        database._calculate_feature_coords(
            Program(
                id="new-min",
                code="# min",
                metrics={"combined_score": 0.5, "x": 0.0},
            )
        )
        self.assertEqual(database._calculate_feature_coords(reference), [9])

    def test_legacy_bootstrap_bins_remain_zero_without_updating_stats(self) -> None:
        diversity_database = ProgramDatabase(
            DatabaseConfig(feature_dimensions=["diversity"], feature_bins=10)
        )
        score_database = ProgramDatabase(
            DatabaseConfig(feature_dimensions=["score"], feature_bins=10)
        )

        self.assertEqual(
            diversity_database._calculate_feature_coords(
                Program(id="first", code="# first", metrics={"combined_score": 0.5})
            ),
            [0],
        )
        self.assertEqual(
            score_database._calculate_feature_coords(
                Program(id="empty", code="# empty", metrics={})
            ),
            [0],
        )
        self.assertEqual(diversity_database.feature_stats, {})
        self.assertEqual(score_database.feature_stats, {})

    def test_fixed_builtin_bootstrap_values_use_physical_edges(self) -> None:
        diversity_database = ProgramDatabase(
            DatabaseConfig(
                feature_dimensions=["diversity"],
                feature_bin_edges={"diversity": [-1, 0, 1]},
            )
        )
        score_database = ProgramDatabase(
            DatabaseConfig(
                feature_dimensions=["score"],
                feature_bin_edges={"score": [-1, 0, 1]},
            )
        )

        self.assertEqual(
            diversity_database._calculate_feature_coords(
                Program(id="first", code="# first", metrics={"combined_score": 0.5})
            ),
            [2],
        )
        self.assertEqual(
            score_database._calculate_feature_coords(
                Program(id="empty", code="# empty", metrics={})
            ),
            [2],
        )

    def test_invalid_fixed_descriptor_is_explicitly_unmapped(self) -> None:
        database = ProgramDatabase(fixed_config())
        invalid = program("invalid", math.nan, -10)
        missing = Program(
            id="missing",
            code="# missing",
            metrics={"combined_score": -1.0, "runs_successfully": 0.0},
        )

        database.add(invalid)
        database.add(missing)

        for candidate in (invalid, missing):
            self.assertIsNone(candidate.metadata["map_elites_cell"])
            self.assertIn("map_elites_invalid_descriptor", candidate.metadata)
            self.assertIn(candidate.id, database.islands[0])
        self.assertEqual(database.island_feature_maps[0], {})

    def test_power_sched_grid_has_one_hundred_cells_and_logs_correct_coverage(self) -> None:
        database = ProgramDatabase(fixed_config())
        representative_wait_values = [-1, 0, 2, 4, 8, 12, 18, 24, 36, 72]

        with self.assertLogs("openevolve.database", level="INFO") as captured:
            for index, wait in enumerate(representative_wait_values):
                database.add(program(f"coverage-{index}", wait, -7, score=0.1 + index / 100))

        self.assertEqual(database.total_feature_cells, 100)
        self.assertEqual(database.feature_bins_per_dim, {
            "additional_wait_hours": 10,
            "mean_price_delta_pct_vs_baseline_off": 10,
        })
        self.assertIn("10.0% (10/100 cells)", "\n".join(captured.output))


class FixedFeatureConfigValidationTests(unittest.TestCase):
    def test_yaml_schema_loads_and_normalizes_edges(self) -> None:
        config = Config.from_dict(
            {
                "database": {
                    "feature_dimensions": ["x"],
                    "feature_bin_edges": {"x": [0, 2, 5]},
                }
            }
        )
        self.assertEqual(config.database.feature_bin_edges, {"x": [0.0, 2.0, 5.0]})

    def test_invalid_edge_configurations_are_rejected(self) -> None:
        invalid_cases = [
            {"x": []},
            {"x": [0, 0]},
            {"x": [1, 0]},
            {"x": [0, math.nan]},
            {"x": [0, math.inf]},
            {"x": [False, 1]},
            {"x": [0, "not-a-number"]},
        ]
        for edges in invalid_cases:
            with self.subTest(edges=edges):
                with self.assertRaises(ValueError):
                    DatabaseConfig(feature_dimensions=["x"], feature_bin_edges=edges)

        with self.assertRaises(ValueError):
            DatabaseConfig(
                feature_dimensions=["x"],
                feature_bin_edges={"not-x": [0]},
            )

    def test_database_revalidates_mutated_config(self) -> None:
        config = DatabaseConfig(feature_dimensions=["x"])
        config.feature_bin_edges = {"x": [2, 1]}
        with self.assertRaises(ValueError):
            ProgramDatabase(config)


class FixedFeaturePersistenceTests(unittest.TestCase):
    def test_checkpoint_and_resume_preserve_authoritative_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = fixed_config(num_islands=2)
            database = ProgramDatabase(config)
            first = program("first", 15, -17)
            second = program("second", 30, 7)
            database.add(first, target_island=0)
            database.add(second, target_island=1)
            expected_maps = [dict(feature_map) for feature_map in database.island_feature_maps]
            database.save(temp_dir, iteration=12)

            metadata = json.loads(
                (Path(temp_dir) / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["map_elites_config"], database._map_elites_config())

            resumed = ProgramDatabase(config)
            resumed.load(temp_dir)
            self.assertEqual(resumed.island_feature_maps, expected_maps)
            self.assertEqual(resumed.get("first").metadata["map_elites_cell"], [5, 3])
            self.assertEqual(resumed.get("second").metadata["map_elites_cell"], [7, 7])

            resumed.add(program("new-extreme", 1_000_000, -1_000_000), target_island=0)
            self.assertEqual(resumed.get("first").metadata["map_elites_cell"], [5, 3])
            self.assertEqual(resumed.island_feature_maps[0]["5-3"], "first")

    def test_resume_rejects_changed_fixed_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = ProgramDatabase(fixed_config())
            database.add(program("first", 1, -10))
            database.save(temp_dir)

            changed = fixed_config(
                feature_bin_edges={
                    "additional_wait_hours": [0, 1, 2],
                    "mean_price_delta_pct_vs_baseline_off": PRICE_EDGES,
                }
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                ProgramDatabase(changed).load(temp_dir)

    def test_occupied_legacy_checkpoint_cannot_be_reinterpreted_as_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = {
                "island_feature_maps": [{"0-0": "old"}],
                "islands": [["old"]],
            }
            Path(temp_dir, "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "legacy checkpoint"):
                ProgramDatabase(fixed_config()).load(temp_dir)

    def test_same_fixed_cell_is_island_specific(self) -> None:
        database = ProgramDatabase(fixed_config(num_islands=2))
        left = program("left", 1, -17, score=0.4)
        right = program("right", 1, -17, score=0.5)
        database.add(left, target_island=0)
        database.add(right, target_island=1)

        self.assertEqual(left.metadata["map_elites_cell"], [1, 3])
        self.assertEqual(right.metadata["map_elites_cell"], [1, 3])
        self.assertEqual(database.island_feature_maps[0]["1-3"], "left")
        self.assertEqual(database.island_feature_maps[1]["1-3"], "right")

    def test_migration_preserves_fixed_cell_and_provenance(self) -> None:
        database = ProgramDatabase(
            fixed_config(num_islands=2, migration_rate=1.0, migration_interval=1)
        )
        source = program("source", 1, -17, score=0.9)
        other = program("other", 30, 7, score=0.2)
        database.add(source, target_island=0)
        database.add(other, target_island=1)

        database.migrate_programs()

        source_migrants = [
            candidate
            for candidate in database.programs.values()
            if candidate.parent_id == source.id and candidate.metadata.get("migrant") is True
        ]
        self.assertEqual(len(source_migrants), 1)
        migrant = source_migrants[0]
        self.assertEqual(migrant.metadata["migration_source_island"], 0)
        self.assertEqual(migrant.metadata["island"], 1)
        self.assertEqual(migrant.metadata["map_elites_cell"], [1, 3])
        self.assertEqual(database.island_feature_maps[1]["1-3"], migrant.id)

    def test_cross_island_archive_parent_child_uses_target_island_fixed_cell(self) -> None:
        database = ProgramDatabase(
            fixed_config(
                num_islands=2,
                exploration_ratio=0.0,
                exploitation_ratio=1.0,
            )
        )
        local = program("local", 30, 7, score=0.2)
        foreign = program("foreign", 1, -17, score=0.8)
        database.add(local, target_island=0)
        database.add(foreign, target_island=1)
        database.archive = {foreign.id}

        parent, _, source = database.sample_from_island(
            island_id=0,
            num_inspirations=0,
            include_selection_source=True,
        )
        child = program("child", 3, -25, score=0.9)
        child.parent_id = parent.id
        database.add(child, target_island=0)

        self.assertEqual(parent.id, foreign.id)
        self.assertEqual(source, "global_fallback_archive")
        self.assertEqual(child.metadata["island"], 0)
        self.assertEqual(child.metadata["map_elites_cell"], [2, 2])
        self.assertEqual(database.island_feature_maps[0]["2-2"], child.id)


if __name__ == "__main__":
    unittest.main()
