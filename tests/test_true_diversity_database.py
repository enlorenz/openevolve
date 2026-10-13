"""Focused tests for the opt-in database-side true-diversity selector."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


def _database(num_islands: int = 2) -> ProgramDatabase:
    return ProgramDatabase(
        DatabaseConfig(
            population_size=100,
            archive_size=100,
            num_islands=num_islands,
            feature_dimensions=["descriptor"],
            feature_bins=12,
            feature_bin_edges={"descriptor": list(range(11))},
            random_seed=7,
        )
    )


def _program(
    program_id: str,
    *,
    code: str | None = None,
    score: float | None = 0.5,
    descriptor: float = 0.5,
    island: int = 0,
    parent_id: str | None = None,
    metadata: dict | None = None,
) -> Program:
    metrics = {"descriptor": descriptor}
    if score is not None:
        metrics["combined_score"] = score
    program_metadata = {"island": island}
    program_metadata.update(metadata or {})
    return Program(
        id=program_id,
        code=code if code is not None else f"def {program_id.replace('-', '_')}(): pass",
        parent_id=parent_id,
        metrics=metrics,
        metadata=program_metadata,
    )


def _install_live(database: ProgramDatabase, candidate: Program) -> None:
    island = candidate.metadata["island"]
    database.programs[candidate.id] = candidate
    database.islands[island].add(candidate.id)


def _install_owner(database: ProgramDatabase, candidate: Program, cell: int) -> None:
    _install_live(database, candidate)
    candidate.metadata["map_elites_cell"] = [cell]
    database.island_feature_maps[candidate.metadata["island"]][str(cell)] = candidate.id


class TrueDiversityDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(19)

    def test_uses_target_map_owners_and_excludes_parent_top_and_duplicate_code(self) -> None:
        database = _database()
        parent = _program(
            "foreign-parent",
            code="def parent(): return 1",
            island=1,
        )
        top = _program("top", code="def top(): return 1")
        parent_clone = _program("parent-clone", code=parent.code)
        top_clone = _program("top-clone", code=top.code)
        first = _program("first", descriptor=3.5)
        second = _program("second", descriptor=4.5)
        current_owner = _program("current-owner", descriptor=5.5)
        former_owner = _program("former-owner", descriptor=5.5)

        _install_live(database, parent)
        _install_owner(database, top, 0)
        _install_owner(database, parent_clone, 1)
        _install_owner(database, top_clone, 2)
        _install_owner(database, first, 3)
        _install_owner(database, second, 4)
        _install_live(database, former_owner)
        former_owner.metadata["map_elites_cell"] = [5]
        _install_owner(database, current_owner, 5)

        selected, audit = database.sample_true_diverse_inspirations(
            parent,
            target_island=0,
            n=10,
            excluded_program_ids={top.id},
        )

        selected_ids = {program.id for program in selected}
        self.assertEqual(selected_ids, {first.id, second.id, current_owner.id})
        self.assertNotIn(former_owner.id, selected_ids)
        self.assertEqual(len(audit["selected_map_cells"]), len(selected))
        self.assertEqual(
            {tuple(cell) for cell in audit["selected_map_cells"].values()},
            {(3,), (4,), (5,)},
        )
        self.assertEqual(audit["excluded_parent_ids"], [parent.id])
        self.assertEqual(audit["excluded_top_ids"], [top.id])
        self.assertCountEqual(
            audit["identical_code_exclusions"],
            [parent_clone.id, top_clone.id],
        )
        self.assertTrue(audit["foreign_parent"])
        self.assertEqual(set(audit["selected_source_islands"].values()), {0})
        self.assertEqual(
            audit["selected_descriptor_values"][first.id], {"descriptor": 3.5}
        )
        json.dumps(audit)

    def test_excludes_direct_sibling_migration_and_recent_common_ancestor_relations(self) -> None:
        database = _database()
        root = _program("root")
        shared_ancestor = _program("shared-ancestor", parent_id=root.id)
        parent_middle = _program("parent-middle", parent_id=shared_ancestor.id)
        candidate_middle = _program("candidate-middle", parent_id=shared_ancestor.id)
        parent = _program("parent", island=1, parent_id=parent_middle.id)
        top = _program("top")
        direct_child = _program("direct-child", parent_id=parent.id)
        sibling = _program("sibling", parent_id=parent_middle.id)
        migration_clone = _program(
            "migration-clone",
            parent_id=parent.id,
            metadata={"migrant": True, "migration_source_island": 1},
        )
        common_ancestor_relative = _program(
            "common-ancestor-relative", parent_id=candidate_middle.id
        )
        top_child = _program("top-child", parent_id=top.id)
        unrelated = _program("unrelated")

        for live_program in (
            root,
            shared_ancestor,
            parent_middle,
            candidate_middle,
            parent,
            top,
        ):
            _install_live(database, live_program)
        for cell, owner in enumerate(
            (
                direct_child,
                sibling,
                migration_clone,
                common_ancestor_relative,
                top_child,
                unrelated,
            )
        ):
            _install_owner(database, owner, cell)

        selected, audit = database.sample_true_diverse_inspirations(
            parent,
            target_island=0,
            n=10,
            excluded_program_ids={top.id},
        )

        self.assertEqual([program.id for program in selected], [unrelated.id])
        self.assertCountEqual(
            audit["close_relative_exclusions"],
            [
                direct_child.id,
                sibling.id,
                migration_clone.id,
                common_ancestor_relative.id,
                top_child.id,
            ],
        )
        reasons = {
            detail["program_id"]: detail["reason"]
            for detail in audit["exclusion_details"]
        }
        self.assertEqual(reasons[direct_child.id], "direct_parent_child")
        self.assertEqual(reasons[sibling.id], "siblings")
        self.assertEqual(reasons[migration_clone.id], "migration_clone")
        self.assertEqual(
            reasons[common_ancestor_relative.id], "recent_common_ancestor"
        )
        self.assertEqual(reasons[top_child.id], "direct_parent_child")
        self.assertEqual(audit["lineage_depth"], 3)

    def test_pairwise_code_and_kin_filters_never_relax_to_fill(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        duplicate_a = _program("duplicate-a", code="def duplicate(): pass")
        duplicate_b = _program("duplicate-b", code="def duplicate(): pass")
        sibling_a = _program("sibling-a", parent_id="retired-parent")
        sibling_b = _program("sibling-b", parent_id="retired-parent")
        unrelated = _program("unrelated")
        _install_live(database, parent)
        for cell, owner in enumerate(
            (duplicate_a, duplicate_b, sibling_a, sibling_b, unrelated)
        ):
            _install_owner(database, owner, cell)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=5
        )

        selected_ids = {program.id for program in selected}
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(selected_ids.intersection({duplicate_a.id, duplicate_b.id})), 1)
        self.assertEqual(len(selected_ids.intersection({sibling_a.id, sibling_b.id})), 1)
        self.assertIn(unrelated.id, selected_ids)
        self.assertEqual(len(audit["identical_code_exclusions"]), 1)
        self.assertEqual(len(audit["close_relative_exclusions"]), 1)
        self.assertEqual(audit["shortfall_count"], 2)
        self.assertEqual(
            audit["shortfall_reason"],
            "insufficient_distinct_eligible_cells_after_exclusions",
        )

    def test_ultimate_founder_alone_does_not_collapse_distinct_branches(self) -> None:
        database = _database()
        founder = _program("founder")
        parent_branch = _program("parent-branch", parent_id=founder.id)
        candidate_branch = _program("candidate-branch", parent_id=founder.id)
        parent = _program("parent", island=1, parent_id=parent_branch.id)
        candidate = _program("candidate", parent_id=candidate_branch.id)

        for live_program in (founder, parent_branch, candidate_branch, parent):
            _install_live(database, live_program)
        _install_owner(database, candidate, 0)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=1
        )

        self.assertEqual([program.id for program in selected], [candidate.id])
        self.assertEqual(audit["close_relative_exclusions"], [])

    def test_requires_live_active_code_and_finite_evidence_backed_fitness(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        valid_zero = _program("valid-zero", score=0.0)
        empty_code = _program("empty-code", code="   ")
        nan_fitness = _program("nan-fitness", score=math.nan)
        infinite_fitness = _program("infinite-fitness", score=math.inf)
        missing_fitness = _program("missing-fitness", score=None)
        missing_fitness.metrics = {}
        inactive = _program("inactive")

        _install_live(database, parent)
        _install_owner(database, valid_zero, 0)
        _install_owner(database, empty_code, 1)
        _install_owner(database, nan_fitness, 2)
        _install_owner(database, infinite_fitness, 3)
        _install_owner(database, missing_fitness, 4)
        database.programs[inactive.id] = inactive
        database.island_feature_maps[0]["5"] = inactive.id
        database.island_feature_maps[0]["6"] = "missing-owner"

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=10
        )

        self.assertEqual([program.id for program in selected], [valid_zero.id])
        reasons = {
            item["program_id"]: item["reason"]
            for item in audit["ineligible_exclusions"]
        }
        self.assertEqual(reasons[empty_code.id], "unusable_source_code")
        self.assertEqual(reasons[nan_fitness.id], "non_finite_or_missing_fitness")
        self.assertEqual(
            reasons[infinite_fitness.id], "non_finite_or_missing_fitness"
        )
        self.assertEqual(
            reasons[missing_fitness.id], "non_finite_or_missing_fitness"
        )
        self.assertEqual(reasons[inactive.id], "owner_not_active_in_target_island")
        self.assertEqual(reasons["missing-owner"], "owner_not_live")

    def test_no_occupied_cells_returns_empty_with_auditable_shortfall(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        _install_live(database, parent)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=4
        )

        self.assertEqual(selected, [])
        self.assertEqual(audit["requested_count"], 4)
        self.assertEqual(audit["returned_count"], 0)
        self.assertEqual(audit["shortfall_count"], 4)
        self.assertEqual(audit["shortfall_reason"], "no_occupied_cells")

    def test_parent_owner_is_excluded_and_result_is_capped_at_requested_count(self) -> None:
        database = _database()
        parent = _program("parent", island=0)
        first = _program("first", descriptor=1.5)
        second = _program("second", descriptor=2.5)
        third = _program("third", descriptor=3.5)
        _install_owner(database, parent, 0)
        _install_owner(database, first, 1)
        _install_owner(database, second, 2)
        _install_owner(database, third, 3)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=2
        )

        self.assertEqual(len(selected), 2)
        self.assertNotIn(parent.id, {program.id for program in selected})
        self.assertEqual(audit["returned_count"], 2)
        self.assertEqual(audit["shortfall_count"], 0)
        self.assertIsNone(audit["shortfall_reason"])

    def test_neutral_superseded_former_owner_is_not_selected(self) -> None:
        database = _database()
        former = _program("former", score=0.5, descriptor=2.5)
        replacement = _program("replacement", score=0.5, descriptor=2.5)
        parent = _program("foreign-parent", island=1)

        database.add(former, target_island=0)
        database.add(replacement, target_island=0)
        database.add(parent, target_island=1)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=2
        )

        self.assertNotIn(former.id, database.programs)
        self.assertEqual([program.id for program in selected], [replacement.id])
        self.assertEqual(audit["selected_program_ids"], [replacement.id])
        self.assertEqual(
            set(database.island_feature_maps[0].values()), {replacement.id}
        )

    def test_strict_superseded_former_owner_is_not_selected(self) -> None:
        database = _database()
        former = _program("former", score=0.4, descriptor=2.5)
        replacement = _program("replacement", score=0.8, descriptor=2.5)
        parent = _program("foreign-parent", island=1)

        database.add(former, target_island=0)
        database.add(replacement, target_island=0)
        database.add(parent, target_island=1)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=2
        )

        self.assertNotIn(former.id, database.programs)
        self.assertEqual([program.id for program in selected], [replacement.id])
        self.assertEqual(audit["selected_program_ids"], [replacement.id])
        self.assertEqual(
            replacement.metadata["map_elites_replacement"]["replacement_kind"],
            "strict",
        )

    def test_migration_duplicate_code_is_excluded(self) -> None:
        database = _database()
        parent = _program("parent", code="def shared(): return 1", island=1)
        migrated_copy = _program(
            "migrated-copy",
            code=parent.code,
            parent_id=parent.id,
            metadata={"migrant": True, "migration_source_island": 1},
        )
        unrelated = _program("unrelated")
        _install_live(database, parent)
        _install_owner(database, migrated_copy, 0)
        _install_owner(database, unrelated, 1)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=2
        )

        self.assertEqual([program.id for program in selected], [unrelated.id])
        self.assertEqual(audit["identical_code_exclusions"], [migrated_copy.id])

    def test_checkpoint_round_trip_preserves_authoritative_selection(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        owner = _program("owner", descriptor=7.5)
        _install_live(database, parent)
        _install_owner(database, owner, 8)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint"
            database.save(str(checkpoint), iteration=12)
            database.config.db_path = str(checkpoint)
            resumed = ProgramDatabase(database.config)

            selected, audit = resumed.sample_true_diverse_inspirations(
                resumed.programs[parent.id], target_island=0, n=2
            )

        self.assertEqual([program.id for program in selected], [owner.id])
        self.assertEqual(audit["selected_map_cells"], {owner.id: [8]})
        self.assertEqual(audit["shortfall_count"], 1)


if __name__ == "__main__":
    unittest.main()
