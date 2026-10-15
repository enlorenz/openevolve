"""Focused tests for the opt-in database-side true-diversity selector."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

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
        self.assertEqual(audit["individually_eligible_candidate_count"], 3)
        self.assertTrue(audit["foreign_parent"])
        self.assertEqual(set(audit["selected_source_islands"].values()), {0})
        self.assertEqual(
            audit["selected_descriptor_values"][first.id], {"descriptor": 3.5}
        )
        json.dumps(audit)

    def test_commentary_only_variants_match_parent_and_top_baseline(self) -> None:
        database = _database()
        parent = _program(
            "parent",
            code=(
                "def policy():\n"
                "    # Parent explanation.\n"
                "    return 1\n"
            ),
            island=1,
        )
        top = _program(
            "top",
            code=(
                "def top_policy():\n"
                "    \"\"\"Top strategy.\"\"\"\n"
                "    return 2\n"
            ),
        )
        parent_without_comment = _program(
            "parent-without-comment",
            code="def policy():\n    return 1\n",
        )
        top_with_different_docstring = _program(
            "top-with-different-docstring",
            code=(
                "def top_policy():\n"
                "    \"\"\"A different, multi-line\n"
                "    strategy description.\n"
                "    \"\"\"\n"
                "    return 2\n"
            ),
        )
        executable_difference = _program(
            "executable-difference",
            code="def policy():\n    return 3\n",
        )

        _install_live(database, parent)
        _install_owner(database, top, 0)
        _install_owner(database, parent_without_comment, 1)
        _install_owner(database, top_with_different_docstring, 2)
        _install_owner(database, executable_difference, 3)

        selected, audit = database.sample_true_diverse_inspirations(
            parent,
            target_island=0,
            n=10,
            excluded_program_ids={top.id},
        )

        self.assertEqual(
            [program.id for program in selected], [executable_difference.id]
        )
        self.assertCountEqual(
            audit["identical_code_exclusions"],
            [parent_without_comment.id, top_with_different_docstring.id],
        )
        duplicate_of = {
            detail["program_id"]: detail["related_program_id"]
            for detail in audit["exclusion_details"]
            if detail["reason"] == "identical_code"
        }
        self.assertEqual(duplicate_of[parent_without_comment.id], parent.id)
        self.assertEqual(duplicate_of[top_with_different_docstring.id], top.id)

    def test_commentary_only_pair_conflict_is_audited_as_identical_code(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        first = _program(
            "first",
            code="def shared():\n    value = 1  # first explanation\n    return value\n",
        )
        second = _program(
            "second",
            code="def shared():\n    value = 1  # second explanation\n    return value\n",
        )
        _install_live(database, parent)
        _install_owner(database, first, 0)
        _install_owner(database, second, 1)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=2
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(audit["conflict_edge_count"], 1)
        self.assertEqual(len(audit["identical_code_exclusions"]), 1)
        excluded_id = audit["identical_code_exclusions"][0]
        self.assertIn(excluded_id, {first.id, second.id})
        self.assertEqual(
            next(
                detail["reason"]
                for detail in audit["exclusion_details"]
                if detail["program_id"] == excluded_id
            ),
            "identical_code",
        )

    def test_wildcard_cannot_restore_commentary_only_duplicate(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        with_docstring = _program(
            "with-docstring",
            code=(
                "def shared():\n"
                "    \"\"\"This commentary is removable.\"\"\"\n"
                "    return 1\n"
            ),
        )
        without_docstring = _program(
            "without-docstring",
            code="def shared():\n    return 1\n",
        )
        _install_live(database, parent)
        _install_owner(database, with_docstring, 0)
        _install_owner(database, without_docstring, 1)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=3
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(len(audit["core_selected_program_ids"]), 1)
        self.assertIsNone(audit["wildcard_selected_program_id"])
        self.assertEqual(audit["conflict_edge_count"], 1)
        self.assertEqual(len(audit["identical_code_exclusions"]), 1)
        self.assertIn(
            audit["identical_code_exclusions"][0],
            {with_docstring.id, without_docstring.id},
        )

    def test_executable_difference_remains_true_diversity_eligible(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        first = _program(
            "first",
            code="def shared():\n    # explanation A\n    return 1\n",
        )
        second = _program(
            "second",
            code="def shared():\n    # explanation B\n    return 2\n",
        )
        _install_live(database, parent)
        _install_owner(database, first, 0)
        _install_owner(database, second, 1)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=2
        )

        self.assertEqual({program.id for program in selected}, {first.id, second.id})
        self.assertEqual(audit["conflict_edge_count"], 0)
        self.assertEqual(audit["identical_code_exclusions"], [])

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

    def test_wildcard_relaxes_kinship_but_not_identical_code(self) -> None:
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
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(selected_ids.intersection({duplicate_a.id, duplicate_b.id})), 1)
        self.assertEqual(len(selected_ids.intersection({sibling_a.id, sibling_b.id})), 2)
        self.assertIn(unrelated.id, selected_ids)
        self.assertEqual(len(audit["identical_code_exclusions"]), 1)
        self.assertEqual(len(audit["core_selected_program_ids"]), 3)
        self.assertIn(
            audit["wildcard_selected_program_id"],
            {sibling_a.id, sibling_b.id},
        )
        self.assertEqual(
            audit["wildcard_conflicts_with_core"][0]["reasons"], ["siblings"]
        )
        self.assertEqual(audit["shortfall_count"], 1)
        self.assertEqual(
            audit["shortfall_reason"],
            "insufficient_distinct_eligible_cells_after_exclusions",
        )

    def test_hub_and_leaves_exact_core_avoids_greedy_shortfall(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        founder_leaf = _program("founder-leaf")
        hub = _program("hub", parent_id=founder_leaf.id)
        descendant_leaf = _program("descendant-leaf", parent_id=hub.id)
        _install_live(database, parent)
        # Preserve the adversarial hub-first order under which the former
        # random-greedy pass returned only the hub.
        for cell, owner in enumerate((hub, founder_leaf, descendant_leaf)):
            _install_owner(database, owner, cell)

        with patch("openevolve.database.random.shuffle", side_effect=lambda items: None):
            selected, audit = database.sample_true_diverse_inspirations(
                parent, target_island=0, n=2
            )
            selected_with_wildcard, wildcard_audit = (
                database.sample_true_diverse_inspirations(
                    parent, target_island=0, n=3
                )
            )

        self.assertEqual(
            {program.id for program in selected},
            {founder_leaf.id, descendant_leaf.id},
        )
        self.assertFalse(audit["wildcard_enabled"])
        self.assertEqual(audit["conflict_edge_count"], 2)
        self.assertEqual(audit["maximum_compatible_core_count_found"], 2)
        self.assertTrue(audit["exact_optimality_established"])
        self.assertEqual(
            {program.id for program in selected_with_wildcard},
            {founder_leaf.id, hub.id, descendant_leaf.id},
        )
        self.assertEqual(
            wildcard_audit["core_selected_program_ids"],
            [founder_leaf.id, descendant_leaf.id],
        )
        self.assertEqual(wildcard_audit["wildcard_selected_program_id"], hub.id)
        self.assertEqual(
            {
                conflict["core_program_id"]: conflict["reasons"]
                for conflict in wildcard_audit["wildcard_conflicts_with_core"]
            },
            {
                founder_leaf.id: ["direct_parent_child"],
                descendant_leaf.id: ["direct_parent_child"],
            },
        )

    def test_zero_and_one_requested_counts_have_no_wildcard(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        candidates = [_program(f"candidate-{index}") for index in range(3)]
        _install_live(database, parent)
        for cell, candidate in enumerate(candidates):
            _install_owner(database, candidate, cell)

        selected_zero, zero_audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=0
        )
        self.assertEqual(selected_zero, [])
        self.assertEqual(zero_audit["core_target_count"], 0)
        self.assertEqual(zero_audit["core_upper_bound"], 0)
        self.assertFalse(zero_audit["wildcard_enabled"])

        selected_one, one_audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=1
        )
        self.assertEqual(len(selected_one), 1)
        self.assertEqual(len(one_audit["core_selected_program_ids"]), 1)
        self.assertFalse(one_audit["wildcard_enabled"])
        self.assertIsNone(one_audit["wildcard_selected_program_id"])

    def test_core_and_wildcard_cardinality_policy(self) -> None:
        cases = (
            (3, 3, 2, 1),
            (5, 5, 4, 1),
            (5, 3, 2, 1),
            (5, 1, 0, 1),
        )
        for requested, candidate_count, expected_core, expected_wildcards in cases:
            with self.subTest(requested=requested, candidate_count=candidate_count):
                database = _database()
                parent = _program("parent", island=1)
                _install_live(database, parent)
                for cell in range(candidate_count):
                    _install_owner(database, _program(f"candidate-{cell}"), cell)

                selected, audit = database.sample_true_diverse_inspirations(
                    parent, target_island=0, n=requested
                )

                self.assertEqual(len(audit["core_selected_program_ids"]), expected_core)
                self.assertEqual(
                    audit["core_upper_bound"],
                    min(requested - 1, max(0, candidate_count - 1)),
                )
                self.assertTrue(audit["wildcard_enabled"])
                self.assertEqual(
                    audit["wildcard_selected_program_id"] is not None,
                    bool(expected_wildcards),
                )
                self.assertEqual(len(selected), expected_core + expected_wildcards)
                self.assertEqual(len({program.id for program in selected}), len(selected))
                self.assertNotIn(
                    audit["wildcard_selected_program_id"],
                    audit["core_selected_program_ids"],
                )
                expected_ids = list(audit["core_selected_program_ids"])
                if audit["wildcard_selected_program_id"] is not None:
                    expected_ids.append(audit["wildcard_selected_program_id"])
                    self.assertEqual(
                        audit["wildcard_selected_map_cell"],
                        audit["selected_map_cells"][
                            audit["wildcard_selected_program_id"]
                        ],
                    )
                self.assertEqual(audit["selected_program_ids"], expected_ids)
                self.assertEqual(audit["returned_count"], len(expected_ids))

    def test_wildcard_can_cross_each_lineage_conflict(self) -> None:
        for relationship in (
            "direct_parent_child",
            "siblings",
            "migration_clone",
            "recent_ancestor_descendant",
            "recent_common_ancestor",
        ):
            with self.subTest(relationship=relationship):
                database = _database()
                parent = _program("prompt-parent", island=1)
                _install_live(database, parent)

                if relationship == "direct_parent_child":
                    first = _program("first")
                    second = _program("second", parent_id=first.id)
                elif relationship == "siblings":
                    first = _program("first", parent_id="retired-parent")
                    second = _program("second", parent_id="retired-parent")
                elif relationship == "migration_clone":
                    first = _program("first")
                    second = _program(
                        "second",
                        parent_id=first.id,
                        metadata={"migrant": True, "migration_source_island": 1},
                    )
                elif relationship == "recent_ancestor_descendant":
                    root = _program("root")
                    first = _program("first", parent_id=root.id)
                    middle = _program("middle", parent_id=first.id)
                    for lineage_program in (root, middle):
                        _install_live(database, lineage_program)
                    second = _program("second", parent_id=middle.id)
                else:
                    root = _program("root")
                    shared = _program("shared", parent_id=root.id)
                    left = _program("left", parent_id=shared.id)
                    right = _program("right", parent_id=shared.id)
                    for lineage_program in (root, shared, left, right):
                        _install_live(database, lineage_program)
                    first = _program("first", parent_id=left.id)
                    second = _program("second", parent_id=right.id)

                _install_owner(database, first, 0)
                _install_owner(database, second, 1)
                selected, audit = database.sample_true_diverse_inspirations(
                    parent, target_island=0, n=3
                )

                self.assertEqual(len(selected), 2)
                self.assertEqual(len(audit["core_selected_program_ids"]), 1)
                wildcard_id = audit["wildcard_selected_program_id"]
                self.assertIn(wildcard_id, {first.id, second.id})
                self.assertNotIn(wildcard_id, audit["core_selected_program_ids"])
                self.assertNotIn(wildcard_id, audit["close_relative_exclusions"])
                self.assertNotIn(
                    wildcard_id,
                    {
                        detail["program_id"]
                        for detail in audit["exclusion_details"]
                    },
                )
                self.assertEqual(
                    audit["wildcard_conflicts_with_core"],
                    [
                        {
                            "core_program_id": audit["core_selected_program_ids"][0],
                            "reasons": [relationship],
                        }
                    ],
                )

    def test_wildcard_never_duplicates_core_source(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        first = _program("first", code="def shared(): return 1")
        duplicate = _program("duplicate", code=first.code)
        _install_live(database, parent)
        _install_owner(database, first, 0)
        _install_owner(database, duplicate, 1)

        selected, audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=3
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(len(audit["core_selected_program_ids"]), 1)
        self.assertIsNone(audit["wildcard_selected_program_id"])
        self.assertEqual(audit["conflict_edge_count"], 1)
        self.assertEqual(audit["shortfall_count"], 2)

    def test_selection_does_not_prefer_fitness_or_rank(self) -> None:
        def selected_id(scores: tuple[float, float]) -> str:
            database = _database()
            parent = _program("parent", island=1)
            _install_live(database, parent)
            _install_owner(database, _program("first", score=scores[0]), 0)
            _install_owner(database, _program("second", score=scores[1]), 1)
            random.seed(811)
            selected, _ = database.sample_true_diverse_inspirations(
                parent, target_island=0, n=1
            )
            return selected[0].id

        self.assertEqual(selected_id((0.99, 0.01)), selected_id((0.01, 0.99)))

    def test_tied_maximum_cores_are_seeded_reproducible(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        _install_live(database, parent)
        for cell in range(4):
            _install_owner(database, _program(f"candidate-{cell}"), cell)

        random.seed(991)
        first, first_audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=2
        )
        random.seed(991)
        second, second_audit = database.sample_true_diverse_inspirations(
            parent, target_island=0, n=2
        )

        self.assertEqual(
            [program.id for program in first], [program.id for program in second]
        )
        self.assertEqual(
            first_audit["core_selected_map_cells"],
            second_audit["core_selected_map_cells"],
        )
        self.assertTrue(first_audit["exact_optimality_established"])

    def test_search_budget_fallback_is_explicit_and_seeded(self) -> None:
        database = _database()
        parent = _program("parent", island=1)
        founder_leaf = _program("founder-leaf")
        hub = _program("hub", parent_id=founder_leaf.id)
        descendant_leaf = _program("descendant-leaf", parent_id=hub.id)
        _install_live(database, parent)
        for cell, candidate in enumerate((hub, founder_leaf, descendant_leaf)):
            _install_owner(database, candidate, cell)

        random.seed(0)
        with patch("openevolve.database._TRUE_DIVERSITY_EXACT_SEARCH_NODE_LIMIT", 0):
            first, first_audit = database.sample_true_diverse_inspirations(
                parent, target_island=0, n=2
            )
        random.seed(0)
        with patch("openevolve.database._TRUE_DIVERSITY_EXACT_SEARCH_NODE_LIMIT", 0):
            second, second_audit = database.sample_true_diverse_inspirations(
                parent, target_island=0, n=2
            )

        self.assertEqual(
            [program.id for program in first], [program.id for program in second]
        )
        self.assertEqual([program.id for program in first], [hub.id])
        self.assertEqual(first_audit["maximum_compatible_core_count_found"], 1)
        self.assertEqual(first_audit["core_upper_bound"], 2)
        self.assertFalse(first_audit["exact_optimality_established"])
        self.assertEqual(
            first_audit["fallback_reason"],
            "maximum_compatible_core_search_budget_exceeded",
        )
        self.assertEqual(first_audit, second_audit)

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
        self.assertEqual(audit["individually_eligible_candidate_count"], 1)
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
        self.assertFalse(audit["wildcard_enabled"])
        self.assertEqual(json.loads(json.dumps(audit)), audit)


if __name__ == "__main__":
    unittest.main()
