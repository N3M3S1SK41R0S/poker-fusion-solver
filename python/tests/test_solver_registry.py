"""Tests du registre de benchmark — cohérence interne, invariants, ICM à jour."""

from __future__ import annotations

import unittest

from pfs.bench.solver_registry import (
    PARAMETERS,
    SOLVERS,
    Category,
    Coverage,
    coverage_report,
    coverage_score,
    differentiators,
    gaps,
    library_only,
)


class TestRegistryIntegrity(unittest.TestCase):
    def test_parameter_count(self) -> None:
        self.assertEqual(len(PARAMETERS), 84)

    def test_unique_names(self) -> None:
        names = [p.name for p in PARAMETERS]
        self.assertEqual(len(names), len(set(names)))

    def test_every_category_populated(self) -> None:
        cats = {p.category for p in PARAMETERS}
        self.assertEqual(cats, set(Category))

    def test_covered_parameters_have_module(self) -> None:
        for p in PARAMETERS:
            if p.coverage is Coverage.COVERED:
                self.assertTrue(p.module, f"{p.name} couvert sans module")

    def test_covered_modules_resolve(self) -> None:
        """Chaque pointeur de module d'un paramètre COUVERT doit exister.

        Le champ ``module`` est un pointeur court (« range_model »,
        « geometry.fisher_rao_distance ») : on le résout contre les espaces
        de noms du paquet, en descendant les suffixes pointés comme attributs.
        Un pointeur qui ne résout pas = mensonge dans le benchmark.
        """
        import importlib

        namespaces = ("pfs", "pfs.core", "pfs.fusion", "pfs.solver",
                      "pfs.data", "pfs.train", "pfs.app", "pfs.bench")

        def resolves(pointer: str) -> bool:
            parts = pointer.split(".")
            for ns in namespaces:
                for cut in range(len(parts), 0, -1):
                    modpath = ".".join([ns, *parts[:cut]])
                    try:
                        obj = importlib.import_module(modpath)
                    except ModuleNotFoundError:
                        continue
                    try:  # suffixes restants = attributs (fonction, constante)
                        for attr in parts[cut:]:
                            obj = getattr(obj, attr)
                    except AttributeError:
                        return False
                    return True
            return False

        for p in PARAMETERS:
            if p.coverage is Coverage.COVERED:
                self.assertTrue(resolves(p.module),
                                f"pointeur introuvable : {p.module!r} ({p.name})")

    def test_report_totals(self) -> None:
        rep = coverage_report()
        self.assertEqual(sum(r["total"] for r in rep.values()), 84)
        for row in rep.values():
            self.assertEqual(
                row["total"],
                row["couvert"] + row["partiel"] + row["absent"] + row["planifié"],
            )


class TestCoverageState(unittest.TestCase):
    def test_global_score_above_three_quarters(self) -> None:
        self.assertGreaterEqual(coverage_score()["score"], 0.90)

    def test_industry_score_above_eighty(self) -> None:
        self.assertGreaterEqual(coverage_score()["industry_score"], 0.95)

    def test_icm_now_covered(self) -> None:
        """Le gap n°1 du premier passage (ICM) doit être clos."""
        icm_params = [p for p in PARAMETERS if p.category is Category.ICM]
        covered = [p for p in icm_params if p.coverage is Coverage.COVERED]
        self.assertGreaterEqual(len(covered), 4)
        self.assertTrue(any("Malmuth-Harville" in p.name for p in covered))
        # L4/L5 : plus AUCUNE lacune ICM (FGS = partiel, PKO = couvert)
        remaining = {p.name for p in gaps() if p.category is Category.ICM}
        self.assertEqual(remaining, set())

    def test_no_gap_exposed_by_four_plus_solvers(self) -> None:
        """Aucune lacune ne doit être un standard de l'industrie (≥4 solveurs)."""
        for g in gaps():
            self.assertLess(len(g.exposed_by), 4,
                            f"{g.name} est exposé par {len(g.exposed_by)} solveurs")

    def test_differentiators_all_covered_and_unexposed(self) -> None:
        d = differentiators()
        self.assertGreaterEqual(len(d), 14)
        for p in d:
            self.assertIs(p.coverage, Coverage.COVERED)
            self.assertEqual(len(p.exposed_by), 0)

    def test_adversary_category_fully_covered(self) -> None:
        """La catégorie I (adversaire) est notre monopole : 100 % couverte."""
        adv = [p for p in PARAMETERS if p.category is Category.OPPONENT]
        self.assertEqual(len(adv), 12)   # +tells temporels, +population (L2, L6)
        for p in adv:
            self.assertIs(p.coverage, Coverage.COVERED, p.name)


class TestWiring(unittest.TestCase):
    """Axe ``wired`` — ajouté par l'audit du 14 août 2026.

    ``coverage`` dit « le code existe et il est testé » ; ``wired`` dit
    « un chemin d'exécution utilisateur y passe ». Le registre laissait
    croire que les deux coïncidaient : ces tests figent la liste des
    modules-bibliothèques pour que tout branchement (ou débranchement)
    futur DOIVE mettre le registre à jour.
    """

    #: Modules implémentés + testés mais importés par aucun code applicatif
    #: (vérifié par grep des imports hors tests, audit du 14 août 2026).
    #: ``fusion.eqr`` retiré le 14 août 2026 : branché dans /api/postflop
    #: (``leaf_model="eqr"``, entraînement mémoïsé dans pfs/app/server.py).
    ORPHANS = {
        "core.bunching", "solver.abstraction", "solver.isomorphism",
        "topology", "fusion.timing", "data.population",
    }

    def test_wired_false_iff_module_orphelin(self) -> None:
        for p in PARAMETERS:
            self.assertEqual(
                not p.wired, p.module in self.ORPHANS,
                f"{p.name} : wired={p.wired} mais module={p.module!r}")

    def test_library_only_liste_les_sept_entrees(self) -> None:
        # huit à l'audit du 14 août 2026 ; EQR branché le même jour → sept.
        lib = library_only()
        self.assertEqual(len(lib), 7)
        for p in lib:
            self.assertTrue(p.module, f"{p.name} en bibliothèque sans module")
            self.assertIn(p.coverage, (Coverage.COVERED, Coverage.PARTIAL))

    def test_les_absents_ne_sont_pas_comptes_bibliotheque(self) -> None:
        for p in library_only():
            self.assertIsNot(p.coverage, Coverage.ABSENT)


class TestSolverProfiles(unittest.TestCase):
    def test_six_profiles(self) -> None:
        self.assertEqual(len(SOLVERS), 6)

    def test_ours_is_last_and_free(self) -> None:
        ours = SOLVERS[-1]
        self.assertIn("Fusion", ours.name)
        self.assertTrue(ours.price_eur.startswith("0"))
        self.assertTrue(ours.icm)                # F14 livré
        self.assertTrue(ours.opponent_modeling)  # monopole
        self.assertTrue(ours.local)

    def test_no_commercial_solver_models_opponents(self) -> None:
        """Le fait central du benchmark : personne d'autre ne modélise l'adversaire."""
        for s in SOLVERS[:-1]:
            self.assertFalse(s.opponent_modeling, s.name)


if __name__ == "__main__":
    unittest.main()
