import tempfile, unittest
from pathlib import Path
import pandas as pd
from prepare import audit, assign, A, B
from snooppi_utils import add_pair_ids, full_sequence_id
from snooppi_protocol import mapped_component_frame, mapped_overlap_report


class ProtocolTests(unittest.TestCase):
    def test_conflicting_reversed_pairs_and_invalid_inputs(self):
        rows = [
            ("AAAA", "CCCC", 1),
            ("CCCC", "AAAA", 0),
            ("DDDD", "EEEE", 1),
            ("EEEE", "DDDD", 1),
            ("bad[ptm]", "FFFF", 0),
        ]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            out = root / "out"
            out.mkdir()
            for name, data in [
                ("train", rows),
                ("validation", [("GGGG", "HHHH", 0)]),
                ("test", [("IIII", "KKKK", 1)]),
            ]:
                pd.DataFrame(data, columns=[A, B, "label"]).to_csv(
                    root / f"{name}.csv", index=False
                )
            f, seqs = audit(root, out)
            self.assertEqual(len(f), 3)
            reasons = (
                pd.read_csv(out / "exclusions.csv").reason.value_counts().to_dict()
            )
            self.assertEqual(
                reasons["conflicting_labels_quarantined_without_relabeling"], 2
            )
            self.assertEqual(reasons["consistent_full_pair_duplicate"], 1)
            self.assertEqual(reasons["invalid_protein_sequence"], 1)

    def test_both_partners_and_truncated_inputs_link_components(self):
        x = "A" * 1024
        f = add_pair_ids(
            pd.DataFrame(
                [(x + "C", "GGGG", 1), (x + "D", "HHHH", 0), ("HHHH", "IIII", 1)],
                columns=[A, B, "label"],
            )
        )
        seqs = set(f[A]) | set(f[B])
        m = pd.DataFrame(
            [
                {"sequence_sha256": full_sequence_id(s), "cluster_id": str(i)}
                for i, s in enumerate(sorted(seqs))
            ]
        )
        linked = mapped_component_frame(f, m, "cluster_id")
        self.assertEqual(linked.mapped_component.nunique(), 1)

    def test_assignment_keeps_components_and_both_classes(self):
        f = pd.DataFrame(
            [
                {"mapped_component": str(i), "label": y}
                for i in range(100)
                for y in (0, 1)
            ]
        )
        parts, _ = assign(f)
        self.assertEqual([len(v) for v in parts.values()], [160, 20, 20])
        for n, v in parts.items():
            self.assertEqual(set(v.label), {0, 1})
            for other, w in parts.items():
                if n != other:
                    self.assertFalse(set(v.mapped_component) & set(w.mapped_component))

    def test_edge_fallback_is_disjoint_and_accounts_for_every_row(self):
        import numpy as np
        from finish import edge_partition

        rng = np.random.default_rng(7)
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        seqs = [
            "".join(alphabet[(i // 20**j) % 20] for j in range(4)) + "ACDEFGHIK"
            for i in range(500)
        ]
        edges = set()
        while len(edges) < 6000:
            a, b = sorted(rng.choice(500, size=2, replace=False))
            edges.add((a, b))
        f = add_pair_ids(
            pd.DataFrame(
                [(seqs[a], seqs[b], i % 2) for i, (a, b) in enumerate(sorted(edges))],
                columns=[A, B, "label"],
            )
        )
        m = pd.DataFrame(
            [
                {"sequence_sha256": full_sequence_id(s), "cluster_id": str(i)}
                for i, s in enumerate(seqs)
            ]
        )
        parts, excluded, assignment, trial = edge_partition(f, m)
        self.assertEqual(sum(map(len, parts.values())) + len(excluded), len(f))
        self.assertFalse(
            mapped_overlap_report(parts, m, "cluster_id").overlapping_cluster_id.any()
        )
        for v in parts.values():
            self.assertGreaterEqual(v.label.value_counts().min(), 50)


if __name__ == "__main__":
    unittest.main()
