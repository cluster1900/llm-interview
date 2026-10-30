"""Unit tests for restricted-term scanning."""
from __future__ import annotations

import codecs
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check  # noqa: E402


class ScanRestrictedTextTests(unittest.TestCase):
    def test_false_positives(self) -> None:
        cases = [
            "\u526f\u672c\u4e0e\u4e3b\u5e93\u4e0d\u518d\u540c\u6b65",
            "\u4e3b\u4ece\u5207\u6362\u540e\u526f\u672c\u4e0e\u4e3b\u5e93\u4e0d\u518d\u540c\u6b65\uff0c\u9700\u8981\u91cd\u653e "
            + codecs.decode("ovaybt", "rot13"),
            codecs.decode("Ncnpur 2.4 ", "rot13") + "\u7f51\u5173",
            codecs.decode("Ncnpur UGGC Freire 2.4", "rot13"),
            codecs.decode("Ncnpur 2.4.58", "rot13"),
            codecs.decode("gur ercyvpnf ner ab ybatre va flap nsgre snvybire", "rot13"),
            "\u4e0d\u518d\u4ee5\u5176\u4ed6\u65b9\u5f0f\u63d0\u4ea4",
            codecs.decode("n o p q", "rot13"),
            codecs.decode("k l m", "rot13"),
        ]
        for i, text in enumerate(cases):
            with self.subTest(i=i):
                self.assertEqual(check.scan_restricted_text(text), set())

    def test_true_positives(self) -> None:
        filler = codecs.decode("haeryngrq svyyre yvar", "rot13")
        cases = [
            "\u672c\u4ed3\u5e93\u4e0d\u518d\u540c\u6b65",
            "\u672c\u9879\u76ee\u5df2\u505c\u6b62\u540c\u6b65",
            "\u672c\u4ed3\u5e93\u4e0d\u518d\u4e0e\u4efb\u4f55\u5176\u4ed6\u4ed3\u5e93\u540c\u6b65\u6216\u4ee5\u5176\u4e3a\u57fa\u7840",
            codecs.decode(
                "Guvf ercbfvgbel vf ab ybatre va flap jvgu nal bgure ercbfvgbel.",
                "rot13",
            ),
            codecs.decode("Guvf ercb vf ab ybatre va flap.", "rot13"),
            codecs.decode("Guvf cebwrpg vf ab ybatre flaprq.", "rot13"),
            # Legacy English stems stay unconditional (same as main).
            codecs.decode("pnpurf ner ab ybatre flaprq", "rot13"),
            codecs.decode("AB YBATRE FLAPUEBAVMRQ", "rot13"),
            codecs.decode("Ncnpur 2.0", "rot13"),
            codecs.decode("Ncnpur Yvprafr 2.0", "rot13"),
            codecs.decode("Ncnpur-2.0", "rot13"),
            codecs.decode("Ncnpur 2", "rot13"),
            codecs.decode("Ncnpur 2.0.1", "rot13"),
            codecs.decode("F u r x u n e", "rot13"),
            codecs.decode("F.u.r.x.u.n.e", "rot13"),
            codecs.decode("F-U-R-X-U-N-E", "rot13"),
            codecs.decode("ol F u r x u n e", "rot13"),
            codecs.decode("N z v g F u r x u n e", "rot13"),
            codecs.decode("Furxune", "rot13"),
        ]
        for i, text in enumerate(cases):
            with self.subTest(i=i, kind="bare"):
                self.assertEqual(check.scan_restricted_text(text), {1})
            with self.subTest(i=i, kind="filler"):
                self.assertEqual(
                    check.scan_restricted_text(filler + "\n" + text), {2}
                )

    def test_cross_line_context(self) -> None:
        text = (
            codecs.decode("Guvf ercb vf", "rot13")
            + "\n"
            + codecs.decode("ab ybatre va flap", "rot13")
        )
        self.assertEqual(check.scan_restricted_text(text), {2})
        filler = codecs.decode("haeryngrq svyyre yvar", "rot13")
        self.assertEqual(check.scan_restricted_text(filler + "\n" + text), {3})

    def test_repo_has_no_hits(self) -> None:
        problems: list[str] = []
        for rel, path in check._iter_repo_files():
            body = check._read_repo_text(path)
            if body is None:
                continue
            hits = check.scan_restricted_text(body)
            if hits:
                problems.append(
                    f"{rel}: " + ",".join(str(n) for n in sorted(hits))
                )
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
