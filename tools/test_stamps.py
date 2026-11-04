"""Tests for tools/stamps.py (no GPU): python3 -m unittest tools/test_stamps.py"""
import io
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stamps  # noqa: E402


def dump(path, wgs, waves, slots, rows):
    vals = [v for r in rows for v in r]
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", wgs, waves, slots, stamps.MAGIC))
        f.write(struct.pack(f"<{len(vals)}I", *vals))


class StampTests(unittest.TestCase):
    def test_wrapping_interval_and_unwritten_slots(self):
        self.assertEqual(stamps.interval(0xFFFFFFF0, 0x10), 0x20)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.u32")
            dump(p, 1, 2, 3, [[100, 150, 400], [200, 0, 700]])   # wave 1 skipped slot 1
            wgs, waves, slots, vals = stamps.load(p)
            st = stamps.per_slot(wgs, waves, slots, vals)
            self.assertEqual(st[1], [50])
            self.assertEqual(st[2], [250])

    def test_bad_magic_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x")
            open(p, "wb").write(struct.pack("<4I", 1, 1, 1, 7) + struct.pack("<I", 1))
            with self.assertRaises(SystemExit):
                stamps.load(p)

    def test_tile_report_finds_steady_state_ramp_and_bubbles(self):
        # 2 waves, slot 0 = kernel start, slots 1..12 = tile boundaries; tile time 100, first tile 300 (ramp), tile 7 has a 500 bubble
        rows = []
        for w in range(2):
            t = 1000 + 5 * w
            r = [t]
            for k in range(1, 13):
                t += 300 if k == 1 else (500 if k == 7 else 100)
                r.append(t)
            rows.append(r)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.u32")
            dump(p, 1, 2, 13, rows)
            out = io.StringIO()
            with redirect_stdout(out):
                stamps.main([p, "--tiles", "0:12", "--phases", "1:prologue"])
            text = out.getvalue()
            self.assertIn("steady-state median 100 cycles", text)
            self.assertIn("first tile 300 (3.00x)", text)
            self.assertIn("tiles slower than 1.15 x steady: 1 (300), 7 (500)", text)
            self.assertIn("wave-to-wave spread", text)

    def test_phase_names_label_the_ending_slot(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.u32")
            dump(p, 1, 1, 3, [[10, 30, 100]])
            out = io.StringIO()
            with redirect_stdout(out):
                stamps.main([p, "--phases", "1:prologue,2:loop"])
            self.assertIn("prologue", out.getvalue())
            self.assertIn("median         70", out.getvalue())


if __name__ == "__main__":
    unittest.main()