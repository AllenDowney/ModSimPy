import unittest
from modsim import *

class TestModSimDataFrame(unittest.TestCase):

    def test_constructor(self):
        msf = ModSimDataFrame(columns=['A', 'T', 'dt'])
        msf.row[1000] = [1, 2, np.nan]
        msf.row['label'] = ['4', 5, 6.0]

        col = msf.A
        self.assertIsInstance(col, ModSimSeries)
        self.assertEqual(col[1000], 1)

        col = msf.T
        self.assertIsInstance(col, ModSimSeries)
        self.assertEqual(col[1000], 2)

        col = msf.dt
        self.assertIsInstance(col, ModSimSeries)
        self.assertEqual(col['label'], 6.0)

        row = msf.row[1000]
        self.assertIsInstance(row, ModSimSeries)

        self.assertEqual(row.A, 1)
        self.assertEqual(row.T, 2)
        self.assertTrue(np.isnan(row.dt))

        self.assertEqual(row['A'], 1)
        self.assertEqual(row['T'], 2)
        self.assertTrue(np.isnan(row['dt']))

class TestTimeFrame(unittest.TestCase):

    def test_constructor(self):
        msf = TimeFrame(columns=['A', 'T', 'dt'])
        msf.row[1000] = [1, 2, np.nan]
        msf.row['label'] = ['4', 5, 6.0]

        col = msf.A
        self.assertIsInstance(col, TimeSeries)

        col = msf.T
        self.assertIsInstance(col, TimeSeries)

        col = msf.dt
        self.assertIsInstance(col, TimeSeries)

        row = msf.row[1000]
        self.assertIsInstance(row, State)

        row = msf.row['label']
        self.assertIsInstance(row, State)

class TestSweepFrame(unittest.TestCase):

    def test_constructor(self):
        msf = SweepFrame(columns=['A', 'T', 'dt'])
        msf.row[1000] = [1, 2, np.nan]
        msf.row['label'] = ['4', 5, 6.0]

        col = msf.A
        self.assertIsInstance(col, SweepSeries)

        col = msf.T
        self.assertIsInstance(col, SweepSeries)

        col = msf.dt
        self.assertIsInstance(col, SweepSeries)

        row = msf.row[1000]
        self.assertIsInstance(row, SweepSeries)

        row = msf.row['label']
        self.assertIsInstance(row, SweepSeries)

class TestCartPol(unittest.TestCase):

    def test_cart2pol(self):
        theta, r = cart2pol(3, 4)
        self.assertAlmostEqual(r, 5)
        self.assertAlmostEqual(theta, 0.9272952180016122)

        theta, r, z = cart2pol(2, 2, 2)
        self.assertAlmostEqual(r, 2 * np.sqrt(2))
        self.assertAlmostEqual(theta, np.pi/4)
        self.assertAlmostEqual(z, 2)

    def test_pol2cart(self):
        theta = 0.9272952180016122
        r = 5
        x, y = pol2cart(theta, r)
        self.assertAlmostEqual(x, 3)
        self.assertAlmostEqual(y, 4)

        angle = 45 * UNITS.degree
        r = 2 * np.sqrt(2)
        z = 2
        x, y, z = pol2cart(angle, r, z)
        self.assertAlmostEqual(x, 2)
        self.assertAlmostEqual(y, 2)
        self.assertAlmostEqual(z, 2)

class TestLinspaceLinRange(unittest.TestCase):

    def test_linspace(self):
        array = linspace(0, 1, 11)
        self.assertEqual(len(array), 11)
        self.assertAlmostEqual(array[0], 0)
        self.assertAlmostEqual(array[1], 0.1)
        self.assertAlmostEqual(array[10], 1.0)

        meter = UNITS.meter
        start = 11
        stop = 13 * meter
        array = linspace(start, stop, 37)
        self.assertEqual(len(array), 37)
        self.assertAlmostEqual(array[0], 11 * meter)
        self.assertAlmostEqual(array[1], 11.055555555555555 * meter)
        self.assertAlmostEqual(array[36], 13 * meter)

    def test_linrange(self):
        array = linrange(0, 1, 0.1)
        self.assertAlmostEqual(array[0], 0)
        self.assertAlmostEqual(array[1], 0.1)
        self.assertAlmostEqual(array[10], 1.0)

        meter = UNITS.meter
        start = 11 * meter
        stop = 13 * meter
        step = 0.2 * meter
        array = linrange(start, stop, step)

        self.assertEqual(len(array), 11)
        self.assertAlmostEqual(array[0], 11 * meter)
        self.assertAlmostEqual(array[1], 11.2 * meter)
        self.assertAlmostEqual(array[10], 13 * meter)

class TestAbsRelDiff(unittest.TestCase):

    def test_abs_diff(self):
        abs_diff = compute_abs_diff([1, 3, 7.5])
        self.assertEqual(len(abs_diff), 3)
        self.assertAlmostEqual(abs_diff[1], 4.5)

        ts = linrange(1950, 1960)
        ps = linspace(3, 4, len(ts))
        abs_diff = compute_abs_diff(ps)
        self.assertEqual(len(abs_diff), 11)
        self.assertAlmostEqual(abs_diff[1], 0.1)
        self.assertTrue(np.isnan(abs_diff[-1]))

        series = TimeSeries(ps, ts)
        abs_diff = compute_abs_diff(series)
        self.assertEqual(len(abs_diff), 11)
        self.assertAlmostEqual(abs_diff[1950], 0.1)
        self.assertTrue(np.isnan(abs_diff[1960]))

    def test_rel_diff(self):
        rel_diff = compute_rel_diff([1, 3, 7.5])
        self.assertEqual(len(rel_diff), 3)
        self.assertAlmostEqual(rel_diff[1], 1.5)

        ts = linrange(1950, 1960)
        ps = linspace(3, 4, len(ts))
        rel_diff = compute_rel_diff(ps)
        self.assertEqual(len(rel_diff), 11)
        self.assertAlmostEqual(rel_diff[0], 0.0333333333)
        self.assertTrue(np.isnan(rel_diff[-1]))

        series = TimeSeries(ps, ts)
        rel_diff = compute_rel_diff(series)
        self.assertEqual(len(rel_diff), 11)
        self.assertAlmostEqual(rel_diff[1950], 0.0333333333)
        self.assertTrue(np.isnan(rel_diff[1960]))


if __name__ == '__main__':
    unittest.main()
