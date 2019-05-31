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
        self.assertEqual(len(array), 10)
        self.assertAlmostEqual(array[0], 0)
        self.assertAlmostEqual(array[1], 0.1)
        self.assertAlmostEqual(array[9], 0.9)

        array = linrange(0, 1, 0.1, endpoint=True)
        self.assertEqual(len(array), 11)
        self.assertAlmostEqual(array[0], 0)
        self.assertAlmostEqual(array[1], 0.1)
        self.assertAlmostEqual(array[10], 1.0)

        meter = UNITS.meter
        start = 11 * meter
        stop = 13 * meter
        step = 0.2 * meter
        array = linrange(start, stop, step, endpoint=True)

        self.assertEqual(len(array), 11)
        self.assertAlmostEqual(array[0], 11 * meter)
        self.assertAlmostEqual(array[1], 11.2 * meter)
        self.assertAlmostEqual(magnitude(array[10]), 13)

class TestAbsRelDiff(unittest.TestCase):

    def test_abs_diff(self):
        abs_diff = compute_abs_diff([1, 3, 7.5])
        self.assertEqual(len(abs_diff), 3)
        self.assertAlmostEqual(abs_diff[1], 4.5)

        ts = linrange(1950, 1960, endpoint=True)
        ps = linspace(3, 4, len(ts))
        abs_diff = compute_abs_diff(ps)
        self.assertEqual(len(abs_diff), 11)
        self.assertAlmostEqual(abs_diff[1], 0.1)
        self.assertTrue(np.isnan(abs_diff[-1]))

        series = TimeSeries(ps, index=ts)
        abs_diff = compute_abs_diff(series)
        self.assertEqual(len(abs_diff), 11)
        self.assertAlmostEqual(abs_diff[1950], 0.1)
        self.assertTrue(np.isnan(abs_diff[1960]))

    def test_rel_diff(self):
        rel_diff = compute_rel_diff([1, 3, 7.5])
        self.assertEqual(len(rel_diff), 3)
        self.assertAlmostEqual(rel_diff[1], 1.5)

        ts = linrange(1950, 1960, endpoint=True)
        ps = linspace(3, 4, len(ts))
        rel_diff = compute_rel_diff(ps)
        self.assertEqual(len(rel_diff), 11)
        self.assertAlmostEqual(rel_diff[0], 0.0333333333)
        self.assertTrue(np.isnan(rel_diff[-1]))

        series = TimeSeries(ps, index=ts)
        rel_diff = compute_rel_diff(series)
        self.assertEqual(len(rel_diff), 11)
        self.assertAlmostEqual(rel_diff[1950], 0.0333333333)
        self.assertTrue(np.isnan(rel_diff[1960]))

class TestRunOdeint(unittest.TestCase):

    def test_run_ideint(self):
        pass

class TestVector(unittest.TestCase):
    def assertArrayEqual(self, res, ans):
        self.assertTrue(isinstance(res, np.ndarray))
        self.assertTrue((res == ans).all())

    def assertVectorEqual(self, res, ans):
        self.assertTrue(isinstance(res, ModSimVector))
        self.assertTrue((res == ans).all())

    def assertVectorAlmostEqual(self, res, ans):
        [self.assertQuantityAlmostEqual(x, y) for x, y in zip(res, ans)]

    def assertQuantityAlmostEqual(self, x, y):
        self.assertEqual(units(x), units(y))
        self.assertAlmostEqual(magnitude(x), magnitude(y))

    def test_vector_mag(self):
        m = UNITS.meter

        v = [3, 4]
        self.assertEqual(vector_mag(v), 5)
        v = Vector(3, 4)
        self.assertEqual(vector_mag(v), 5)
        v = Vector(3, 4)*m
        self.assertEqual(vector_mag(v), 5*m)

    def test_vector_mag2(self):
        m = UNITS.meter

        v = [3, 4]
        self.assertEqual(vector_mag2(v), 25)
        v = Vector(3, 4)
        self.assertEqual(vector_mag2(v), 25)
        v = Vector(3, 4)*m
        self.assertEqual(vector_mag2(v), 25*m*m)

    def test_vector_angle(self):
        m = UNITS.meter
        ans = 0.927295218
        v = [3, 4]
        self.assertAlmostEqual(vector_angle(v), ans)
        v = Vector(3, 4)
        self.assertAlmostEqual(vector_angle(v), ans)
        v = Vector(3, 4)*m
        self.assertAlmostEqual(vector_angle(v), ans)

    def test_vector_hat(self):
        m = UNITS.meter
        v = [3, 4]
        ans = [0.6, 0.8]
        self.assertArrayEqual(vector_hat(v), ans)

        v = Vector(3, 4)
        self.assertVectorEqual(vector_hat(v), ans)
        v = Vector(3, 4)*m
        self.assertVectorEqual(vector_hat(v), ans)

        v = [0, 0]
        ans = [0, 0]
        self.assertArrayEqual(vector_hat(v), ans)
        v = Vector(0, 0)
        self.assertVectorEqual(vector_hat(v), ans)
        v = Vector(0, 0)*m
        self.assertVectorEqual(vector_hat(v), ans)

    def test_vector_perp(self):
        m = UNITS.meter
        v = [3, 4]
        ans = [-4, 3]
        self.assertTrue((vector_perp(v) == ans).all())
        v = Vector(3, 4)
        self.assertTrue((vector_perp(v) == ans).all())
        v = Vector(3, 4)*m
        self.assertTrue((vector_perp(v) == ans*m).all())

    def test_vector_dot(self):
        m = UNITS.meter
        s = UNITS.second
        v = [3, 4]
        w = [5, 6]
        ans = 39
        self.assertAlmostEqual(vector_dot(v, w), ans)
        v = Vector(3, 4)
        self.assertAlmostEqual(vector_dot(v, w), ans)
        self.assertAlmostEqual(vector_dot(w, v), ans)

        v = Vector(3, 4)*m
        self.assertAlmostEqual(vector_dot(v, w), ans*m)
        self.assertAlmostEqual(vector_dot(w, v), ans*m)

        w = Vector(5, 6)/s
        self.assertAlmostEqual(vector_dot(v, w), ans*m/s)
        self.assertAlmostEqual(vector_dot(w, v), ans*m/s)

    def test_vector_cross_2D(self):
        m = UNITS.meter
        s = UNITS.second
        ans = -2

        v = [3, 4]
        w = [5, 6]
        self.assertAlmostEqual(vector_cross(v, w), ans)
        self.assertAlmostEqual(vector_cross(w, v), -ans)

        v = Vector(3, 4)
        self.assertAlmostEqual(vector_cross(v, w), ans)
        self.assertAlmostEqual(vector_cross(w, v), -ans)

        v = Vector(3, 4)*m
        self.assertAlmostEqual(vector_cross(v, w), ans*m)
        self.assertAlmostEqual(vector_cross(w, v), -ans*m)

        w = Vector(5, 6)/s
        self.assertAlmostEqual(vector_cross(v, w), ans*m/s)
        self.assertAlmostEqual(vector_cross(w, v), -ans*m/s)

    def test_vector_cross_3D(self):
        m = UNITS.meter
        s = UNITS.second
        ans = [-2,  4, -2]

        v = [3, 4, 5]
        w = [5, 6, 7]
        self.assertArrayEqual(vector_cross(v, w), ans)
        self.assertArrayEqual(-vector_cross(w, v), ans)

        v = Vector(3, 4, 5)
        self.assertVectorEqual(vector_cross(v, w), ans)
        self.assertVectorEqual(-vector_cross(w, v), ans)

        v = Vector(3, 4, 5)*m
        self.assertVectorEqual(vector_cross(v, w), ans*m)
        self.assertVectorEqual(-vector_cross(w, v), ans*m)

        w = Vector(5, 6, 7)/s
        self.assertVectorEqual(vector_cross(v, w), ans*m/s)
        self.assertVectorEqual(-vector_cross(w, v), ans*m/s)

    def test_scalar_proj(self):
        m = UNITS.meter
        s = UNITS.second
        ans = 4.9934383
        ans2 = 7.8

        v = [3, 4]
        w = [5, 6]
        self.assertAlmostEqual(scalar_proj(v, w), ans)
        self.assertAlmostEqual(scalar_proj(w, v), ans2)

        v = Vector(3, 4)
        self.assertAlmostEqual(scalar_proj(v, w), ans)
        self.assertAlmostEqual(scalar_proj(w, v), ans2)

        v = Vector(3, 4)*m
        self.assertQuantityAlmostEqual(scalar_proj(v, w), ans*m)
        self.assertAlmostEqual(scalar_proj(w, v), ans2)

        w = Vector(5, 6)/s
        self.assertQuantityAlmostEqual(scalar_proj(v, w), ans*m)
        self.assertQuantityAlmostEqual(scalar_proj(w, v), ans2/s)

    def test_vector_proj(self):
        m = UNITS.meter
        s = UNITS.second
        ans = [3.19672131, 3.83606557]
        ans2 = Quantity([4.68, 6.24])

        v = [3, 4]
        w = [5, 6]
        self.assertVectorAlmostEqual(vector_proj(v, w), ans)
        self.assertVectorAlmostEqual(vector_proj(w, v), ans2)

        v = Vector(3, 4)
        self.assertVectorAlmostEqual(vector_proj(v, w), ans)
        self.assertVectorAlmostEqual(vector_proj(w, v), ans2)

        v = Vector(3, 4)*m
        self.assertVectorAlmostEqual(vector_proj(v, w), ans*m)
        self.assertVectorAlmostEqual(vector_proj(w, v), ans2)

        w = Vector(5, 6)/s
        self.assertVectorAlmostEqual(vector_proj(v, w), ans*m)
        self.assertVectorAlmostEqual(vector_proj(w, v), ans2/s)

    def test_vector_dist(self):
        m = UNITS.meter
        v = [3, 4]
        w = [6, 8]
        ans = 5
        self.assertAlmostEqual(vector_dist(v, w), ans)
        self.assertAlmostEqual(vector_dist(w, v), ans)

        v = Vector(3, 4)
        self.assertAlmostEqual(vector_dist(v, w), ans)
        self.assertAlmostEqual(vector_dist(w, v), ans)

        v = Vector(3, 4)*m
        w = Vector(6, 8)*m
        self.assertAlmostEqual(vector_dist(v, w), ans*m)
        self.assertAlmostEqual(vector_dist(w, v), ans*m)

    def test_vector_diff_angle(self):
        m = UNITS.meter
        v = [3, 4]
        w = [5, 6]
        ans = 0.0512371674
        self.assertAlmostEqual(vector_diff_angle(v, w), ans)
        self.assertAlmostEqual(vector_diff_angle(w, v), -ans)

        v = Vector(3, 4)
        self.assertAlmostEqual(vector_diff_angle(v, w), ans)
        self.assertAlmostEqual(vector_diff_angle(w, v), -ans)

        v = Vector(3, 4)*m
        w = Vector(5, 6)*m
        self.assertAlmostEqual(vector_diff_angle(v, w), ans)
        self.assertAlmostEqual(vector_diff_angle(w, v), -ans)


class TestSeriesCopy(unittest.TestCase):
    def test_series_copy(self):
        series = TimeSeries()
        res = series.copy()
        #print(type(res))
        self.assertTrue(isinstance(res, TimeSeries))

if __name__ == '__main__':
    unittest.main()
