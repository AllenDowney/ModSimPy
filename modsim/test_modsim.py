import unittest
from modsim import *

import os

import pint
from pint.errors import UnitStrippedWarning

import warnings

warnings.simplefilter("error", Warning)


class TestModSimSeries(unittest.TestCase):
    def test_constructor(self):
        s = ModSimSeries([1, 2, 3])
        self.assertEqual(s[0], 1)

        q = Quantity(2, UNITS.meter)
        s[q] = 4
        self.assertEqual(s[q], 4)
        self.assertEqual(s[2], 4)


class TestModSimDataFrame(unittest.TestCase):
    def test_constructor(self):
        msf = ModSimDataFrame(columns=["A", "T", "dt"])
        msf.row[1000] = [1, 2, np.nan]
        msf.row["label"] = ["4", 5, 6.0]

        col = msf.A
        self.assertIsInstance(col, ModSimSeries)
        self.assertEqual(col[1000], 1)

        col = msf.T
        self.assertIsInstance(col, ModSimSeries)
        self.assertEqual(col[1000], 2)

        col = msf.dt
        self.assertIsInstance(col, ModSimSeries)
        self.assertEqual(col["label"], 6.0)

        row = msf.row[1000]
        self.assertIsInstance(row, ModSimSeries)

        self.assertEqual(row.A, 1)
        self.assertEqual(row.T, 2)
        self.assertTrue(np.isnan(row.dt))

        self.assertEqual(row["A"], 1)
        self.assertEqual(row["T"], 2)
        self.assertTrue(np.isnan(row["dt"]))


class TestTimeFrame(unittest.TestCase):
    def test_constructor(self):
        msf = TimeFrame(columns=["A", "T", "dt"])
        msf.row[1000] = [1, 2, np.nan]
        msf.row["label"] = ["4", 5, 6.0]

        col = msf.A
        self.assertIsInstance(col, TimeSeries)

        col = msf.T
        self.assertIsInstance(col, TimeSeries)

        col = msf.dt
        self.assertIsInstance(col, TimeSeries)

        row = msf.row[1000]
        self.assertIsInstance(row, State)

        row = msf.row["label"]
        self.assertIsInstance(row, State)


class TestSweepFrame(unittest.TestCase):
    def test_constructor(self):
        msf = SweepFrame(columns=["A", "T", "dt"])
        msf.row[1000] = [1, 2, np.nan]
        msf.row["label"] = ["4", 5, 6.0]

        col = msf.A
        self.assertIsInstance(col, SweepSeries)

        col = msf.T
        self.assertIsInstance(col, SweepSeries)

        col = msf.dt
        self.assertIsInstance(col, SweepSeries)

        row = msf.row[1000]
        self.assertIsInstance(row, SweepSeries)

        row = msf.row["label"]
        self.assertIsInstance(row, SweepSeries)


class TestCartPol(unittest.TestCase):
    def test_cart2pol(self):
        theta, r = cart2pol(3, 4)
        self.assertAlmostEqual(r, 5)
        self.assertAlmostEqual(theta, 0.9272952180016122)

        theta, r, z = cart2pol(2, 2, 2)
        self.assertAlmostEqual(r, 2 * np.sqrt(2))
        self.assertAlmostEqual(theta, np.pi / 4)
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
        warnings.simplefilter("error", Warning)
        array = linspace(0, 1, 11)
        self.assertEqual(len(array), 11)
        self.assertAlmostEqual(array[0], 0)
        self.assertAlmostEqual(array[1], 0.1)
        self.assertAlmostEqual(array[10], 1.0)

        array = linspace(0, 1, 10, endpoint=False)
        self.assertEqual(len(array), 10)
        self.assertAlmostEqual(array[0], 0)
        self.assertAlmostEqual(array[1], 0.1)
        self.assertAlmostEqual(array[9], 0.9)

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

        # TODO: bring back this test when np.ediff1 is fixed
        # self.assertTrue(np.isnan(abs_diff[-1]))

        series = TimeSeries(ps, index=ts)
        abs_diff = compute_abs_diff(series)
        self.assertEqual(len(abs_diff), 11)
        self.assertAlmostEqual(abs_diff[1950], 0.1)
        # self.assertTrue(np.isnan(abs_diff[1960]))

    def test_rel_diff(self):
        rel_diff = compute_rel_diff([1, 3, 7.5])
        self.assertEqual(len(rel_diff), 3)
        self.assertAlmostEqual(rel_diff[1], 1.5)

        ts = linrange(1950, 1960, endpoint=True)
        ps = linspace(3, 4, len(ts))
        rel_diff = compute_rel_diff(ps)
        self.assertEqual(len(rel_diff), 11)
        self.assertAlmostEqual(rel_diff[0], 0.0333333333)
        # self.assertTrue(np.isnan(rel_diff[-1]))

        series = TimeSeries(ps, index=ts)
        rel_diff = compute_rel_diff(series)
        self.assertEqual(len(rel_diff), 11)
        self.assertAlmostEqual(rel_diff[1950], 0.0333333333)
        # self.assertTrue(np.isnan(rel_diff[1960]))


class TestOdeSolvers(unittest.TestCase):
    def test_run_euler(self):
        s = UNITS.second
        m = UNITS.meter

        init = State(y=2 * m)
        system = System(init=init, t_0=1 * s, t_end=3 * s)

        def slope_func(state, t, system):
            [y] = state
            dydt = y / s + t * m / s ** 2
            return [dydt]

        results, details = run_euler(system, slope_func)
        y_end = get_last_value(results.y)
        self.assertAlmostEqual(y_end, 24.9737147 * m)

    def test_run_ralston(self):
        s = UNITS.second
        m = UNITS.meter

        init = State(y=2 * m)
        system = System(init=init, t_0=1 * s, t_end=3 * s)

        def slope_func(state, t, system):
            [y] = state
            dydt = y / s + t * m / s ** 2
            return [dydt]

        results, details = run_ralston(system, slope_func)
        y_end = get_last_value(results.y)
        self.assertAlmostEqual(y_end, 25.8344700133 * m)

    def test_run_solve_ivp(self):
        s = UNITS.second
        m = UNITS.meter

        init = State(y=2 * m)
        system = System(init=init, t_0=1 * s, t_end=3 * s)

        def slope_func(state, t, system):
            [y] = state
            dydt = y + t
            return [dydt]

        results, details = run_solve_ivp(system, slope_func)
        y_end = get_last_value(results.y)
        self.assertAlmostEqual(y_end, 25.5571533)


class TestRootFinders(unittest.TestCase):
    def test_root_scalar(self):
        def func(x):
            return (x - 1) * (x - 2) * (x - 3)

        res = root_scalar(func, [0, 1.9])
        self.assertAlmostEqual(res.root, 1.0)

    def test_root_secant(self):
        def func(x):
            return (x - 1) * (x - 2) * (x - 3)

        res = root_bisect(func, [0, 1.9])
        self.assertAlmostEqual(res.root, 1.0)

        res = root_bisect(func, [0, 0.5])
        self.assertFalse(res.converged)


class TestRunInterpolate(unittest.TestCase):
    def test_has_nan(self):
        a = [1, 2, 3]
        self.assertFalse(has_nan(a))
        self.assertFalse(has_nan(np.array(a)))
        self.assertFalse(has_nan(pd.Series(a)))
        a.append(np.nan)
        self.assertTrue(has_nan(a))
        self.assertTrue(has_nan(np.array(a)))
        self.assertTrue(has_nan(pd.Series(a)))

    def test_is_strictly_increasing(self):
        a = [1, 2, 3]
        self.assertTrue(is_strictly_increasing(a))
        self.assertTrue(is_strictly_increasing(np.array(a)))
        self.assertTrue(is_strictly_increasing(pd.Series(a)))
        a.append(3)
        self.assertFalse(is_strictly_increasing(a))
        self.assertFalse(is_strictly_increasing(np.array(a)))
        self.assertFalse(is_strictly_increasing(pd.Series(a)))

    def test_interpolate(self):
        index = [1, 2, 3]
        values = np.array(index) * 2 - 1
        series = pd.Series(values, index=index)
        i = interpolate(series)
        self.assertAlmostEqual(i(1.5), 2.0)

    def test_interpolate_with_units(self):
        index = [1, 2, 3]
        values = np.array(index) * 2 - 1
        series = pd.Series(values, index=index) * UNITS.meter
        i = interpolate(series)
        self.assertAlmostEqual(i(1.5), 2.0 * UNITS.meter)


class TestGradient(unittest.TestCase):
    def test_gradient(self):
        a = [1, 2, 4]
        s = TimeSeries(a)
        r = gradient(s)
        self.assertTrue(isinstance(r, TimeSeries))
        self.assertAlmostEqual(r[1], 1.5)

    def test_gradient_with_units(self):
        s = SweepSeries()
        s[0] = 1 * UNITS.meter
        s[1] = 2 * UNITS.meter
        s[2] = 4 * UNITS.meter
        r = gradient(s)
        self.assertTrue(isinstance(r, SweepSeries))
        self.assertAlmostEqual(r[1], 1.5 * UNITS.meter)


class TestGolden(unittest.TestCase):
    def test_minimize(self):
        def min_func(x, system):
            return (x - system.actual_min) ** 2

        system = System(actual_min=2)
        res = minimize_golden(min_func, [0, 5], system, rtol=1e-7)
        self.assertAlmostEqual(res.x, 2)
        self.assertAlmostEqual(res.fun, 0)

    def test_maximize(self):
        def max_func(x, system):
            return -(x - system.actual_min) ** 2

        system = System(actual_min=2)
        res = maximize_golden(max_func, [0, 5], system, rtol=1e-7)
        self.assertAlmostEqual(res.x, 2)
        self.assertAlmostEqual(res.fun, 0)


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
        self.assertEqual(get_units(x), get_units(y))
        self.assertAlmostEqual(magnitude(x), magnitude(y))

    def test_vector_mag(self):
        warnings.simplefilter("error", Warning)
        m = UNITS.meter

        v = [3, 4]
        self.assertEqual(vector_mag(v), 5)
        v = Vector(3, 4)
        self.assertEqual(vector_mag(v), 5)
        v = Vector(3, 4) * m
        self.assertEqual(vector_mag(v), 5 * m)
        self.assertEqual(v.mag, 5 * m)

    def test_vector_mag2(self):
        warnings.simplefilter("error", Warning)
        m = UNITS.meter

        v = [3, 4]
        self.assertEqual(vector_mag2(v), 25)
        v = Vector(3, 4)
        self.assertEqual(vector_mag2(v), 25)
        v = Vector(3, 4) * m
        self.assertEqual(vector_mag2(v), 25 * m * m)

    def test_vector_angle(self):
        warnings.simplefilter("error", Warning)
        m = UNITS.meter
        ans = 0.927295218
        v = [3, 4]
        self.assertAlmostEqual(vector_angle(v), ans)
        v = Vector(3, 4)
        self.assertAlmostEqual(vector_angle(v), ans)
        v = Vector(3, 4) * m
        self.assertAlmostEqual(vector_angle(v), ans)

    def test_vector_hat(self):
        warnings.simplefilter("error", Warning)
        m = UNITS.meter
        v = [3, 4]
        ans = [0.6, 0.8]
        self.assertArrayEqual(vector_hat(v), ans)

        v = Vector(3, 4)
        self.assertVectorEqual(vector_hat(v), ans)
        v = Vector(3, 4) * m
        self.assertVectorEqual(vector_hat(v), ans)

        v = [0, 0]
        ans = [0, 0]
        self.assertArrayEqual(vector_hat(v), ans)
        v = Vector(0, 0)
        self.assertVectorEqual(vector_hat(v), ans)
        v = Vector(0, 0) * m
        self.assertVectorEqual(vector_hat(v), ans)

    def test_vector_perp(self):
        warnings.simplefilter("error", Warning)
        m = UNITS.meter
        v = [3, 4]
        ans = [-4, 3]
        self.assertTrue((vector_perp(v) == ans).all())
        v = Vector(3, 4)
        self.assertTrue((vector_perp(v) == ans).all())
        v = Vector(3, 4) * m
        self.assertTrue((vector_perp(v) == ans * m).all())

    def test_vector_dot(self):
        warnings.simplefilter("error", Warning)
        m = UNITS.meter
        s = UNITS.second
        v = [3, 4]
        w = [5, 6]
        ans = 39
        self.assertAlmostEqual(vector_dot(v, w), ans)
        v = Vector(3, 4)
        self.assertAlmostEqual(vector_dot(v, w), ans)
        self.assertAlmostEqual(vector_dot(w, v), ans)

        v = Vector(3, 4) * m
        self.assertAlmostEqual(vector_dot(v, w), ans * m)
        self.assertAlmostEqual(vector_dot(w, v), ans * m)

        w = Vector(5, 6) / s
        self.assertAlmostEqual(vector_dot(v, w), ans * m / s)
        self.assertAlmostEqual(vector_dot(w, v), ans * m / s)

    def test_vector_cross_2D(self):
        warnings.simplefilter("error", Warning)
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

        v = Vector(3, 4) * m
        self.assertAlmostEqual(vector_cross(v, w), ans * m)
        self.assertAlmostEqual(vector_cross(w, v), -ans * m)

        w = Vector(5, 6) / s
        self.assertAlmostEqual(vector_cross(v, w), ans * m / s)
        self.assertAlmostEqual(vector_cross(w, v), -ans * m / s)

    def test_vector_cross_3D(self):
        warnings.simplefilter("error", Warning)
        m = UNITS.meter
        s = UNITS.second
        ans = [-2, 4, -2]

        v = [3, 4, 5]
        w = [5, 6, 7]
        self.assertArrayEqual(vector_cross(v, w), ans)
        self.assertArrayEqual(-vector_cross(w, v), ans)

        v = Vector(3, 4, 5)
        self.assertVectorEqual(vector_cross(v, w), ans)
        self.assertVectorEqual(-vector_cross(w, v), ans)

        v = Vector(3, 4, 5) * m
        self.assertVectorEqual(vector_cross(v, w), ans * m)
        self.assertVectorEqual(-vector_cross(w, v), ans * m)

        w = Vector(5, 6, 7) / s
        self.assertVectorEqual(vector_cross(v, w), ans * m / s)
        self.assertVectorEqual(-vector_cross(w, v), ans * m / s)

    def test_scalar_proj(self):
        warnings.simplefilter("error", Warning)
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

        v = Vector(3, 4) * m
        self.assertQuantityAlmostEqual(scalar_proj(v, w), ans * m)
        self.assertAlmostEqual(scalar_proj(w, v), ans2)

        w = Vector(5, 6) / s
        self.assertQuantityAlmostEqual(scalar_proj(v, w), ans * m)
        self.assertQuantityAlmostEqual(scalar_proj(w, v), ans2 / s)

    def test_vector_proj(self):
        warnings.simplefilter("error", Warning)
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

        v = Vector(3, 4) * m
        self.assertVectorAlmostEqual(vector_proj(v, w), ans * m)
        self.assertVectorAlmostEqual(vector_proj(w, v), ans2)

        w = Vector(5, 6) / s
        self.assertVectorAlmostEqual(vector_proj(v, w), ans * m)
        self.assertVectorAlmostEqual(vector_proj(w, v), ans2 / s)

    def test_vector_dist(self):
        warnings.simplefilter("error", Warning)
        m = UNITS.meter
        v = [3, 4]
        w = [6, 8]
        ans = 5
        self.assertAlmostEqual(vector_dist(v, w), ans)
        self.assertAlmostEqual(vector_dist(w, v), ans)

        v = Vector(3, 4)
        self.assertAlmostEqual(vector_dist(v, w), ans)
        self.assertAlmostEqual(vector_dist(w, v), ans)

        v = Vector(3, 4) * m
        w = Vector(6, 8) * m
        self.assertAlmostEqual(vector_dist(v, w), ans * m)
        self.assertAlmostEqual(vector_dist(w, v), ans * m)

    def test_vector_diff_angle(self):
        warnings.simplefilter("error", Warning)
        m = UNITS.meter
        v = [3, 4]
        w = [5, 6]
        ans = 0.0512371674
        self.assertAlmostEqual(vector_diff_angle(v, w), ans)
        self.assertAlmostEqual(vector_diff_angle(w, v), -ans)

        v = Vector(3, 4)
        self.assertAlmostEqual(vector_diff_angle(v, w), ans)
        self.assertAlmostEqual(vector_diff_angle(w, v), -ans)

        v = Vector(3, 4) * m
        w = Vector(5, 6) * m
        self.assertAlmostEqual(vector_diff_angle(v, w), ans)
        self.assertAlmostEqual(vector_diff_angle(w, v), -ans)


class TestSeriesCopy(unittest.TestCase):
    def test_series_copy(self):
        series = TimeSeries()
        res = series.copy()
        self.assertTrue(isinstance(res, TimeSeries))


class TestMagnitudeUnits(unittest.TestCase):
    def test_magnitudes(self):
        # scalar
        x = 5
        res = magnitudes(x)
        self.assertEqual(res, 5)
        res = magnitudes(x * UNITS.meter)
        self.assertEqual(res, 5)

        # list (result is NumPy array)
        t = [1, 2, 3]
        res = magnitudes(t)
        self.assertEqual(res, [1, 2, 3])
        res = magnitudes(t * UNITS.meter)
        self.assertTrue((res == [1, 2, 3]).all())

        # Series (result is list)
        s = ModSimSeries([1, 2, 3])
        res = magnitudes(s)
        self.assertTrue((res == [1, 2, 3]).all())
        res = magnitudes(s * UNITS.meter)
        self.assertTrue((res == [1, 2, 3]).all())

        # Quantity containing Series(result is Series)
        res = magnitudes(UNITS.meter * s)
        self.assertTrue((res == [1, 2, 3]).all())

    def test_units(self):
        # scalar
        x = 5
        res = get_units(x)
        self.assertEqual(res, 1)
        res = get_units(x * UNITS.meter)
        self.assertEqual(res, UNITS.meter)

        # list (result is list)
        t = [1, 2, 3]
        res = get_units(t)
        self.assertEqual(res, [1, 1, 1])
        res = get_units(t * UNITS.meter)
        self.assertEqual(res, UNITS.meter)

        # Series (result Series)
        s = ModSimSeries([1, 2, 3])
        res = get_units(s)
        self.assertTrue((res == [1, 1, 1]).all())

        # Series containing Quantities (result is a Series)
        res = get_units(s * UNITS.meter)
        self.assertTrue((res == [UNITS.meter] * 3).all())

        # Quantity containing Series(result is a single Unit object)
        res = get_units(UNITS.meter * s)
        self.assertEqual(res, UNITS.meter)


class TestPlot(unittest.TestCase):
    def test_plot(self):
        os.environ["QT_XKB_CONFIG_ROOT"] = "/usr/share/X11/xkb"

        t = [1, 2, 3]
        plot(t)

        t = [1, 2, 3] * UNITS.meter
        plot(t)

        x = [4, 5, 6]
        plot(x, t)

        x = [4, 5, 6] * UNITS.second
        plot(x, t)

        a = np.array(t)
        plot(a)

        x = np.array(x)
        plot(x, a)

        s = TimeSeries(t)
        plot(s)

        s = TimeSeries(t, x)
        plot(s)

        s = TimeSeries(a, x)
        plot(s)


if __name__ == "__main__":
    unittest.main()
