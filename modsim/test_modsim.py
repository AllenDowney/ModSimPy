import unittest
from modsim import *
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("error", Warning)


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

        angle = np.pi/4  # 45 degrees in radians
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
        self.assertEqual(len(array), 11)
        self.assertAlmostEqual(array[0], 0)
        self.assertAlmostEqual(array[1], 0.1)
        self.assertAlmostEqual(array[9], 0.9)


class TestOdeSolvers(unittest.TestCase):
    def test_run_solve_ivp(self):
        init = State(y=2)
        system = System(init=init, t_0=1, t_end=3)

        def slope_func(t, state, system):
            y = state[0] if isinstance(state, np.ndarray) else state.iloc[0]
            dydt = y + t
            return [dydt]

        results, details = run_solve_ivp(system, slope_func)
        y_end = results.y.iloc[-1]
        self.assertAlmostEqual(y_end, 25.5571533)


class TestRootFinders(unittest.TestCase):
    def test_root_scalar(self):
        def func(x):
            return (x - 1) * (x - 2) * (x - 3)

        res = root_scalar(func, bracket=[0, 1.9])
        self.assertAlmostEqual(res.root, 1.0, places=5)

    def test_minimize_scalar(self):
        def func(x):
            return (x - 2)**2 + 1

        # Test with bracket
        res = minimize_scalar(func, bracket=[0, 4])
        self.assertAlmostEqual(res.x, 2.0, places=5)
        self.assertAlmostEqual(res.fun, 1.0, places=5)

        # Test with bounds
        res = minimize_scalar(func, bounds=[0, 4])
        self.assertAlmostEqual(res.x, 2.0, places=5)
        self.assertAlmostEqual(res.fun, 1.0, places=5)

    def test_maximize_scalar(self):
        def func(x):
            return -(x - 2)**2 + 1

        # Test with bracket
        res = maximize_scalar(func, bracket=[0, 4])
        self.assertAlmostEqual(res.x, 2.0, places=5)
        self.assertAlmostEqual(res.fun, 1.0, places=5)

        # Test with bounds
        res = maximize_scalar(func, bounds=[0, 4])
        self.assertAlmostEqual(res.x, 2.0, places=5)
        self.assertAlmostEqual(res.fun, 1.0, places=5)


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


class TestGradient(unittest.TestCase):
    def test_gradient(self):
        a = [1, 2, 4]
        s = TimeSeries(a)
        r = gradient(s)
        self.assertTrue(isinstance(r, pd.Series))
        self.assertAlmostEqual(r[1], 1.5)


class TestVector(unittest.TestCase):
    def assertArrayEqual(self, res, ans):
        self.assertTrue(isinstance(res, (np.ndarray, pd.Series)))
        self.assertTrue((res == ans).all())

    def assertVectorEqual(self, res, ans):
        self.assertTrue(isinstance(res, (pd.Series, np.ndarray)))
        self.assertTrue((res == ans).all())

    def assertVectorAlmostEqual(self, res, ans):
        for x, y in zip(res, ans):
            self.assertAlmostEqual(x, y)

    def test_vector_mag(self):
        v = [3, 4]
        self.assertEqual(vector_mag(v), 5)
        v = Vector(3, 4)
        self.assertEqual(vector_mag(v), 5)

    def test_vector_mag2(self):

        v = [3, 4]
        self.assertEqual(vector_mag2(v), 25)
        v = Vector(3, 4)
        self.assertEqual(vector_mag2(v), 25)

    def test_vector_angle(self):
        ans = 0.927295218
        v = [3, 4]
        self.assertAlmostEqual(vector_angle(v), ans)
        v = Vector(3, 4)
        self.assertAlmostEqual(vector_angle(v), ans)

    def test_vector_hat(self):
        v = [3, 4]
        ans = [0.6, 0.8]
        self.assertVectorAlmostEqual(vector_hat(v), ans)

        v = Vector(3, 4)
        self.assertVectorAlmostEqual(vector_hat(v), ans)

        v = [0, 0]
        ans = [0, 0]
        self.assertVectorAlmostEqual(vector_hat(v), ans)
        v = Vector(0, 0)
        self.assertVectorAlmostEqual(vector_hat(v), ans)

    def test_vector_perp(self):
        v = [3, 4]
        ans = [-4, 3]
        self.assertTrue((vector_perp(v) == ans).all())
        v = Vector(3, 4)
        self.assertTrue((vector_perp(v) == ans).all())

    def test_vector_dot(self):
        v = [3, 4]
        w = [5, 6]
        ans = 39
        self.assertAlmostEqual(vector_dot(v, w), ans)
        v = Vector(3, 4)
        self.assertAlmostEqual(vector_dot(v, w), ans)
        self.assertAlmostEqual(vector_dot(w, v), ans)

    def test_vector_cross_2D(self):
        ans = -2

        v = [3, 4]
        w = [5, 6]
        self.assertAlmostEqual(vector_cross(v, w), ans)
        self.assertAlmostEqual(vector_cross(w, v), -ans)

        v = Vector(3, 4)
        self.assertAlmostEqual(vector_cross(v, w), ans)
        self.assertAlmostEqual(vector_cross(w, v), -ans)

    def test_vector_cross_3D(self):
        ans = [-2, 4, -2]

        v = [3, 4, 5]
        w = [5, 6, 7]
        res = vector_cross(v, w)
        self.assertTrue(isinstance(res, (np.ndarray, pd.Series)))
        self.assertTrue((res == ans).all())
        self.assertTrue((-vector_cross(w, v) == ans).all())

        v = Vector(3, 4, 5)
        self.assertVectorEqual(vector_cross(v, w), ans)
        self.assertVectorEqual(-vector_cross(w, v), ans)

    def test_scalar_proj(self):
        ans = 4.9934383
        ans2 = 7.8

        v = [3, 4]
        w = [5, 6]
        self.assertAlmostEqual(scalar_proj(v, w), ans)
        self.assertAlmostEqual(scalar_proj(w, v), ans2)

        v = Vector(3, 4)
        self.assertAlmostEqual(scalar_proj(v, w), ans)
        self.assertAlmostEqual(scalar_proj(w, v), ans2)

    def test_vector_proj(self):
        warnings.simplefilter("error", Warning)
        ans = [3.19672131, 3.83606557]
        ans2 = [4.68, 6.24]

        v = [3, 4]
        w = [5, 6]
        self.assertVectorAlmostEqual(vector_proj(v, w), ans)
        self.assertVectorAlmostEqual(vector_proj(w, v), ans2)

        v = Vector(3, 4)
        self.assertVectorAlmostEqual(vector_proj(v, w), ans)
        self.assertVectorAlmostEqual(vector_proj(w, v), ans2)

    def test_vector_dist(self):
        v = [3, 4]
        w = [6, 8]
        ans = 5
        self.assertAlmostEqual(vector_dist(v, w), ans)
        self.assertAlmostEqual(vector_dist(w, v), ans)

        v = Vector(3, 4)
        self.assertAlmostEqual(vector_dist(v, w), ans)
        self.assertAlmostEqual(vector_dist(w, v), ans)

    def test_vector_diff_angle(self):
        v = [3, 4]
        w = [5, 6]
        ans = 0.0512371674
        self.assertAlmostEqual(vector_diff_angle(v, w), ans)
        self.assertAlmostEqual(vector_diff_angle(w, v), -ans)

        v = Vector(3, 4)
        self.assertAlmostEqual(vector_diff_angle(v, w), ans)
        self.assertAlmostEqual(vector_diff_angle(w, v), -ans)


class TestSeriesCopy(unittest.TestCase):
    def test_series_copy(self):
        series = TimeSeries()
        res = series.copy()
        self.assertTrue(isinstance(res, pd.Series))


class TestLeastsq(unittest.TestCase):
    def test_leastsq(self):
        # Create noise-free test data: y = 2x + 1
        x = np.array([0, 1, 2, 3, 4])
        y = 2 * x + 1

        def error_func(params, x, y):
            m, b = params
            return y - (m * x + b)

        # Initial guess
        x0 = [1, 0]  # m=1, b=0

        # Run leastsq
        best_params, details = leastsq(error_func, x0, x, y)

        # Check results
        self.assertAlmostEqual(best_params[0], 2.0, places=5)  # slope
        self.assertAlmostEqual(best_params[1], 1.0, places=5)  # intercept
        self.assertTrue(details.success)


class TestCrossings(unittest.TestCase):
    def test_crossings(self):
        # Create a simple linear series from 0 to 10
        index = np.linspace(0, 10, 11)
        values = index.copy()
        series = pd.Series(values, index=index)

        # Find where the series crosses 5
        result = crossings(series, 5)
        # Should cross exactly at 5
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 5.0, places=5)


class TestDataStructures(unittest.TestCase):
    def test_state(self):
        s = State(a=1, b=2)
        self.assertIsInstance(s, pd.Series)
        self.assertEqual(s['a'], 1)
        self.assertEqual(s['b'], 2)
        self.assertEqual(s.name, 'state')

    def test_timeseries(self):
        ts = TimeSeries([1, 2, 3], index=[10, 20, 30])
        self.assertIsInstance(ts, pd.Series)
        self.assertEqual(list(ts), [1, 2, 3])
        self.assertEqual(list(ts.index), [10, 20, 30])
        self.assertEqual(ts.index.name, 'Time')
        self.assertEqual(ts.name, 'Quantity')

    def test_sweepseries(self):
        ss = SweepSeries([4, 5, 6], index=[0.1, 0.2, 0.3])
        self.assertIsInstance(ss, pd.Series)
        self.assertEqual(list(ss), [4, 5, 6])
        self.assertEqual(list(ss.index), [0.1, 0.2, 0.3])
        self.assertEqual(ss.index.name, 'Parameter')
        self.assertEqual(ss.name, 'Metric')

    def test_timeframe(self):
        df = TimeFrame([[1, 2], [3, 4]], columns=['a', 'b'], index=[0, 1])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(list(df.columns), ['a', 'b'])
        self.assertEqual(list(df.index), [0, 1])

    def test_sweepframe(self):
        df = SweepFrame([[5, 6], [7, 8]], columns=['x', 'y'], index=[0.1, 0.2])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(list(df.columns), ['x', 'y'])
        self.assertEqual(list(df.index), [0.1, 0.2])

    def test_make_series(self):
        s = make_series([10, 20, 30], [1, 2, 3])
        self.assertIsInstance(s, pd.Series)
        self.assertEqual(list(s.index), [10, 20, 30])
        self.assertEqual(list(s.values), [1, 2, 3])
        self.assertEqual(s.name, 'values')
        self.assertEqual(s.index.name, 'index')

    def test_vector_polar(self):
        v = [3, 4]
        mag, angle = vector_polar(v)
        self.assertAlmostEqual(mag, 5.0, places=5)
        self.assertAlmostEqual(angle, np.arctan2(4, 3), places=5)

    def test_interpolate_inverse(self):
        # y = 2x + 1, invert to get x from y
        x = np.array([0, 1, 2, 3, 4])
        y = 2 * x + 1
        series = pd.Series(y, index=x)
        inv = interpolate_inverse(series)
        # y=5 -> x=2
        self.assertAlmostEqual(inv(5), 2.0, places=5)

    def test_gradient(self):
        s = pd.Series([1, 4, 9], index=[0, 1, 2])
        g = gradient(s)
        self.assertIsInstance(g, pd.Series)
        # Should be [3, 4, 5] for [1,4,9] at [0,1,2]
        self.assertTrue(np.allclose(g, [3, 4, 5]))

    def test_magnitude(self):
        self.assertEqual(magnitude(5), 5)
        class Dummy:
            magnitude = 42
        self.assertEqual(magnitude(Dummy()), 42)

    def test_remove_units(self):
        class Dummy:
            def __init__(self):
                self.x = 5
                self.y = 10
        d = Dummy()
        res = remove_units(d)
        self.assertEqual(res.x, 5)
        self.assertEqual(res.y, 10)

    def test_remove_units_series(self):
        s = pd.Series({'a': 1, 'b': 2})
        res = remove_units_series(s)
        self.assertTrue((res == s).all())

    def test_underride(self):
        d = {'a': 1}
        underride(d, a=2, b=3)
        self.assertEqual(d['a'], 1)
        self.assertEqual(d['b'], 3)


if __name__ == "__main__":
    unittest.main()
