"""Numerical contracts and transparent population benchmarks."""
import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from directional_irreversibility import DII, dii_from_residuals, hsic, half_life


class TestHSIC(unittest.TestCase):
    def test_matches_explicit_centering_matrix(self):
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=13), rng.normal(size=13)
        k = np.exp(-((x[:, None] - x[None, :]) ** 2) / 2)
        l = np.exp(-((y[:, None] - y[None, :]) ** 2) / 2)
        centering = np.eye(13) - np.ones((13, 13)) / 13
        expected = np.trace(k @ centering @ l @ centering) / 13**2
        self.assertAlmostEqual(hsic(x, y), expected, places=13)

    def test_constant_is_zero(self):
        self.assertAlmostEqual(hsic(np.ones(15), np.arange(15)), 0, places=13)

    def test_swap_changes_sign(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=100)
        y = x**2 + rng.normal(size=100)
        a = dii_from_residuals(y - x**2, x, x, y)
        b = dii_from_residuals(x, y, y - x**2, x)
        self.assertAlmostEqual(a.dii, -b.dii, places=13)

    def test_oracle_nonlinear_benchmark(self):
        rng = np.random.default_rng(10)
        x, noise = rng.normal(size=1200), rng.normal(size=1200) * 0.25
        y = x**2 + noise
        # E[Y|X]=X^2 and E[X|Y]=0 by symmetry.
        result = dii_from_residuals(noise, x, x, y)
        self.assertGreater(result.dii, 0.01)
        self.assertLess(result.forward_hsic, 0.002)

    def test_pure_scale_equal_positive_components(self):
        x = np.repeat([-2., -1., 1., 2.], 2)
        e = np.tile([-1., 1.], 4)
        y = np.sqrt(1+x*x) * e
        result = dii_from_residuals(y, x, x, y)
        self.assertGreater(result.forward_hsic, 0)
        self.assertAlmostEqual(result.dii, 0, places=13)

    def test_invalid_inputs(self):
        for a, b in [([1, np.nan], [1, 2]), ([1, 2], [1, 2, 3])]:
            with self.assertRaises(ValueError):
                hsic(a, b)
        with self.assertRaises(ValueError):
            hsic([1, 2], [3, 4], bandwidth_a=0)

    def test_complete_regressor_history_matters(self):
        c = np.tile([-1., 1.], 50)
        x = np.repeat([-1., 1.], 50)
        self.assertGreater(hsic(c, np.column_stack((x, c))), hsic(c, x))


class TestHeldOutEstimator(unittest.TestCase):
    def test_linear_gaussian_oracle_fit_is_small(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=1800)
        y = 0.8*x + rng.normal(size=1800)
        model = DII(LinearRegression()).fit(x[:1000], y[:1000])
        result = model.score(x[1000:], y[1000:])
        self.assertLess(abs(result.dii), 0.003)

    def test_score_does_not_refit_scaling(self):
        x = np.arange(30.)
        model = DII(LinearRegression()).fit(x, x + np.sin(x))
        scale = model.y_scale_
        model.score(x + 30, (x + 30) * 3)
        self.assertEqual(model.y_scale_, scale)

    def test_requires_matching_history(self):
        x = np.arange(30.)
        model = DII(LinearRegression()).fit(x, x + np.sin(x), x**2)
        with self.assertRaises(ValueError):
            model.score(x, x)

    def test_requires_fit(self):
        with self.assertRaises(ValueError):
            DII().score([1, 2], [3, 4])


class TestHalfLife(unittest.TestCase):
    def test_known_exponential_grid(self):
        h = np.arange(5)
        r = half_life(h, np.exp(-np.log(2)*h/2))
        self.assertEqual(r.half_life, 2)
        self.assertEqual(r.bracket, (1, 2))

    def test_reference_is_elapsed_time_and_floor(self):
        r = half_life([3, 4, 5], [1.2, .9, .7], floor=.2)
        self.assertEqual(r.half_life, 2)

    def test_first_versus_sustained(self):
        first = half_life([0, 1, 2, 3], [1, .4, .8, .3], mode="first")
        sustained = half_life([0, 1, 2, 3], [1, .4, .8, .3])
        self.assertEqual(first.half_life, 1)
        self.assertEqual(sustained.half_life, 3)
        self.assertTrue(sustained.nonmonotone)

    def test_censoring_and_zero_baseline(self):
        self.assertEqual(half_life([0, 1], [1, .8]).status, "not_reached")
        self.assertEqual(half_life([0, 1], [0, -.1]).status, "no_positive_reference")

    def test_sign_reversal_flagged(self):
        self.assertTrue(half_life([0, 1], [1, -.2]).sign_reversal)

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            half_life([0, 0], [1, .4])


if __name__ == "__main__":
    unittest.main()
