from __future__ import annotations

import unittest

import pandas as pd

from nusri_project.training.signal_transform import (
    apply_component_transform,
    fit_component_transform,
)


class SignalTransformTests(unittest.TestCase):
    def test_robust_norm_clip_uses_train_only_statistics(self) -> None:
        train = pd.Series([0.0, 1.0, 2.0, 3.0])
        test = pd.Series([3.0, 4.0])

        params = fit_component_transform(
            train,
            transform="robust_norm_clip",
            clip_value=3.0,
        )
        result = apply_component_transform(
            test,
            transform="robust_norm_clip",
            params=params,
            clip_value=3.0,
        )

        self.assertTrue((result <= 1.0).all())
        self.assertTrue((result >= -1.0).all())
        self.assertAlmostEqual(result.iloc[0], ((3.0 - 1.5) / 1.4826) / 3.0, places=6)

    def test_robust_norm_clip_falls_back_to_std_when_mad_is_zero(self) -> None:
        train = pd.Series([1.0, 1.0, 1.0, 3.0])

        params = fit_component_transform(
            train,
            transform="robust_norm_clip",
            clip_value=3.0,
        )

        self.assertGreater(params["scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
