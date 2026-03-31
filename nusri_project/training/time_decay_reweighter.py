from __future__ import annotations

import numpy as np
import pandas as pd
from qlib.data.dataset.weight import Reweighter


class ExpHalflifeReweighter(Reweighter):
    def __init__(self, *, reference_time: str, half_life_months: float) -> None:
        self.reference_time = pd.Timestamp(reference_time)
        self.half_life_hours = float(half_life_months) * 30 * 24

    def reweight(self, data: pd.DataFrame) -> np.ndarray:
        timestamps = pd.to_datetime(data.index.get_level_values(0))
        age_hours = np.asarray((self.reference_time - timestamps).total_seconds(), dtype=float) / 3600.0
        weights = np.exp(-np.log(2) * age_hours / self.half_life_hours)
        return weights / weights.mean()
