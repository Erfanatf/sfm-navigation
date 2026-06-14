"""Abstract data source and ATC implementation."""

import pandas as pd
from abc import ABC, abstractmethod
from ..crowd_analysis.binning import convert_units
from ..data.atc_loader import load_atc_raw

class AbstractDataSource(ABC):
    """Interface for data sources."""
    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        """Return a DataFrame with columns:
        timestamp, agent_id, pos_x_mm, pos_y_mm, velocity_mm_s,
        motion_angle_rad, facing_angle_rad.
        """
        pass

class ATCDataSource(AbstractDataSource):
    """Load from a single ATC CSV file."""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_raw_data(self) -> pd.DataFrame:
        df = load_atc_raw(self.csv_path)
        df = convert_units(df)
        return df