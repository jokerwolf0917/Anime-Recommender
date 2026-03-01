import numpy as np
import pandas as pd

try:
    from app_helpers import build_numeric_features
except ImportError:
    from src.app_helpers import build_numeric_features


def test_build_numeric_features_handles_unknown_and_strings():
    df = pd.DataFrame(
        {
            "Episodes": ["12", "Unknown", None],
            "Score": ["8.7", "nan", "7.0"],
            "Members": ["1,200", "300", "Unknown"],
            "Favorites": ["10", "-", "5"],
        }
    )

    numeric = build_numeric_features(df)

    assert numeric.shape == (3, 4)
    assert numeric.dtype.kind == "f"

    np.testing.assert_allclose(numeric[0], np.array([8.7, 1200.0, 10.0, 12.0]))
    np.testing.assert_allclose(numeric[1], np.array([0.0, 300.0, 0.0, 0.0]))
    np.testing.assert_allclose(numeric[2], np.array([7.0, 0.0, 5.0, 0.0]))
