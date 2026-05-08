import pandas as pd

from analytics_core.candidates.cross_candidates import generate_cross_candidates


def test_generate_cross_candidates_smoke(tmp_path):
    interactions = pd.DataFrame(
        [
            {"feature_a": "a", "feature_b": "b", "interaction_score": 1.0},
            {"feature_a": "a", "feature_b": "c", "interaction_score": 0.5},
        ]
    )
    df = pd.DataFrame({"a": ["x", "y", "x"], "b": ["u", "u", "v"], "c": [1, 2, 3]})
    ypath = tmp_path / "candidate.yaml"
    mpath = tmp_path / "meta.csv"
    res = generate_cross_candidates(
        interactions_df=interactions,
        prediction_df=df,
        output_yaml_path=str(ypath),
        output_metadata_path=str(mpath),
        top_n=10,
        max_cardinality_estimate=1000,
        max_feature_nunique=1000,
    )
    assert res.n_candidates >= 1
    assert ypath.exists()
    assert mpath.exists()

