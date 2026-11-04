from otter.test_files import test_case

OK_FORMAT = False
name = "q2_cv_models"
points = 1


@test_case(points=None, hidden=False)
def test_cv_results(np, cv_results):
    assert isinstance(cv_results, dict), "Store the scores in a dictionary."
    for key in ("baseline", "svm", "forest"):
        assert key in cv_results, f"Missing scores for {key}."
        scores = np.asarray(cv_results[key])
        assert scores.shape == (5,), f"{key} should have one score for each of five folds."
        assert np.all(np.isfinite(scores)) and np.all((scores >= 0) & (scores <= 1)), f"{key} scores should be finite accuracies between 0 and 1."
