from otter.test_files import test_case

OK_FORMAT = False
name = "q3_tfidf"
points = 1


@test_case(points=None, hidden=False)
def test_tfidf_scores(np, tfidf_scores):
    scores = np.asarray(tfidf_scores)
    assert scores.shape == (5,), "Return one accuracy for each of five folds."
    assert np.all(np.isfinite(scores)) and np.all((scores >= 0) & (scores <= 1)), "Scores should be finite accuracies between 0 and 1."
