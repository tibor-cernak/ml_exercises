from otter.test_files import test_case

OK_FORMAT = False
name = "q1_bag_of_words"
points = 1


@test_case(points=None, hidden=False)
def test_counts(np, count_words):
    result = count_words(["sad", "not", "sad", "unknown"], ["happy", "not", "sad"])
    assert isinstance(result, np.ndarray), "Return a NumPy array."
    assert result.shape == (3,), "Return one count for each vocabulary word."
    assert np.array_equal(result, [0, 1, 2]), "Count repeats in vocabulary order and ignore unknown words."


@test_case(points=None, hidden=False)
def test_empty(np, count_words):
    result = count_words([], ["sad", "happy"])
    assert np.array_equal(result, [0, 0]), "An empty document should have zero counts."
