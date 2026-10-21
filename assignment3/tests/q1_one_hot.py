from otter.test_files import test_case

OK_FORMAT = False

name = "q1_one_hot"
points = 1

@test_case(points=None, hidden=False)
def test_shape(np, y, y_onehot):
    assert np.shape(y_onehot) == (len(y), 2), f'y_onehot has shape {np.shape(y_onehot)}, expected ({len(y)}, 2): one row per example, one column per class'

@test_case(points=None, hidden=False)
def test_values(np, y, y_onehot):
    assert np.all((y_onehot == 0) | (y_onehot == 1)), 'y_onehot should only contain zeros and ones'
    assert np.allclose(np.sum(y_onehot, axis=1), 1), 'each row of y_onehot should contain exactly one 1'
    assert np.array_equal(np.argmax(y_onehot, axis=1), y), 'the 1 in each row should be in the column of the true class'
