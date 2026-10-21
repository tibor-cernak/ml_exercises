from otter.test_files import test_case

OK_FORMAT = False

name = "q4_normalization"
points = 1

@test_case(points=None, hidden=False)
def test_shapes(np, X_train, X_test):
    # Later exercises reshape the images to 28x28x1, so only check the number of examples and pixels
    assert len(X_train) == 60000 and np.size(X_train) == 60000 * 784, f'X_train has shape {np.shape(X_train)}, expected (60000, 784)'
    assert len(X_test) == 10000 and np.size(X_test) == 10000 * 784, f'X_test has shape {np.shape(X_test)}, expected (10000, 784)'

@test_case(points=None, hidden=False)
def test_range(np, X_train, X_test):
    for name, A in [('X_train', X_train), ('X_test', X_test)]:
        assert np.issubdtype(A.dtype, np.floating), f'{name} should hold floats, it holds {A.dtype}'
        assert A.min() >= 0 and A.max() <= 1, f'{name} values should lie between 0 and 1, they lie in [{A.min()}, {A.max()}]'
        assert np.isclose(A.max(), 1), f'the largest value of {name} is {A.max()}, expected 1: did you divide by 255 more than once?'
