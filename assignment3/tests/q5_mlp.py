from otter.test_files import test_case

OK_FORMAT = False

name = "q5_mlp"
points = 1

@test_case(points=None, hidden=False)
def test_outputs(mlp):
    assert mlp.output_shape == (None, 10), f'mlp should output ten class probabilities, its output shape is {mlp.output_shape}'

@test_case(points=None, hidden=False)
def test_accuracy(np, mlp, X_test, y_test):
    accuracy = np.mean(np.argmax(mlp.predict(X_test.reshape(-1, 784), verbose=0), axis=1) == y_test)
    assert accuracy >= 0.85, f'mlp reaches a test accuracy of {accuracy:.3f}, expected at least 0.85'
