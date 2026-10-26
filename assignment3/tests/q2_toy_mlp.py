from otter.test_files import test_case

OK_FORMAT = False

name = "q2_toy_mlp"
points = 1

@test_case(points=None, hidden=False)
def test_outputs(toy_mlp):
    assert toy_mlp.output_shape == (None, 2), f'toy_mlp should output two class probabilities, its output shape is {toy_mlp.output_shape}'

@test_case(points=None, hidden=False)
def test_accuracy(np, toy_mlp, X, y):
    accuracy = np.mean(np.argmax(toy_mlp.predict(X, verbose=0), axis=1) == y)
    assert accuracy >= 0.97, f'toy_mlp reaches an accuracy of {accuracy:.3f} on X, expected at least 0.97 (did you train it on the moons data?)'
