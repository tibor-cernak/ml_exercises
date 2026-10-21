from otter.test_files import test_case

OK_FORMAT = False

name = "q7_cnn"
points = 1

@test_case(points=None, hidden=False)
def test_architecture(cnn):
    names = [type(layer).__name__ for layer in cnn.layers]
    assert names == ['Conv2D', 'Conv2D', 'MaxPooling2D', 'Flatten', 'Dense'], f'cnn has layers {names}, expected Conv2D, Conv2D, MaxPooling2D, Flatten, Dense'
    assert [cnn.layers[i].filters for i in (0, 1)] == [32, 64], 'the convolutional layers should have 32 and 64 filters'
    assert cnn.output_shape == (None, 10), f'cnn should output ten class probabilities, its output shape is {cnn.output_shape}'

@test_case(points=None, hidden=False)
def test_accuracy(np, cnn, X_test, y_test):
    accuracy = np.mean(np.argmax(cnn.predict(X_test.reshape(-1, 28, 28, 1), verbose=0), axis=1) == y_test)
    assert accuracy >= 0.88, f'cnn reaches a test accuracy of {accuracy:.3f}, expected at least 0.88 (is your data normalized?)'
