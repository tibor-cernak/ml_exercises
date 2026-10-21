from otter.test_files import test_case

OK_FORMAT = False

name = "q8_best_model"
points = 1

@test_case(points=None, hidden=False)
def test_accuracy(np, best_model, X_test, y_test):
    assert np.isclose(X_test.max(), 1), 'X_test should still be normalized to values between 0 and 1'
    accuracy = np.mean(np.argmax(best_model.predict(X_test.reshape(-1, 28, 28, 1), verbose=0), axis=1) == y_test)
    assert accuracy >= 0.91, f'best_model reaches a test accuracy of {accuracy:.3f}, expected at least 0.91'
