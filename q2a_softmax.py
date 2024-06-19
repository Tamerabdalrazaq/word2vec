import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        max_in_rows = np.max(x, axis=1, keepdims=True)
        x = np.exp(x - max_in_rows)
        sum_of_rows = np.sum(x, axis=1, keepdims=True)
        x = x / sum_of_rows
    else:
        # Vector
        x = np.exp(x - np.max(x))
        x = x / np.sum(x)

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    print("Running your tests...")
    x = np.array([1,3,2,4,2])
    test = softmax(x)
    ans = np.array([0.029489, 0.217895, 0.080159, 0.592299, 0.080159])
    print(test)
    print(ans)
    assert np.allclose(test, ans, rtol=1e-05, atol=1e-06)



if __name__ == "__main__":
    # test_softmax_basic()
    # your_softmax_test()
    pass
