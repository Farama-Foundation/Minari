import gym
import numpy as np


def check_box_dtype_supported(space, dtype):
    # Create an example observation with the specified dtype
    observation = np.zeros(space.shape, dtype=dtype)

    try:
        # Check if the observation is within the space
        assert space.contains(observation)
        return True
    except AssertionError:
        return False


def check_discrete_dtype_supported(space, dtype):
    # Create an example action with the specified dtype
    action = np.zeros(1, dtype=dtype)

    try:
        # Check if the action is within the space
        assert space.contains(action)
        return True
    except AssertionError:
        return False


def test_box_dtypes():
    # Create a Box space with a specific shape
    shape = (4,)
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float64)

    # Get all registered numpy scalar types
    all_dtypes = list(np.sctypeDict.values())

    unsupported_dtypes = []
    supported_dtypes = []

    for dtype in all_dtypes:
        if check_box_dtype_supported(space, dtype):
            supported_dtypes.append(dtype)
        else:
            unsupported_dtypes.append(dtype)

    supported_dtypes = sorted(set(supported_dtypes), key=lambda x: x.__name__)
    unsupported_dtypes = sorted(set(unsupported_dtypes), key=lambda x: x.__name__)

    print(
        "Supported dtypes for Box space:",
        [dtype.__name__ for dtype in supported_dtypes],
    )

    if unsupported_dtypes:
        print(
            "Unsupported dtypes for Box space:",
            [dtype.__name__ for dtype in unsupported_dtypes],
        )
    else:
        print("All dtypes are supported for Box space")


def test_discrete_dtypes():
    # Create a Discrete space with a specific number of actions
    n_actions = 10
    space = gym.spaces.Discrete(n_actions)

    # Get all registered numpy scalar types
    all_dtypes = list(np.sctypeDict.values())

    unsupported_dtypes = []
    supported_dtypes = []

    for dtype in all_dtypes:
        if check_discrete_dtype_supported(space, dtype):
            supported_dtypes.append(dtype)
        else:
            unsupported_dtypes.append(dtype)

    supported_dtypes = sorted(set(supported_dtypes), key=lambda x: x.__name__)
    unsupported_dtypes = sorted(set(unsupported_dtypes), key=lambda x: x.__name__)

    print(
        "Supported dtypes for Discrete space:",
        [dtype.__name__ for dtype in supported_dtypes],
    )

    if unsupported_dtypes:
        print(
            "Unsupported dtypes for Discrete space:",
            [dtype.__name__ for dtype in unsupported_dtypes],
        )
    else:
        print("All dtypes are supported for Discrete space")


test_box_dtypes()
test_discrete_dtypes()
