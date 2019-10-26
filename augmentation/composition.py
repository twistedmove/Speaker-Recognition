import random
import numpy as np


class Compose:
    """Base compose
    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 1.0.
    """
    def __init__(self, transforms: 'BasicTransform', p: float=1.0):
        self.transforms = transforms
        self.p = p

        name_list = []
        for transform in self.transforms:
            name_list.append(type(transform).__name__)
        self.__name__ = "_".join(name_list)

    def __call__(self, samples: np.ndarray, **kwargs):
        transforms = self.transforms.copy()
        if random.random() < self.p:

            for transform in transforms:
                samples = transform(samples, **kwargs)

        return samples


class OneOf(Compose):
    """Select one of transforms to apply
    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 1.0.
    """

    def __init__(self, transforms: 'BasicTransform', p: float=1.0):
        super(OneOf, self).__init__(transforms, p)
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, samples: np.ndarray, **kwargs):
        transforms = self.transforms.copy()
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        transform = random_state.choice(transforms, p=self.transforms_ps)

        if random.random() < self.p:

            samples = transform(samples, **kwargs)

        return samples
