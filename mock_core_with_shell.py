import numpy as np


class CoreWithShell:
    def __init__(self, shape, core_radius=0.05, shell_min=0.15, shell_max=0.175):
        """
        Create a central core with a surrounding shell
        :param shape: 3D shape of the output array
        :param core_radius: fractional radius of the core
        :param shell_min: fractional radius of the inner shell boundary
        :param shell_max: fractional radius of the outer shell boundary
        :type shape (int, int, int)
        :type core_radius: float
        :type shell_min: float
        :type shell_max: float
        """
        self.ar = np.ndarray(shape=shape)
        x = (np.arange(0, shape[0]) + 0.5 - shape[0] / 2.) / shape[0]
        y = (np.arange(0, shape[1]) + 0.5 - shape[1] / 2.) / shape[1]
        z = (np.arange(0, shape[2]) + 0.5 - shape[2] / 2.) / shape[2]
        dist = np.sqrt(np.add.outer(np.add.outer(x ** 2, y ** 2), z ** 2))
        self.dist = dist
        self.ar = self._core_with_shell_profile(dist, core_radius, shell_min, shell_max).astype(float)

    @property
    def array(self):
        """

        :rtype : np.array
        """
        return self.ar

    @staticmethod
    def _core_with_shell_profile(x, core_radius, shell_min, shell_max):
        return np.logical_or(x < core_radius, np.logical_and(x > shell_min, x < shell_max))


class MockForest:
    def __init__(self):
        self.size = 300
        self.core_with_shell = CoreWithShell([self.size, self.size, self.size])

    def get_forest(self, x, y, z):
        x_indexes = (x % self.size).astype(int)
        y_indexes = (y % self.size).astype(int)
        z_indexes = (z % self.size).astype(int)
        return self.core_with_shell.array[x_indexes, y_indexes, z_indexes]
