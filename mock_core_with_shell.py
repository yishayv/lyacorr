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
        self.ar = self._core_with_shell_profile(dist, core_radius, shell_min, shell_max)

    @property
    def array(self):
        """

        :rtype : np.array
        """
        return self.ar

    @staticmethod
    def _core_with_shell_profile(x, core_radius, shell_min, shell_max):
        core = x < core_radius
        shell = np.logical_and(x > shell_min, x < shell_max)
        return shell * 0.017 + core * 1.


class MockForest:
    def __init__(self, resolution, shell_fractional_width, shell_separation, core_radius, shell_radius):
        self.res = resolution
        # sizes in Mpc
        shell_max_dist = shell_radius * (1 + shell_fractional_width)
        shell_min_dist = shell_radius * (1 - shell_fractional_width)
        # scale of the entire grid:
        self.cube_size = max(shell_min_dist, shell_max_dist, core_radius) + shell_separation / 2.
        # sizes relative to cube size
        self.shell_min_fraction = 0.5 * shell_min_dist / self.cube_size
        self.shell_max_fraction = 0.5 * shell_max_dist / self.cube_size
        self.core_radius_fraction = 0.5 * core_radius / self.cube_size
        # pixel density:
        self.inv_pixel_size = self.res / self.cube_size * 0.5
        self.core_with_shell = CoreWithShell([self.res, self.res, self.res],
                                             core_radius=self.core_radius_fraction,
                                             shell_min=self.shell_min_fraction,
                                             shell_max=self.shell_max_fraction)
        # remove the positive bias of the correlation
        # self.core_with_shell.array -= self.core_with_shell.array.mean()

    def get_forest(self, x, y, z):
        x_indexes = ((x * self.inv_pixel_size) % self.res).astype(int)
        y_indexes = ((y * self.inv_pixel_size) % self.res).astype(int)
        z_indexes = ((z * self.inv_pixel_size) % self.res).astype(int)
        return self.core_with_shell.array[x_indexes, y_indexes, z_indexes]
