from abc import ABCMeta, abstractmethod


class AccumulatorBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def flush(self):
        """
        Save intermediate results to file.

        :return:
        """
        return

    @abstractmethod
    def set_filename(self):
        """
        Set filename for saving results.
        :return:
        """
        return

    @abstractmethod
    def add_array_with_mask(self, ar_flux, ar_x, ar_y, mask):
        """
        Add flux with two corresponding coordinates, and a mask.
        :param ar_flux:
        :param ar_x:
        :param ar_y:
        :param mask:
        :return:
        """
        return

    @abstractmethod
    def __radd__(self, other):
        return

    @abstractmethod
    def __add__(self, other):
        return
