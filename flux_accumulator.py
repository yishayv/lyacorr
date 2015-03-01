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
    def add_array_with_mask(self, ar_flux, ar_x, ar_y, mask, ar_weights):
        """
        Add flux with two corresponding coordinates, and a mask.
        :type ar_flux:
        :type ar_x:
        :type ar_y:
        :type mask:
        :type ar_weights:
        :return:
        """
        return

    @abstractmethod
    def __radd__(self, other):
        return

    @abstractmethod
    def __add__(self, other):
        return
