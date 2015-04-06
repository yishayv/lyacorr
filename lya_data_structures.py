class LyaForestTransmittance:
    def __init__(self, ar_z, ar_transmittance, ar_pipeline_ivar):
        """
        a simple wrapper for holding lya-forest data.
        :type ar_z: np.array
        :type ar_transmittance: np.array
        :type ar_pipeline_ivar: np.array
        """
        self.ar_z = ar_z
        self.ar_transmittance = ar_transmittance
        self.ar_ivar = ar_pipeline_ivar


class LyaForestTransmittanceBinned:
    def __init__(self, ar_mask, ar_transmittance, ar_pipeline_ivar):
        """
        a simple wrapper for holding lya-forest data.
        :type ar_mask: np.array
        :type ar_transmittance: np.array
        :type ar_pipeline_ivar: np.array
        """
        self.ar_mask = ar_mask
        self.ar_transmittance = ar_transmittance
        self.ar_ivar = ar_pipeline_ivar


class LyaForestDeltaT:
    def __init__(self, ar_z, ar_delta_t, ar_delta_t_ivar):
        """
        a simple wrapper for holding lya-forest data.
        :type ar_z: np.array
        :type ar_delta_t: np.array
        :type ar_delta_t_ivar: np.array
        """
        self.ar_z = ar_z
        self.ar_delta_t = ar_delta_t
        self.ar_ivar = ar_delta_t_ivar
