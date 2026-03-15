"""
1-Euro Filter implementation for low-latency jitter reduction.
Reference: Casiez et al. 2012, "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input"
"""
import math
import time


class LowPassFilter:
    def __init__(self, alpha: float):
        self._alpha = alpha
        self._y = None
        self._s = None

    def set_alpha(self, alpha: float):
        self._alpha = max(0.0, min(1.0, alpha))

    def filter(self, value: float, alpha: float = None) -> float:
        if alpha is not None:
            self.set_alpha(alpha)
        if self._y is None:
            self._y = value
            self._s = value
        else:
            self._y = self._alpha * value + (1.0 - self._alpha) * self._s
            self._s = self._y
        return self._y

    @property
    def last_value(self) -> float:
        return self._s


class OneEuroFilter:
    """
    1-Euro filter for a single scalar signal.
    min_cutoff: minimum cutoff frequency (Hz) — controls lag at low speeds
    beta: speed coefficient — controls lag at high speeds
    d_cutoff: cutoff for derivative (typically 1.0 Hz)

    Reference: Casiez et al. 2012, "1€ Filter"
    Formula: alpha = (2*pi*fc*te) / (2*pi*fc*te + 1)
    where fc = cutoff frequency, te = sample period (dt)
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.1, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        # Initialize with a reasonable default dt (will be updated on first filter call)
        default_dt = 1.0 / 30.0  # assume 30fps initially
        self._x = LowPassFilter(self._alpha(min_cutoff, default_dt))
        self._dx = LowPassFilter(self._alpha(d_cutoff, default_dt))
        self._last_time = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        """Compute smoothing factor alpha from cutoff frequency and sample period.

        Correct 1-euro formula: alpha = (2*pi*fc*te) / (2*pi*fc*te + 1)
        where fc = cutoff frequency (Hz), te = sample period (seconds)
        """
        te = max(dt, 1e-10)  # sample period (avoid division by zero)
        fc = max(cutoff, 1e-10)  # cutoff frequency
        r = 2.0 * math.pi * fc * te
        return r / (r + 1.0)

    def filter(self, value: float, timestamp: float = None) -> float:
        if timestamp is None:
            timestamp = time.monotonic()

        if self._last_time is None:
            self._last_time = timestamp
            return self._x.filter(value)

        dt = timestamp - self._last_time
        if dt <= 0:
            return self._x.last_value

        self._last_time = timestamp

        # Derivative
        dx = (value - (self._x.last_value or value)) / dt
        edx = self._dx.filter(dx, alpha=self._alpha(self.d_cutoff, dt))

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.filter(value, alpha=self._alpha(cutoff, dt))


class OneEuroFilter3D:
    """1-Euro filter for 3D positions (x, y, z independently)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.1):
        self.filters = [
            OneEuroFilter(min_cutoff, beta),
            OneEuroFilter(min_cutoff, beta),
            OneEuroFilter(min_cutoff, beta),
        ]

    def filter(self, x: float, y: float, z: float, timestamp: float = None) -> tuple:
        ts = timestamp or time.monotonic()
        fx = self.filters[0].filter(x, ts)
        fy = self.filters[1].filter(y, ts)
        fz = self.filters[2].filter(z, ts)
        return (fx, fy, fz)
