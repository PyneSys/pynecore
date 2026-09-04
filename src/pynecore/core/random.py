import time


class PineRandom:
    """
    A Pine random generator reimplementation in Python, mimicking Java's nextDouble().
    """

    def __init__(self, seed=None):
        """
        :param seed: int, the initial seed
        """
        self.a = 25214903917
        self.c = 11
        self.m = 1 << 48
        # If seed not given, use current time
        if seed is None:
            seed = int(time.time() * 1000)
        # ``seed`` is an int-TYPED Pine value, which may carry a fractional
        # one; the bit operations below consume it as an integer
        self.state = (int(seed) ^ 0x5DEECE66D) & (self.m - 1)

    def _next_bits(self, bits):
        """
        :param bits: int, number of bits
        :return: int, that many bits from updated state
        """
        self.state = (self.a * self.state + self.c) & (self.m - 1)
        return self.state >> (48 - bits)

    def random(self, lower=0.0, upper=1.0):
        """
        Generate a random number in [lower, upper]

        :param lower: float, lower bound
        :param upper: float, upper bound
        :return: float, random number in [lower, upper]
        """
        hi = self._next_bits(26)  # top 26 bits
        lo = self._next_bits(27)  # next 27 bits
        rnd_double = ((hi << 27) + lo) / float(1 << 53)
        return lower + rnd_double * (upper - lower)
