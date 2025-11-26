"""
spidev Mock Implementation for Development
==========================================

This is a development stub for spidev that allows code to run on non-Raspberry Pi systems.
On actual Raspberry Pi hardware, install the real package: pip install spidev

Note: This mock does NOT provide actual SPI functionality - it's for development/testing only.
"""


class SpiDev:
    """Mock SPI device class"""

    def __init__(self):
        self.bus = None
        self.device = None
        self.mode = 0
        self.max_speed_hz = 500000
        self.bits_per_word = 8
        self.lsbfirst = False
        self.cshigh = False
        self.threewire = False
        self.loop = False
        self.no_cs = False
        self.ready = False
        print("[spidev MOCK] SpiDev created")

    def open(self, bus, device):
        """Open SPI connection"""
        self.bus = bus
        self.device = device
        self.ready = True
        print(f"[spidev MOCK] open(bus={bus}, device={device})")

    def close(self):
        """Close SPI connection"""
        self.ready = False
        print("[spidev MOCK] close()")

    def readbytes(self, n):
        """Read n bytes from SPI device"""
        data = [0] * n
        print(f"[spidev MOCK] readbytes({n}) -> {data}")
        return data

    def writebytes(self, values):
        """Write bytes to SPI device"""
        print(f"[spidev MOCK] writebytes({values})")

    def writebytes2(self, values):
        """Write bytes to SPI device (variant)"""
        print(f"[spidev MOCK] writebytes2({values})")

    def xfer(self, values, speed_hz=0, delay_usec=0, bits_per_word=0):
        """Transfer data (write and read simultaneously)"""
        result = [0] * len(values)
        print(f"[spidev MOCK] xfer({values}, speed={speed_hz}, delay={delay_usec}, bits={bits_per_word}) -> {result}")
        return result

    def xfer2(self, values, speed_hz=0, delay_usec=0, bits_per_word=0):
        """Transfer data variant"""
        result = [0] * len(values)
        print(f"[spidev MOCK] xfer2({values}, speed={speed_hz}, delay={delay_usec}, bits={bits_per_word}) -> {result}")
        return result

    def xfer3(self, values, speed_hz=0, delay_usec=0, bits_per_word=0):
        """Transfer data variant 3"""
        result = [0] * len(values)
        print(f"[spidev MOCK] xfer3({values}, speed={speed_hz}, delay={delay_usec}, bits={bits_per_word}) -> {result}")
        return result

    @property
    def mode(self):
        """SPI mode (0-3)"""
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        print(f"[spidev MOCK] mode = {value}")

    @property
    def max_speed_hz(self):
        """Maximum SPI speed in Hz"""
        return self._max_speed_hz

    @max_speed_hz.setter
    def max_speed_hz(self, value):
        self._max_speed_hz = value
        print(f"[spidev MOCK] max_speed_hz = {value}")

    @property
    def bits_per_word(self):
        """Bits per word"""
        return self._bits_per_word

    @bits_per_word.setter
    def bits_per_word(self, value):
        self._bits_per_word = value
        print(f"[spidev MOCK] bits_per_word = {value}")

    @property
    def lsbfirst(self):
        """LSB first mode"""
        return self._lsbfirst

    @lsbfirst.setter
    def lsbfirst(self, value):
        self._lsbfirst = value
        print(f"[spidev MOCK] lsbfirst = {value}")

    @property
    def cshigh(self):
        """Chip select active high"""
        return self._cshigh

    @cshigh.setter
    def cshigh(self, value):
        self._cshigh = value
        print(f"[spidev MOCK] cshigh = {value}")

    @property
    def threewire(self):
        """3-wire mode (MOSI/MISO combined)"""
        return self._threewire

    @threewire.setter
    def threewire(self, value):
        self._threewire = value
        print(f"[spidev MOCK] threewire = {value}")

    @property
    def loop(self):
        """Loopback mode"""
        return self._loop

    @loop.setter
    def loop(self, value):
        self._loop = value
        print(f"[spidev MOCK] loop = {value}")

    @property
    def no_cs(self):
        """No chip select"""
        return self._no_cs

    @no_cs.setter
    def no_cs(self, value):
        self._no_cs = value
        print(f"[spidev MOCK] no_cs = {value}")

    @property
    def ready(self):
        """Device ready status"""
        return self._ready

    @ready.setter
    def ready(self, value):
        self._ready = value

    def fileno(self):
        """Return file descriptor (mock)"""
        return -1


# Module constants
VERSION = "3.8-mock"


print("[spidev] Mock module loaded - running on non-Raspberry Pi system")
print("[spidev] For actual hardware, install: pip install spidev")
