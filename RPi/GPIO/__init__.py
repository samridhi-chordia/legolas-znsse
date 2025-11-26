"""
RPi.GPIO Mock Implementation for Development
============================================

This is a development stub for RPi.GPIO that allows code to run on non-Raspberry Pi systems.
On actual Raspberry Pi hardware, install the real package: pip install RPi.GPIO

Note: This mock does NOT provide actual GPIO functionality - it's for development/testing only.
"""

# GPIO Modes
BCM = 11
BOARD = 10

# Pin modes
IN = 1
OUT = 0

# Pull up/down resistors
PUD_OFF = 20
PUD_DOWN = 21
PUD_UP = 22

# Pin values
LOW = 0
HIGH = 1

# Edge detection
RISING = 31
FALLING = 32
BOTH = 33

# Mock state
_mode = None
_setup_pins = {}
_warnings = True


def setmode(mode):
    """Set the numbering mode (BCM or BOARD)"""
    global _mode
    _mode = mode
    print(f"[RPi.GPIO MOCK] setmode({mode})")


def setwarnings(flag):
    """Enable/disable warnings"""
    global _warnings
    _warnings = flag


def setup(channel, mode, initial=None, pull_up_down=None):
    """Setup a GPIO channel"""
    _setup_pins[channel] = {
        'mode': mode,
        'initial': initial,
        'pull': pull_up_down,
        'value': initial if initial is not None else LOW
    }
    print(f"[RPi.GPIO MOCK] setup(channel={channel}, mode={mode}, initial={initial}, pull={pull_up_down})")


def output(channel, value):
    """Set output value of a GPIO channel"""
    if channel in _setup_pins:
        _setup_pins[channel]['value'] = value
    print(f"[RPi.GPIO MOCK] output(channel={channel}, value={value})")


def input(channel):
    """Read value from a GPIO channel"""
    value = _setup_pins.get(channel, {}).get('value', LOW)
    print(f"[RPi.GPIO MOCK] input(channel={channel}) -> {value}")
    return value


def cleanup(channel=None):
    """Clean up GPIO channels"""
    global _setup_pins
    if channel is None:
        _setup_pins = {}
        print("[RPi.GPIO MOCK] cleanup() - all channels")
    else:
        if channel in _setup_pins:
            del _setup_pins[channel]
        print(f"[RPi.GPIO MOCK] cleanup(channel={channel})")


def gpio_function(channel):
    """Return the current function of a GPIO channel"""
    if channel in _setup_pins:
        return _setup_pins[channel]['mode']
    return -1


def add_event_detect(channel, edge, callback=None, bouncetime=None):
    """Add edge detection to a channel"""
    print(f"[RPi.GPIO MOCK] add_event_detect(channel={channel}, edge={edge}, callback={callback}, bouncetime={bouncetime})")


def remove_event_detect(channel):
    """Remove edge detection from a channel"""
    print(f"[RPi.GPIO MOCK] remove_event_detect(channel={channel})")


def event_detected(channel):
    """Check if an event was detected"""
    return False


def add_event_callback(channel, callback):
    """Add callback for event detection"""
    print(f"[RPi.GPIO MOCK] add_event_callback(channel={channel}, callback={callback})")


def wait_for_edge(channel, edge, bouncetime=None, timeout=None):
    """Wait for edge detection"""
    print(f"[RPi.GPIO MOCK] wait_for_edge(channel={channel}, edge={edge}, bouncetime={bouncetime}, timeout={timeout})")
    return channel


# PWM Mock
class PWM:
    """Mock PWM class"""

    def __init__(self, channel, frequency):
        self.channel = channel
        self.frequency = frequency
        self.duty_cycle = 0
        self.running = False
        print(f"[RPi.GPIO MOCK] PWM(channel={channel}, frequency={frequency})")

    def start(self, duty_cycle):
        """Start PWM output"""
        self.duty_cycle = duty_cycle
        self.running = True
        print(f"[RPi.GPIO MOCK] PWM.start(duty_cycle={duty_cycle})")

    def stop(self):
        """Stop PWM output"""
        self.running = False
        print(f"[RPi.GPIO MOCK] PWM.stop()")

    def ChangeDutyCycle(self, duty_cycle):
        """Change duty cycle"""
        self.duty_cycle = duty_cycle
        print(f"[RPi.GPIO MOCK] PWM.ChangeDutyCycle({duty_cycle})")

    def ChangeFrequency(self, frequency):
        """Change frequency"""
        self.frequency = frequency
        print(f"[RPi.GPIO MOCK] PWM.ChangeFrequency({frequency})")


# Version info
VERSION = "0.7.1-mock"
RPI_INFO = {'TYPE': 'Mock', 'PROCESSOR': 'Development', 'REVISION': 'dev'}


print("[RPi.GPIO] Mock module loaded - running on non-Raspberry Pi system")
print("[RPi.GPIO] For actual hardware, install: pip install RPi.GPIO")
