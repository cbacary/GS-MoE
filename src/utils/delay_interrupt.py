import signal


class delay_keyboard_interrupt:
    """Ctrl-C / kernel interrupt inside this block is recorded and re-raised
    as KeyboardInterrupt only after the block exits."""

    def __init__(self, manual=False):
        self._received = None
        self._old = signal.signal(signal.SIGINT, self._handler)  # main thread only
        self.manual = manual
        self.interrupted = False

    def __enter__(self):
        self._received = None
        self._old = signal.signal(signal.SIGINT, self._handler)  # main thread only
        self.interrupted = False
        return self

    def _handler(self, sig, frame):
        self._received = (sig, frame)
        self.interrupted = True
        print("Interrupt received — will fire after critical section finishes...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._old)  # restore prior handler
        if self._received is not None and exc_type is None and not self.manual:
            raise KeyboardInterrupt
        return False
