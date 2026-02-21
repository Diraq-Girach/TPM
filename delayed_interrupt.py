import signal
import logging


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        # Initialize as None, not False
        self.signal_received = None
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        # Check if we got a signal AND if the old handler is actually a function
        if self.signal_received and callable(self.old_handler):
            self.old_handler(*self.signal_received)

