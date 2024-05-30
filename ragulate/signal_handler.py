import signal

# Global variable to indicate if a SIGINT was received
sigint_received = False

def signal_handler(sig, frame):
    global sigint_received
    sigint_received = True

# Set up the signal handler for SIGINT (Ctrl-C)
signal.signal(signal.SIGINT, signal_handler)

def interrupt_received() -> bool:
    global sigint_received
    return sigint_received
