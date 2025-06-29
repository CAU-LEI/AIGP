import time
import contextlib

@contextlib.contextmanager
def Timer(name="Task"):
    """
    Context manager for measuring and printing task execution time
    """
    start = time.time()
    print("Starting {}...".format(name))
    yield
    elapsed = time.time() - start
    print("{} completed in {:.2f} seconds".format(name, elapsed))
