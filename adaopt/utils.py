

def n2(x):
    import numpy as np
    return np.sum(x*x)


def soft_thresholding(x, theta):
    '''Return the soft-thresholding of x with theta
    '''
    import numpy as np
    return np.sign(x)*np.maximum(0, np.abs(x) - theta)


def start_handler(logger, log_lvl, out=None):
    """Add a StreamHandler to logger is no handler as been started. The default
    behavior send the log to stdout.

    Parameters
    ----------
    logger: Logger object
    log_lvl: int
        Minimal level for a log message to be printed. Refers to the levels
        defined in the logging module.
    out: writable object
        Should be an opened writable file object. The default behavior is to
        log tht messages to STDOUT.
    """
    import logging
    logger.setLevel(log_lvl)
    if len(logger.handlers) == 0:
        if out is None:
            from sys import stdout as out

        ch = logging.StreamHandler(out)
        ch.setLevel(log_lvl)
        formatter = logging.Formatter('\r[%(name)s] - %(levelname)s '
                                      '- %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    t = np.linspace(-1, 1, 1000)
    plt.plot(t, soft_thresholding(t, .2))
    plt.plot(t, [n2(tt) for tt in t])
    plt.show()
