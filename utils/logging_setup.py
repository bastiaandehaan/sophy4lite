import logging, os
def setup_logging(level: str="INFO", logfile: str | None=None) -> logging.Logger:
    logger = logging.getLogger("sophy4lite")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logger.level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if logfile:
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            fh = logging.FileHandler(logfile)
            fh.setLevel(logger.level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger
