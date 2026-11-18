import logging

log = logging.getLogger(__name__)


def say() -> None:
    log.info('hello')
