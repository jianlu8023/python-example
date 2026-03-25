from utilities import get_logger

logger = get_logger(__name__)


def hi(name: str = "pycharm"):
    """
    :param name:
    :return:
    """
    print(f"Hi {name}")


if __name__ == "__main__":
    hi("pytorch")
    logger.info(f"hello info")
