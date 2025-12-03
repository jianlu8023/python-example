import json
import logging.config
import os
from json import JSONDecodeError


def configparser() -> str:
    """
    configparser 返回logging.json路径
    :return: 返回logging.json路径
    """
    pwd = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pwd, "logging.json")


def genconfig(fpath: str) -> {}:
    """
    getconfig 读取json配置生成dict
    :return:
    """
    if not os.path.exists(fpath):
        print(f'config file `{fpath}` not found, use default config')
        fpath = configparser()
    with open(fpath, 'r') as f:
        c = json.load(f)
        for handler_name, handler_config in c.get("handlers", {}).items():
            if handler_name == "file":
                log_file = handler_config["filename"]
                log_dir = os.path.dirname(log_file)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
        return c


def configure_log(fpath: str) -> None:
    try:
        logging.config.dictConfig(genconfig(fpath))
    except FileNotFoundError as e:
        print(f'config file `{fpath}` not found:{e}')
    except JSONDecodeError as e:
        print(f'parse config file `{fpath}` error:{e}')
    except Exception as e:
        print(f'use config file `{fpath}` configure error:{e}')
