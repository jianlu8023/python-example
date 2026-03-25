import os
from pathlib import Path
from typing import Iterator, Tuple, Union


def get_dir_all_files(path: str, recursion: bool = False,
                      extensions: Union[str, Tuple[str, ...]] = None) -> Iterator[str]:
    """
    get_dir_all_files 获取文件夹下所有文件

    :param path: 文件夹路径
    :param recursion:  是否递归
    :param extensions:  后缀
    :return: 文件全路径
    """

    # 1. 预处理后缀 且都带点
    if extensions:
        if isinstance(extensions, str):
            # 处理单字符串
            ext_list = [extensions.lower() if extensions.startswith('.') else f".{extensions.lower()}"]
        else:
            # 处理列表或元组
            ext_list = [ext.lower() if ext.startswith('.') else f".{ext.lower()}" for ext in extensions]

        extensions = tuple(ext_list)

    # p = Path(path)
    # pathlib 检查非常直观
    # if not p.exists() or not p.is_dir():
    #     return
    # files_iter = p.rglob("*") if recursion else p.glob("*")
    # for f in files_iter:
    #     if f.is_file():
    #         if extensions:
    #             if f.suffix.lower() in extensions:
    #                 yield str(f.absolute())
    #         else:
    #             yield str(f.absolute())

    # 1. 健壮性检查：路径是否存在
    if not os.path.exists(path):
        print(f"警告: 路径不存在 -> {path}")
        raise ValueError(f"Path not found: {path}")

    # 2. 健壮性检查：是否是文件夹
    if not os.path.isdir(path):
        print(f"警告: 路径不是一个文件夹 -> {path}")
        return

    if recursion:
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                if extensions and not filename.lower().endswith(extensions):
                    continue
                yield os.path.join(dirpath, filename)
    else:
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isfile(full_path):
                if extensions and not item.lower().endswith(extensions):
                    continue
                yield full_path


def get_filename(path: str, with_suffix: bool = False) -> str:
    """
    get_filename 获取文件名

    :param path: 文件路径
    :param with_suffix: 是否带后缀
    :return: 文件名字符串
    """
    if not path:
        return ""
    p = Path(path)
    # p.name 获取带后缀的文件名 (basename)
    # p.stem 获取不带后缀的文件名
    return p.name if with_suffix else p.stem
