import textwrap
import threading
from config import colors

lock = threading.Lock()

def wrapped_print(text: str, indent_count: int, width: int=80) -> None:
    """
    Prints wrapped text
    :param text: str
    :param indent_count: int
    :param width: int
    :return:
    """
    wrapped_text = textwrap.wrap(text, width)
    first_line = True
    for line in wrapped_text:
        if first_line:
            first_line = False
            print(str.format("{0}{1}", "." * indent_count, line))
        else:
            print(str.format("{0}{1}", " " * indent_count, line))


def indent_print_warning(text: str, indent_level:int = 0, indent_symbol: str=" ") -> None:
    """
    Prints given text in warning style with given indentation level considering new lines
    :param text: str
    :param indent_level: int
    :param indent_symbol: str
    :return:
    """
    __print__(colors.Yellow, text, indent_level, indent_symbol)


def indent_print_error(text: str, indent_level: int = 0, indent_symbol: str=" ") -> None:
    """
    Prints given text in error style with given indentation level considering new lines
    :param text: str
    :param indent_level: int
    :param indent_symbol: str
    :return:
    """
    __print__(colors.LightRed, text, indent_level, indent_symbol)


def indent_print_ok(text: str, indent_level: int = 0, indent_symbol: str=" ") -> None:
    """
    Prints given text in ok style with given indentation level considering new lines
    :param text: str
    :param indent_level: int
    :param indent_symbol: str
    :return:
    """
    __print__(colors.Blue, text, indent_level, indent_symbol)


def indent_print_header(text: str, indent_level: int = 0, indent_symbol: str=" ") -> None:

    """
    Prints given text in header style with given indentation level considering new lines
    :param text: str
    :param indent_level: int
    :param indent_symbol: str
    :return:
    """
    __print__(str.format("{0}{1}", colors.Bold, colors.Green), text, indent_level, indent_symbol)


def indent_print(text: str, indent_level: int = 0, indent_symbol: str=" ") -> None:
    """
    Prints given text in standard_policy color (black/white) with given indentation level considering new lines
    :rtype:
    :param text: str
    :param indent_level: int
    :param indent_symbol: str
    :return:
    """
    __print__(colors.ResetAll, text, indent_level, indent_symbol)


def indent_print_color(color_code: str, text: str, indent_level: int = 0, indent_symbol: str=" ") -> None:
    """
    Prints given text with given indentation level using the specified color code considering new lines
    :param color_code: colors/str
    :param text: str
    :param indent_level: int
    :param indent_symbol: str
    :return:
    """
    __print__(color_code, text, indent_level, indent_symbol)


def __print__(color_code: str, text: str, indent_level: int = 0, indent_symbol: str=" ") -> None:
    """
    Prints given text with given indentation level considering new lines
    :param color_code: colors/str
    :param text: str
    :param indent_level: int
    :param indent_symbol: str
    :return:
    """
    # lock print function for multi threading
    lock.acquire()
    tmp = text.split("\n")
    first_line = True
    for t_tmp in tmp:
        if first_line:
            first_line = False
            print(str.format("{0}{1}{2}", color_code, indent_symbol * indent_level, t_tmp))
        else:
            print(str.format("{0}{1}{2}", color_code, " " * indent_level, t_tmp))

    # release print function for multi threading
    lock.release()

def print_line() -> None:
    print(str.format("{0}{1}", colors.Red, "-" * 79))


def status(type: str, result: int) -> None:
    """

    :param type: str
        additional information for the result type
    :param result: int
        result value (0 = failure, 1 = success)
    :return:
    """
    indent_print(str.format("{0}: {1}", type, "SUCCESS" if result == 0 else "FAILURE"))
