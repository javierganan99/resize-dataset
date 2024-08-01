from pathlib import Path
import json
import platform
import sys
import logging.config
import yaml
import os

# Configuration constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
DEFAULT_CFG_PATH = ROOT / "cfg" / "default.yaml"
MACOS, LINUX, WINDOWS = (
    platform.system() == x for x in ["Darwin", "Linux", "Windows"]
)  # environment booleans
LOGGING_NAME = "ResizeDataset"


def load_json(path):
    """
    Loads a JSON file from the specified path and returns its contents as a dictionary.

    Args:
        path (str): The path to the JSON file to be loaded.

    Returns:
        dict: The contents of the JSON file loaded as a dictionary.
    """
    with open(str(path), "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def ensure_folder_exist(path):
    """
    Checks that a folder exists at the specified path, and if it does not exist, creates it.
    This function takes a single string argument representing a filesystem path and ensures
    that a folder exists at that path. It creates all necessary intermediate directories if
    they do not exist. If the operation is successful, or if the folder already exists
    , the function returns True.
    Args:
        path (str): The filesystem path where the folder should exist.
    Returns:
        bool: True if the folder already exists or was created successfully, False otherwise.
    """
    path = str(path)
    separated = path.split(os.path.sep)
    # To consider absolute paths
    if separated[0] == "":
        separated.pop(0)
        separated[0] = os.path.sep + separated[0]
    exists = True
    for f in range(len(separated)):
        path = (
            os.path.sep.join(separated[: f + 1])
            if f > 0
            else (separated[0] + os.path.sep)
        )
        if not os.path.exists(path):
            os.mkdir(path)
            exists = False
    return exists


def save_json(data, path):
    """
    Saves the given dictionary as a JSON file to the specified path.

    Args:
        data (dict): The dictionary to be saved as a JSON file.
        path (str): The path to the JSON file to be saved.

    Returns:
        None
    """
    ensure_folder_exist(path=Path(path).parent)
    with open(str(path), "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def emojis(string=""):
    """
    Return a platform-dependent emoji-safe version of the given string.

    This function processes the input string and converts it to a version that
    is safe for use with emojis, depending on the platform. If the platform is
    Windows, it will encode the string and decode it to remove any non-ASCII
    characters. On other platforms, it will return the string as is.

    Args:
        string (str, optional): Input string to be converted to an emoji-safe version
            (default is "").

    Returns:
        str: Emoji-safe version of the input string.
    """
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


def set_logging(name=LOGGING_NAME):
    """Sets up logging for the given name with UTF-8 encoding support."""
    # Configure the console (stdout) encoding to UTF-8
    formatter = logging.Formatter("%(message)s")  # Default formatter
    if WINDOWS and sys.stdout.encoding != "utf-8":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                sys.stdout.encoding = "utf-8"
        except Exception as e:
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")

            class CustomFormatter(logging.Formatter):
                """
                CustomFormatter class for setting up logging with UTF-8 encoding
                and configurable verbosity.

                This class extends the standard logging.Formatter to provide custom
                logging behavior, such as applying emojis to log messages and
                supporting different verbosity levels. The format method is overridden
                to incorporate this additional functionality.

                Args:
                    record (LogRecord): The log record containing the information
                        to be formatted.

                Attributes:
                    None

                Methods:
                    format(record): Formats the specified log record, applying
                        emojis and returning the final log message.

                Private Methods:
                    None
                """

                def format(self, record):
                    """Sets up logging with UTF-8 encoding and configurable verbosity."""
                    return emojis(super().format(record))

            formatter = CustomFormatter("%(message)s")
    # Create and configure the StreamHandler
    level = logging.INFO
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME)


def colorstr(*input_str):
    """
    Colors a string using ANSI escape codes.

    This function allows you to apply multiple colors and formatting styles
    (like bold) to a given string using ANSI escape codes. If only one
    argument is provided, it defaults to using 'blue' and 'bold' as the formatting options.

    Args:
        *input_str (str): Color names and the string to be colored.
                          Example: colorstr('blue', 'bold', 'hello world').

    Returns:
        str: The colored string formatted with the specified ANSI escape codes.
    """
    *args, string = (
        input_str if len(input_str) > 1 else ("green", "bold", input_str[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


class ConfigDict(dict):
    """
    A dictionary subclass that enables attribute-style access to its keys.

    This custom dictionary class throws a KeyError when a key is not found, similar
    to a standard dictionary. However, it allows accessing keys as attributes.

    Args:
        None

    Attributes:
        None

    Methods:
        register(obj=None, name=None): Register the given object under the given name or
            `obj.__name__`.
        __missing__(key): Returns KeyError when the given key is not found.
        __getattr__(name): Retrieves the value of the attribute with the specified name.
        __setattr__(name, value): Set the value of the attribute.
        __delattr__(name): Delete the specified attribute if it exists.

    Private Methods:
        _do_register(name, obj): Register the given object under the specified name.
    """

    def _do_register(self, name, obj):
        """
        Registers an object with a given name in the registry.

        This function adds an object to the registry if the name provided
        has not already been registered. An assertion will raise an error
        if an attempt is made to register an object with a duplicate name.

        Args:
            name (str): The name under which the object will be registered.
            obj (Any): The object to be registered.

        Returns:
            None: This function does not return a value.
        """
        assert name not in self, (
            f"An object named '{name}' was already registered "
            f"in '{self._name}' registry!"
        )
        self[name] = obj

    def register(self, obj=None, name=None):
        """
        Register the given object under the specified name or `obj.__name__`.

        This function can be used as either a decorator or a regular function call.
        When used as a decorator, it allows for easy registration of functions or
        classes under a specified name. If no name is provided, the object's name
        will be used.

        Args:
            obj (callable, optional): The object to be registered. If None, it
                indicates that the function is being used as a decorator.
            name (str, optional): The name under which to register the object.
                If not provided, `obj.__name__` will be used.

        Returns:
            callable: If used as a decorator, this returns the decorated function
            or class. If called with an object, it returns None.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                reg_name = name if name is not None else func_or_class.__name__
                self._do_register(reg_name, func_or_class)
                return func_or_class

            return deco
        # used as a function call
        reg_name = name if name is not None else obj.__name__
        self._do_register(reg_name, obj)

    def __missing__(self, key):
        """
        Raises a KeyError when the requested key is not found in the collection.

        This method is typically used in conjunction with __getitem__ to handle cases
        when a key is missing from a dictionary-like object. It ensures that any attempt
        to access a non-existent key will result in a KeyError being raised, facilitating
        error handling in the calling code.

        Args:
            key (hashable): The key that was not found.

        Returns:
            KeyError: Raises a KeyError with the specified missing key.
        """
        raise KeyError(key)

    def __getattr__(self, name):
        """
        Retrieves the value of the attribute with the specified name.

        This method allows dynamic access to the object's attributes by their name.
        If the attribute exists, its value is returned; otherwise, an AttributeError
        is raised.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            object: The value of the attribute with the specified name.

        Raises:
            AttributeError: If the attribute does not exist in the object.
        """
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e

    def __setattr__(self, name, value):
        """
        Set the value of an attribute in the object.

        This method allows you to set an attribute of the instance using the
        attribute's name as a string. It effectively enables dynamic
        attribute assignment within the object.

        Args:
            name (str): The name of the attribute to set.
            value (object): The value to assign to the attribute.

        Returns:
            None: This method does not return a value.
        """
        self[name] = value

    def __delattr__(self, name):
        """
        Delete the specified attribute from the object.

        This method attempts to delete the attribute with the given name.
        If the attribute does not exist, it raises an AttributeError with
        a message indicating that the attribute is not found.

        Args:
            name (str): The name of the attribute to delete.

        Returns:
            None: This function does not return a value.
        """
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e


def load_config(cfg_filepath):
    """
    Loads a configuration from a YAML file into a ConfigDict object.

    This function reads a YAML configuration file, parses it,
    and returns a ConfigDict object that allows attribute-style access
    to the YAML content.

    Args:
        cfg_filepath (str): The file path to the YAML configuration file.

    Returns:
        resize_dataset.utils.ConfigDict: A ConfigDict object containing the
            parsed YAML configuration.
    """
    with open(cfg_filepath, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return ConfigDict(cfg if cfg is not None else {})


DEFAULT_CFG = load_config(DEFAULT_CFG_PATH)
