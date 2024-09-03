from typing import Any, Dict, Iterator, Tuple, KeysView, ValuesView, ItemsView
import pickle

class Config:
    """
    A class representing a configuration object.

    Attributes:
        _config (dict): The configuration dictionary.

    Methods:
        get(key): Get the value associated with the given key.
        set(key, value): Set the value associated with the given key.
        __getitem__(key): Get the value associated with the given key using indexing.
        __setitem__(key, value): Set the value associated with the given key using indexing.
        __repr__(): Return a string representation of the configuration dictionary.
        __str__(): Return a string representation of the configuration dictionary.
        __iter__(): Return an iterator over the configuration dictionary.
        __len__(): Return the number of items in the configuration dictionary.
        __contains__(key): Check if the configuration dictionary contains the given key.
        keys(): Return a view object of all the keys in the configuration dictionary.
        values(): Return a view object of all the values in the configuration dictionary.
        items(): Return a view object of all the key-value pairs in the configuration dictionary.
        update(config): Update the configuration dictionary with the given dictionary.
        to_dict(): Return the configuration dictionary.
        from_dict(config): Set the configuration dictionary using the given dictionary.
        save(path): Save the configuration dictionary to a file.
        load(path): Load the configuration dictionary from a file.

    """

    def __init__(self, config: Dict[str, Any] = {}) -> None:
        self._config: Dict[str, Any] = config

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value

    def __repr__(self) -> str:
        return str(self._config)

    def __str__(self) -> str:
        return str(self._config)

    def __iter__(self) -> Iterator[str]:
        return iter(self._config)

    def __len__(self) -> int:
        return len(self._config)

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def keys(self) -> KeysView[str]:
        return self._config.keys()

    def values(self) -> ValuesView[Any]:
        return self._config.values()

    def items(self) -> ItemsView[str, Any]:
        return self._config.items()

    def update(self, config: Dict[str, Any]) -> None:
        self._config.update(config)

    def get(self, key: str) -> Any:
        """
        Get the value associated with the given key.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            The value associated with the given key.

        """
        return self._config.get(key)

    def set(self, key: str, value: Any) -> None:
        """
        Set the value associated with the given key.

        Args:
            key (str): The key to set the value for.
            value: The value to be set.

        """
        self._config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the configuration dictionary.

        Returns:
            The configuration dictionary.

        """
        return self._config

    def from_dict(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration dictionary using the given dictionary.

        Args:
            config (dict): The dictionary to set the configuration from.

        """
        self._config = config

    def save(self, path: str) -> None:
        """
        Save the configuration dictionary to a file.

        Args:
            path (str): The path to the file.

        """
        with open(path, 'wb') as f:
            pickle.dump(self._config, f)

    def load(self, path: str) -> None:
        """
        Load the configuration dictionary from a file.

        Args:
            path (str): The path to the file.

        """
        with open(path, 'rb') as f:
            self._config = pickle.load(f)