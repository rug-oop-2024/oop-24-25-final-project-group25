from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Class representing a custom error for a path not found"""

    def __init__(self, path: str) -> None:
        """
        Initialise a NotFoundError instance.

        Args:
            path (str): path not found
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract class representing a storage for data to be stored in.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    Class representing a locally existent storage.

    Attributes:
        _base_path (str): base path for files to be stored.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initialize an object of the LocalStorage type.

        Args:
            base_path: base path for files to be stored.

        Returns:
            None
        """

        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data with a specific key.

        Args:
            data (bytes): Data to save
            key (str): Key of the saved data, used in creating its path

        Returns:
            None
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data with a given key.
        Args:
            key (str): Key of the data
        Returns:
            bytes: Loaded data
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete data with a given key.
        Args:
            key (str): Key of the data
        Returns:
            None
        """
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """
        List all paths under a given prefix
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self, path: str) -> None:
        """
        Check if a given path exists.

        Args:
            path (str): the path to check
        Returns:
            None
        Raises:
            NotFoundError: if the path given is not found.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Joins a given addition to the base path.

        Args:
            path (str): addition to be joined to the base path
        Returns:
            str: the joined path.
        """
        return os.path.join(self._base_path, path)
