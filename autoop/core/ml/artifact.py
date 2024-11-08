import pandas as pd
import io


class Artifact:
    """
    Class representing an artifact.

    Attributes:
        type (str): asset's type
        name (str): asste's name
        version (str): asset's version
        asset_path (str): the path to the asset
        data (bytes): state of the data
        tags (list[str]): list of the tags for the asset
        metadata (dict): dcitionary holding the asset's metadata
        id: id of the artifact in the database
    """

    def __init__(
        self,
        type: str = None,
        name: str = None,
        version: str = None,
        asset_path: str = None,
        data: bytes = None,
        tags: list[str] = None,
        metadata: dict = None,
        id: str = None,
    ) -> None:
        """
        Initialize an Artifact object.

        Args:
            type (str): asset's type
            name (str): asste's name
            version (str): asset's version
            asset_path (str): the path to the asset
            data (bytes): state of the data
            tags (list[str]): list of the tags for the asset
            metadata (dict): dcitionary holding the asset's metadata
            id: id of the artifact in the database
        returns:
            None
        """
        self.type = type
        self.name = name
        self.version = version
        self.asset_path = asset_path
        self.data = data
        self.tags = tags
        self.metadata = metadata
        self.id = id

    def read(self) -> pd.DataFrame:
        """
        Return a dataframe containing the artifact's data.

        Returns:
            DataFrame: dataframe containing the artifact's data
        """
        csv = self.data.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, new_data: bytes) -> bytes:
        """
        Save new data and return it.

        Args:
            new_data: data to be saved

        Returns:
            bytes: the saved data
        """
        self.data = new_data
        return self.data
