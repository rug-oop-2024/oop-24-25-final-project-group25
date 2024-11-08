from autoop.core.ml.artifact import Artifact
import pandas as pd


class Dataset(Artifact):
    """
    Class representing a dataset.
    """


class Dataset(Artifact):
    """A class to represent an ML dataset"""
    def __init__(self, *args, **kwargs):
        """Initialize a dataset object"""
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod

    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        id: str = None,
        version: str = "1.0.0",
    ):
        """
        Construct a dataset from a given dataframe.

        Args:
            data: provided dataframe
            name: name of the dataset
            asset_path: path of the asset corresponding to the dataset
            id: id of the dataset in the database
            version: version of the dataset

        Returns:
            Dataste: created dataset
        """

        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
            id=id,
        )

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save data into the database.

        Args:
            data: dataframe containing the data.

        Returns:
            bytes: the encoded version of the saved data.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)

    def __repr__(self) -> str:
        return self.name

