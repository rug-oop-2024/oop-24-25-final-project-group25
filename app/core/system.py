from __future__ import annotations
from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """
    Class representing a registry of artifacts.

    Attributes:
        _database(Database): database containing the json representations of
            stored artifacts
        _storage(Storage): sorage storing the data of the artifacts
    """
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """Initialise and ArtifactRegistry instance."""
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Register a new artifact in the registry.

        Args:
            artifact: artifact to store

        Returns:
            None
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
            "id": artifact.id
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        List all entries of a given type.

        Args:
            type: type of the entries to list

        Returns:
            list: list of listed artifacts
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
                id=data["id"]
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Get a specific artifact based on its id.

        Args:
            artifact_id: id of the artifact to get

        Returns:
            Artifact: artifact to return
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
            id=data["id"]
        )

    def delete(self, artifact_id: str) -> None:
        """
        Delete an indicated artifact.

        Args:
            artifact_id: id of artifact to delete

        Returns:
            None
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    Singleton class giving access to a common storage, database, and registry.

    Attributes:
        _instance(None|AutoMLSystem): class attribute representing the
            current instance of the class in existence
        _storage(LocalStorage): a LocalStorage instance
        _database(Database): a Database instance
        _registry(ArtifactRegistry): an ArtifactRegistry instance
    """
    _instance: None | AutoMLSystem = None

    def __init__(self, storage: LocalStorage, database: Database):
        """Initialise an instance of the class"""
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> AutoMLSystem:
        """
        Method used for the singeton strategy. ALways return the first instance
        made. If no such instance, make a new one.

        Returns:
            AutoMLSystem: instance of AutoMLSystem
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Return the registry of the class.

        Returns:
            ArtifactRegistry: the registry of artifacts
        """
        return self._registry
