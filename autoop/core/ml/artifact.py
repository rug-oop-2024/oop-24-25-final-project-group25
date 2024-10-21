from pydantic import BaseModel, Field
import base64

class Artifact():
    """
    Class representing an artifact.

    Attributes:
        type:
        name:
        version:
        asset_path:
        data:
        tags:
        metadata:
    """
    def __init__(self, type=None, name=None, version=None, asset_path=None, data=None, tags=None, metadata=None):
        """
        Initialize an Artifact object.
        """
        self.type = type
        self.name = name
        self.version = version
        self.asset_path = asset_path
        self.data = data
        self.tags = tags
        self.metadata = metadata

    def read(self) -> bytes:
        return self.data

    def save(self) -> bytes:
        pass








