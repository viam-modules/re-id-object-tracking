import json
import sqlite3
from typing import Dict, List

import numpy as np
import torch

from src.config.config import TracksManagerConfig
from src.tracker.track import Track

tables: Dict[str, str] = {
    "tracks": """
        CREATE TABLE IF NOT EXISTS tracks(
            id CHAR PRIMARY KEY,
            bbox BLOB,
            embedding BLOB
        )
    """,
    "category_count": """
        CREATE TABLE IF NOT EXISTS category_count(
            id CHAR PRIMARY KEY,
            count INTEGER
        )
    """,
    "tracks_ids_with_label": """
        CREATE TABLE IF NOT EXISTS tracks_ids_with_label(
            label CHAR PRIMARY KEY,
            ids TEXT
        )
    """,
}


class TracksManager:
    # TODO: keep track of the updated tracks since last sync so we don't serialize and write everything
    def __init__(self, config: TracksManagerConfig):
        self.tracks_on_disk: Dict[str, Track] = {}
        self.category_count_on_disk: Dict[str, int] = {}
        self.path_to_database = config.path_to_db.value
        self.track_ids_with_label_on_disk: Dict[str, List[str]] = {}
        self.save_period = config.save_period
        self.con: sqlite3.Connection = sqlite3.connect(self.path_to_database)
        for table_name in tables.keys():
            instantiate_table(table_name, self.con)

    def close(self):
        self.con.close()

    def parse_tracks_on_disk(self):
        # Fetch all track records from the database, including bbox
        res = self.con.execute("SELECT * FROM tracks").fetchall()

        for serialized_track in res:
            track = self.get_new_track_from_bytes(serialized_track)
            self.tracks_on_disk[track.track_id] = track

    @staticmethod
    def get_new_track_from_bytes(serialized_track: tuple):
        if len(serialized_track) != 3:
            raise ValueError(
                "can't deserialize track, too much or too few values to unpack."
            )

        track_id, bbox_blob, embedding_blob = serialized_track

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_vector = np.frombuffer(embedding_blob, dtype=np.float32).reshape(
            512,
        )
        feature_vector = torch.from_numpy(feature_vector)
        feature_vector = feature_vector.to(device).contiguous()

        return Track(
            track_id=track_id,
            bbox=np.frombuffer(bbox_blob, dtype=int).reshape(
                4,
            ),
            feature_vector=feature_vector,
            distance=None,
            is_candidate=False,
        )

    def get_tracks_on_disk(self):
        return self.tracks_on_disk

    def write_tracks_on_db(self, tracks: Dict[str, Track]):
        for track_id, track in tracks.items():
            # Get the serialized data from each track
            track_id, bbox_blob, embedding_blob = track.serialize()

            # Insert the track data into the database
            self.con.execute(
                "INSERT OR REPLACE INTO tracks(id, bbox, embedding) VALUES(?, ?, ?)",
                (
                    track_id,
                    bbox_blob,
                    embedding_blob,
                ),
            )

        self.con.commit()

    def parse_category_count_on_disk(self):
        res = self.con.execute("SELECT * FROM category_count").fetchall()
        for category, count in res:
            self.category_count_on_disk[category] = count

    def get_category_count_on_disk(self):
        return self.category_count_on_disk

    def write_category_count_on_db(self, category_count: Dict[str, int]):
        for category, count in category_count.items():
            self.con.execute(
                "INSERT OR REPLACE INTO category_count(id, count) VALUES(?, ?)",
                (category, count),
            )
        self.con.commit()

    def parse_map_label_track_ids(self):
        # Fetch all label-track_id pairs from the database
        res = self.con.execute("SELECT * FROM tracks_ids_with_label").fetchall()
        for label, track_ids_json in res:
            # Deserialize the JSON string back to a list of track_ids
            self.track_ids_with_label_on_disk[label] = json.loads(track_ids_json)

    def get_track_ids_with_label_on_disk(self):
        return self.track_ids_with_label_on_disk

    def write_track_ids_with_label_on_db(
        self, track_ids_with_label: Dict[str, List[str]]
    ):
        for label, track_ids in track_ids_with_label.items():
            # Serialize the list of track_ids as a JSON string
            track_ids_json = json.dumps(track_ids)

            # Insert the label and serialized track_ids into the database
            self.con.execute(
                "INSERT OR REPLACE INTO tracks_ids_with_label(label, ids) VALUES(?, ?)",
                (label, track_ids_json),
            )
        self.con.commit()


def instantiate_table(table_name: str, con: sqlite3.Connection):
    con.execute(tables.get(table_name))
