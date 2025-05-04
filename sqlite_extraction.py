#!/usr/bin/env python3
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Item:
    """Class representing an episode item."""

    id: int
    podcast_id: int
    title: str
    slug: str
    description: str
    pub_date_timestamp: int
    item_pub_date: str

    def to_dict(self):
        """Convert the item to a dictionary."""
        return {
            "id": self.id,
            "podcast_id": self.podcast_id,
            "title": self.title,
            "slug": self.slug,
            "description": self.description,
            "pub_date_timestamp": self.pub_date_timestamp,
            "item_pub_date": self.item_pub_date,
        }

    @classmethod
    def from_dict(cls, data):
        """Create an Item from a dictionary."""
        return cls(
            id=data["id"],
            podcast_id=data["podcast_id"],
            title=data["title"],
            slug=data["slug"],
            description=data["description"],
            pub_date_timestamp=data["pub_date_timestamp"],
            item_pub_date=data["item_pub_date"],
        )


@dataclass
class Podcast:
    """Class representing a podcast."""

    id: int
    title: str
    slug: str
    image_url: str
    url: str
    language: str

    def to_dict(self):
        """Convert the podcast to a dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "image_url": self.image_url,
            "url": self.url,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Podcast from a dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            slug=data["slug"],
            image_url=data["image_url"],
            url=data["url"],
            language=data["language"],
        )


@dataclass
class Person:
    """Class representing a person."""

    id: int
    name: str
    slug: str
    description: str

    def to_dict(self):
        """Convert the person to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Person from a dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            slug=data["slug"],
            description=data["description"],
        )


@dataclass
class Transcription:
    """Class representing a transcription segment."""

    id: int
    episode_id: int
    from_ts: str
    to_ts: str
    text: str
    person_id: int

    def to_dict(self):
        """Convert the transcription to a dictionary."""
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "from_ts": self.from_ts,
            "to_ts": self.to_ts,
            "text": self.text,
            "person_id": self.person_id,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Transcription from a dictionary."""
        return cls(
            id=data["id"],
            episode_id=data["episode_id"],
            from_ts=data["from_ts"],
            to_ts=data["to_ts"],
            text=data["text"],
            person_id=data["person_id"],
        )


@dataclass
class EpisodeData:
    """Class representing all episode data."""

    item: Item
    podcast: Podcast
    persons: List[Person] = field(default_factory=list)
    transcriptions: List[Transcription] = field(default_factory=list)

    def to_dict(self):
        """Convert the episode data to a dictionary."""
        return {
            "item": self.item.to_dict(),
            "podcast": self.podcast.to_dict(),
            "persons": [person.to_dict() for person in self.persons],
            "transcriptions": [
                transcription.to_dict() for transcription in self.transcriptions
            ],
        }

    def to_json(self, indent=2):
        """Convert the episode data to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data):
        """Create an EpisodeData instance from a dictionary."""
        return cls(
            item=Item.from_dict(data["item"]),
            podcast=Podcast.from_dict(data["podcast"]),
            persons=[Person.from_dict(p) for p in data["persons"]],
            transcriptions=[Transcription.from_dict(t) for t in data["transcriptions"]],
        )

    @classmethod
    def from_json(cls, json_str):
        """Create an EpisodeData instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, filename):
        """Load episode data from a JSON file."""
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_file(self, filename):
        """Save the episode data to a JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Successfully saved episode data to '{filename}'")


class EpisodeExtractor:
    def __init__(self, db_path: str = "hackathon.db"):
        """Initialize the episode extractor with the database path."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            # Return rows as dictionaries
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            sys.exit(1)

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def get_item(self, episode_id: int) -> Item:
        """Get the item data for the specified episode ID."""
        self.cursor.execute("SELECT * FROM items WHERE id = ?", (episode_id,))
        item_row = self.cursor.fetchone()
        if not item_row:
            print(f"Error: No episode found with ID {episode_id}")
            sys.exit(1)

        # Convert sqlite3.Row to Item class
        item_dict = dict(item_row)
        return Item(
            id=item_dict["id"],
            podcast_id=item_dict["podcast_id"],
            title=item_dict["title"],
            slug=item_dict["slug"],
            description=item_dict["description"],
            pub_date_timestamp=item_dict["pub_date_timestamp"],
            item_pub_date=item_dict["item_pub_date"],
        )

    def get_podcast(self, podcast_id: int) -> Podcast:
        """Get the podcast data for the specified podcast ID."""
        self.cursor.execute("SELECT * FROM podcasts WHERE id = ?", (podcast_id,))
        podcast_row = self.cursor.fetchone()
        if not podcast_row:
            print(f"Error: No podcast found with ID {podcast_id}")
            sys.exit(1)

        # Convert sqlite3.Row to Podcast class
        podcast_dict = dict(podcast_row)
        return Podcast(
            id=podcast_dict["id"],
            title=podcast_dict["title"],
            slug=podcast_dict["slug"],
            image_url=podcast_dict["image_url"],
            url=podcast_dict["url"],
            language=podcast_dict["language"],
        )

    def get_transcriptions(self, episode_id: int) -> List[Transcription]:
        """Get all transcriptions for the specified episode ID."""
        self.cursor.execute(
            "SELECT * FROM transcriptions WHERE episode_id = ?", (episode_id,)
        )
        transcription_rows = self.cursor.fetchall()

        # Convert sqlite3.Row objects to Transcription objects
        transcriptions = []
        for row in transcription_rows:
            row_dict = dict(row)
            transcription = Transcription(
                id=row_dict["id"],
                episode_id=row_dict["episode_id"],
                from_ts=row_dict["from_ts"],
                to_ts=row_dict["to_ts"],
                text=row_dict["text"],
                person_id=row_dict["person_id"],
            )
            transcriptions.append(transcription)

        return transcriptions

    def get_person(self, person_id: int) -> Optional[Person]:
        """Get person data for the specified person ID."""
        self.cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        person_row = self.cursor.fetchone()
        if not person_row:
            return None

        # Convert sqlite3.Row to Person class
        person_dict = dict(person_row)
        return Person(
            id=person_dict["id"],
            name=person_dict["name"],
            slug=person_dict["slug"],
            description=person_dict["description"],
        )

    def get_persons_from_transcriptions(
        self, transcriptions: List[Transcription]
    ) -> List[Person]:
        """Get unique persons involved in the transcriptions."""
        person_ids = set()
        for t in transcriptions:
            person_ids.add(t.person_id)

        persons = []
        for person_id in person_ids:
            person = self.get_person(person_id)
            if person:
                persons.append(person)

        return persons

    def extract_episode_data(self, episode_id: int) -> EpisodeData:
        """Extract all data related to the specified episode ID."""
        self.connect()

        try:
            # Get item (episode) data
            item = self.get_item(episode_id)

            # Get podcast data
            podcast = self.get_podcast(item.podcast_id)

            # Get transcriptions
            transcriptions = self.get_transcriptions(episode_id)

            # Get persons from transcriptions
            persons = self.get_persons_from_transcriptions(transcriptions)

            # Compile all data into EpisodeData class
            episode_data = EpisodeData(
                item=item,
                podcast=podcast,
                persons=persons,
                transcriptions=transcriptions,
            )

            return episode_data

        finally:
            self.close()


def main() -> None:
    """Main function to process command line arguments and run the extractor."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <episode_id> [output_file.json] [database_path]")
        sys.exit(1)

    try:
        episode_id = int(sys.argv[1])
    except ValueError:
        print(f"Error: Episode ID must be an integer.")
        sys.exit(1)

    output_file = "episode_data.json"
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    db_path = "hackathon.db"
    if len(sys.argv) >= 4:
        db_path = sys.argv[3]

    # Check if database file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        sys.exit(1)

    # Extract episode data
    extractor = EpisodeExtractor(db_path)
    episode_data = extractor.extract_episode_data(episode_id)

    # Save to file
    episode_data.save_to_file(output_file)


if __name__ == "__main__":
    main()
