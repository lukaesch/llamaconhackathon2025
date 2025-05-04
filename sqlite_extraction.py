#!/usr/bin/env python3
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple


class EpisodeExtractor:
    def __init__(self, db_path: str = "hackathon.db"):
        """Initialize the episode extractor with the database path."""
        self.db_path = db_path
        self.conn = Any 
        self.cursor = sqlite3.Cursor 

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

    def get_item(self, episode_id: int) -> Dict[str, Any]:
        """Get the item data for the specified episode ID."""
        self.cursor.execute("SELECT * FROM items WHERE id = ?", (episode_id,))
        item = self.cursor.fetchone()
        if not item:
            print(f"Error: No episode found with ID {episode_id}")
            sys.exit(1)
        
        # Convert sqlite3.Row to dict
        return {
            "id": item["id"],
            "podcast_id": item["podcast_id"],
            "title": item["title"],
            "slug": item["slug"],
            "description": item["description"],
            "pub_date_timestamp": item["pub_date_timestamp"],
            "item_pub_date": item["item_pub_date"]
        }

    def get_podcast(self, podcast_id: int) -> Dict[str, Any]:
        """Get the podcast data for the specified podcast ID."""
        self.cursor.execute("SELECT * FROM podcasts WHERE id = ?", (podcast_id,))
        podcast = self.cursor.fetchone()
        if not podcast:
            print(f"Error: No podcast found with ID {podcast_id}")
            sys.exit(1)
        
        return {
            "id": podcast["id"],
            "title": podcast["title"],
            "slug": podcast["slug"],
            "image_url": podcast["image_url"],
            "url": podcast["url"],
            "language": podcast["language"]
        }

    def get_transcriptions(self, episode_id: int) -> List[Dict[str, Any]]:
        """Get all transcriptions for the specified episode ID."""
        self.cursor.execute("SELECT * FROM transcriptions WHERE episode_id = ?", (episode_id,))
        transcriptions = self.cursor.fetchall()
        
        result = []
        for t in transcriptions:
            result.append({
                "id": t["id"],
                "episode_id": t["episode_id"],
                "from_ts": t["from_ts"],
                "to_ts": t["to_ts"],
                "text": t["text"],
                "person_id": t["person_id"]
            })
        
        return result

    def get_person(self, person_id: int) -> Dict[str, Any]:
        """Get person data for the specified person ID."""
        self.cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        person = self.cursor.fetchone()
        if not person:
            return None
        
        return {
            "id": person["id"],
            "name": person["name"],
            "slug": person["slug"],
            "description": person["description"]
        }

    def get_persons_from_transcriptions(self, transcriptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get unique persons involved in the transcriptions."""
        person_ids = set()
        for t in transcriptions:
            person_ids.add(t["person_id"])
        
        persons = []
        for person_id in person_ids:
            person = self.get_person(person_id)
            if person:
                persons.append(person)
        
        return persons

    def extract_episode_data(self, episode_id: int) -> Dict[str, Any]:
        """Extract all data related to the specified episode ID."""
        self.connect()
        
        try:
            # Get item (episode) data
            item = self.get_item(episode_id)
            
            # Get podcast data
            podcast = self.get_podcast(item["podcast_id"])
            
            # Get transcriptions
            transcriptions = self.get_transcriptions(episode_id)
            
            # Get persons from transcriptions
            persons = self.get_persons_from_transcriptions(transcriptions)
            
            # Compile all data
            episode_data = {
                "item": item,
                "podcast": podcast,
                "persons": persons,
                "transcriptions": transcriptions
            }
            
            return episode_data
        
        finally:
            self.close()

    def save_to_file(self, data: Dict[str, Any], filename: str) -> None:
        """Save the episode data to a JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved episode data to '{filename}'")
        except Exception as e:
            print(f"Error saving data to file: {e}")
            sys.exit(1)


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
    
    # Extract and save episode data
    extractor = EpisodeExtractor(db_path)
    episode_data = extractor.extract_episode_data(episode_id)
    extractor.save_to_file(episode_data, output_file)


if __name__ == "__main__":
    main()
