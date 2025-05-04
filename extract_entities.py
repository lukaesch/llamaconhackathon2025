import concurrent.futures
import json
import os
import sys
import threading
import time

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

from sqlite_extraction import EpisodeExtractor

# Load environment variables from .env file
load_dotenv()

# API and connection constants from environment variables
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "https://api.llama.com/v1/chat/completions")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
WRITE_LOG_FILES = os.getenv("WRITE_LOG_FILES", "true").lower() == "true"


def create_prompt(transcript):
    """Create the full prompt for the Llama API"""
    prompt = """### SYSTEM
You are a data-extraction assistant for Audioscrape.  
Your job:  
1. Read a podcast transcript (possibly truncated).  
2. Identify every entity that is **explicitly mentioned**.  
3. Classify each entity with the label set below.  
4. Assign sentiment (-1.0 … 1.0) with a one-sentence justification.  
5. Detect relationships (only those stated or clearly implied in the text).  
6. Return a single, valid JSON object that matches the schema given at the end.

Allowed entity types  
- person            (real human)  
- organization      (company, non-profit, government body, team, etc.)  
- product           (tangible or digital offering)  
- location          (geographical place)  
- concept           (idea, technology, trend, field of study)  
- event             (conference, launch, historical incident, match, etc.)

Allowed relationship types (use UPPER_SNAKE_CASE)  
- CEO_OF, FOUNDED, OWNS, PRODUCED_BY, LOCATED_IN, SUBSIDIARY_OF, PART_OF, MENTIONS, RELATED_TO …  
- Use a sensible generic name if nothing above fits, but stay concise (max 2 words).

### USER
Extract named entities, their sentiment, and their relationships from the following podcast data.
The transcript contains beside other fields the information from_ts and to_ts, which needed to be added to each mentions.

{}

---

**Return JSON only check that it is valid** with the structure:

{{
  "entities": [
    {{
      "name": "...",
      "type": "...",
      "description": "...",
      "sentiment": 0.0,
      "sentiment_explanation": "..."
    }}
  ],
  "relationships": [
    {{
      "source": "...",
      "target": "...",
      "relationship_type": "..."
    }}
  ],
  "mentions": [
    {{
      "entity_name": "...",
      "context": "...",
      "sentiment": 0.0,
      "sentiment_explanation": "..."
      "from_ts": 0,
      "to_ts": 0
      "text": "..."
    }}
  ]
}}
""".format(
        transcript
    )

    return prompt


JSON_SCHEMA_PROMPT_RESULT = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "sentiment": {"type": "number"},
                            "sentiment_explanation": {"type": "string"},
                        },
                        "required": [
                            "name",
                            "type",
                            "description",
                            "sentiment",
                            "sentiment_explanation",
                        ],
                    },
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "relationship_type": {"type": "string"},
                        },
                        "required": ["source", "target", "relationship_type"],
                    },
                },
                "mentions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_name": {"type": "string"},
                            "context": {"type": "string"},
                            "sentiment": {"type": "number"},
                            "sentiment_explanation": {"type": "string"},
                            "from_ts": {"type": "integer"},
                            "to_ts": {"type": "integer"},
                            "text": {"type": "string"},
                        },
                        "required": [
                            "entity_name",
                            "context",
                            "sentiment",
                            "sentiment_explanation",
                            "from_ts",
                            "to_ts",
                            "text",
                        ],
                    },
                },
            },
            "required": ["entities", "relationships", "mentions"],
        }
    },
}

from llama_api_client import LlamaAPIClient


def call_llama_api(prompt, i):

    client = LlamaAPIClient(
        api_key=LLAMA_API_KEY,  # This is the default and can be omitted
    )

    completion = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format=JSON_SCHEMA_PROMPT_RESULT,
    )
    wtf(
        "llama_response_" + str(i) + ".json",
        json.loads(json.dumps(completion.to_dict(), indent=2)),
    )
    # if completion.status == "429":
    #     time.sleep(60)  # Wait for 60 seconds before retrying
    #     log(i, "Rate limit exceeded. Waiting for 60 seconds...")
    #
    #     completion = client.chat.completions.create(
    #         model="Llama-4-Maverick-17B-128E-Instruct-FP8",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": prompt,
    #             }
    #         ],
    #         response_format=JSON_SCHEMA_PROMPT_RESULT,
    #     )
    return completion.to_dict()


def pause_operations(pause_event, should_pause):
    """Helper function to set or clear the pause event"""
    if should_pause:
        pause_event.set()
        log("", "Pausing all operations...")
    else:
        pause_event.clear()
        log("", "Resuming all operations...")


def call_llama_api_raw(prompt, i):
    """Call the Llama API with the given prompt"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLAMA_API_KEY}",
    }

    data = {
        "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": JSON_SCHEMA_PROMPT_RESULT,
    }

    try:
        response = requests.post(LLAMA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        log(i, f"Error calling Llama API: {e}")
        if hasattr(e, "response") and e.response:
            log(i, f"Response status code: {e.response.status_code}")
            wtf(
                "llama_response_" + str(i) + ".json",
                f"Response text: {e.response.text}",
            )
        return None


def extract_json_from_response(response, i):
    """Extract JSON content from the Llama API response"""
    try:
        if (
            "completion_message" in response
            and "content" in response["completion_message"]
        ):
            content = response["completion_message"]["content"]
            if "text" in content:
                # The text may contain markdown code blocks, so extract just the JSON part
                text = content["text"]

                # If the text is wrapped in a code block, extract just the JSON
                if text.startswith("```json") and text.endswith("```"):
                    text = text[7:-3].strip()
                elif text.startswith("```") and text.endswith("```"):
                    text = text[3:-3].strip()

                # Try to parse the text as JSON
                return json.loads(text)

        # Fallback if the structure doesn't match expectations
        wtf(
            "llama_response_error_" + str(i) + ".txt",
            "Unexpected response structure:",
            json.dumps(response, indent=2),
        )
        return None
    except json.JSONDecodeError as e:
        log(i, f"Error parsing JSON from response: {e}")
        wtf("response_parsing_error_" + str(i) + ".txt", f"{response}")
        return None


class Neo4jDatabase:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def insert_data(self, data):
        episode = data.get("item")
        podcast = data.get("podcast")
        podcast_id = podcast.get("id")
        episode_id = episode.get("id")
        podcast_title = podcast.get("title", "ERROR: Unknown Podcast")
        episode_title = episode.get("title", "ERROR: Unknown Episode")
        release_date = episode.get("pub_date_timestamp")

        with self.driver.session() as session:
            # Create podcast node if it doesn't exist
            session.run(
                """
                MERGE (p:Podcast {podcast_id: $podcast_id})
                SET p.podcast_title = $podcast_title
            """,
                podcast_id=podcast_id,
                podcast_title=podcast_title,
            )

            # Create episode node if it doesn't exist
            session.run(
                """
                MERGE (e:Episode {episode_id: $episode_id})
                SET e.episode_title = $episode_title,
                    e.release_date = $release_date
                WITH e
                MATCH (p:Podcast {podcast_id: $podcast_id})
                MERGE (p)-[:HAS_EPISODE]->(e)
            """,
                episode_id=episode_id,
                episode_title=episode_title,
                release_date=release_date,
                podcast_id=podcast_id,
            )

            # Create entities
            for entity in data.get("entities", []):
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.description = $description,
                        e.sentiment = $sentiment,
                        e.sentiment_explanation = $sentiment_explanation
                    WITH e
                    MATCH (ep:Episode {episode_id: $episode_id})
                    MERGE (e)-[:MENTIONED_IN]->(ep)
                """,
                    name=entity.get("name", "Unknown"),
                    type=entity.get("type", "Unknown"),
                    description=entity.get("description", ""),
                    sentiment=entity.get("sentiment", 0.0),
                    sentiment_explanation=entity.get("sentiment_explanation", ""),
                    episode_id=episode_id,
                )

            # Create relationships between entities
            for rel in data.get("relationships", []):
                session.run(
                    """
                    MATCH (source:Entity {name: $source})
                    MATCH (target:Entity {name: $target})
                    MERGE (source)-[r:RELATES_TO {type: $rel_type}]->(target)
                """,
                    source=rel.get("source", ""),
                    target=rel.get("target", ""),
                    rel_type=rel.get("relationship_type", "RELATED_TO"),
                )

            # Create mentions
            for mention in data.get("mentions", []):
                session.run(
                    """
                    MATCH (e:Entity {name: $entity_name})
                    MATCH (ep:Episode {episode_id: $episode_id})
                    MERGE (m:Mention {
                        context: $context,
                        sentiment: $sentiment,
                        sentiment_explanation: $sentiment_explanation,
                        from_ts: $from_ts,
                        to_ts: $to_ts,
                        text: $text
                    })
                    MERGE (m)-[:REFERS_TO]->(e)
                    MERGE (m)-[:IN_EPISODE]->(ep)
                """,
                    entity_name=mention.get("entity_name", ""),
                    context=mention.get("context", ""),
                    sentiment=mention.get("sentiment", 0.0),
                    sentiment_explanation=mention.get("sentiment_explanation", ""),
                    episode_id=episode_id,
                    from_ts=mention.get("from_ts", 0),
                    to_ts=mention.get("to_ts", 0),
                    text=mention.get("text", ""),
                )


def extract_metadata(transcript):
    """Extract podcast metadata from transcript"""
    metadata = {}
    for line in transcript.split("\n"):
        if "podcast_id:" in line:
            metadata["podcast_id"] = line.split("podcast_id:")[1].strip()
        elif "podcast_title:" in line:
            metadata["podcast_title"] = line.split("podcast_title:")[1].strip()
        elif "episode_id:" in line:
            metadata["episode_id"] = line.split("episode_id:")[1].strip()
        elif "episode_title:" in line:
            metadata["episode_title"] = line.split("episode_title:")[1].strip()
        elif "release_date:" in line:
            metadata["release_date"] = line.split("release_date:")[1].strip()

    return metadata


episode_list = []


def process_episode(episode_id: str, episode_idx: int) -> None:
    """Process a single episode in parallel."""
    try:
        log(str(episode_idx), f"\nProcessing episode {episode_id}...")
        extractor = EpisodeExtractor("hackathon.db")
        episode_data = extractor.extract_episode_data(int(episode_id))
        if len(episode_data.transcriptions) == 0:
            log(str(episode_idx), "No transcriptions found for this episode.")
            return

        wtf(
            "podcast_episode_" + str(episode_idx) + ".txt",
            json.dumps(episode_data.to_dict(), indent=2),
        )

        # Create prompt with transcript
        prompt = create_prompt(json.dumps(episode_data.to_dict()))
        wtf("prompt_" + str(episode_idx) + "_.txt", prompt, "a")

        # Call Llama API
        log(str(episode_idx), "Calling Llama API...")
        response = None  # if lama is skipped temporarely
        response = call_llama_api(prompt, episode_idx)

        if response:
            # Extract JSON data from response
            data = extract_json_from_response(response, episode_idx)
            if data:
                # Add metadata to the data
                data.update(episode_data.to_dict())
                wtf(
                    "extracted_data_prompt_" + str(episode_idx) + ".txt",
                    json.dumps(data, indent=2),
                    "w",
                )

                # Insert data into Neo4j
                log(str(episode_idx), "Inserting data into Neo4j...")
                db = Neo4jDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
                try:
                    db.insert_data(data)
                    log(str(episode_idx), "Data successfully inserted into Neo4j!")
                except Exception as e:
                    log(str(episode_idx), f"Error inserting data into Neo4j: {e}")
                finally:
                    db.close()
            else:
                log(str(episode_idx), "Failed to extract JSON data from API response")
        else:
            log(str(episode_idx), "Failed to get response from Llama API")
    except Exception as e:
        log(str(episode_idx), f"Error processing episode {episode_id}: {e}")


def main():
    if not all([LLAMA_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        print(
            "Error: Missing required environment variables. Please check your .env file."
        )
        sys.exit(1)

    delete_directory_if_exists("logs")
    extractor = EpisodeExtractor("hackathon.db")
    extractor.connect()
    episode_list = extractor.get_all_ids()

    log("main", f"Episode list: {episode_list}")

    # Maximum number of workers (parallel processes)
    # You can adjust this based on your system's capabilities
    max_workers = min(32, os.cpu_count() * 2 + 4)

    # Process episodes in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = []

        # Submit tasks to the executor
        for i, episode_id in enumerate(episode_list, start=1):
            future = executor.submit(process_episode, str(episode_id), i)
            futures.append(future)

        # Wait for all futures to complete (optional, as the context manager will do this)
        concurrent.futures.wait(futures)

        # Check for exceptions
        for future in futures:
            try:
                future.result()  # This will raise any exceptions that occurred during execution
            except Exception as e:
                log("", f"An error occurred during parallel processing: {e}")


def wtf(filename: str, content: str, mode: str = "a"):
    """Write to a file with the given filename and content."""
    print(f"Writing to file {filename}")
    create_folder("logs")
    if WRITE_LOG_FILES:
        with open("./logs/" + filename, mode) as file:
            file.write(str(content) + "\n")
            file.close()


def log(pid: str, content: str):
    """Log content to a file with a timestamp."""
    if content.startswith("Error"):
        print(f"\033[91m{str(pid)}:{str(content)}\033[0m")
    else:
        print(str(pid) + ":" + str(content))
    wtf("log.txt", str(pid) + ":" + content, "a")


from pathlib import Path


def create_folder(path: str) -> None:
    """
    Create a folder at the given path if it doesn't already exist.

    Args:
        path (str): The folder path to create (can be nested).
    """
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)


import shutil


def delete_directory_if_exists(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Deleted directory: {dir_path}")
    else:
        print(f"Directory does not exist: {dir_path}")


if __name__ == "__main__":
    main()
