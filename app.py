import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# API and connection constants
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "https://api.llama.com/v1/chat/completions")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Flask app
app = Flask(__name__)


class Neo4jDatabase:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._labels = None
        self._relationship_types = None
        self._property_keys = None

    def close(self):
        self.driver.close()

    def get_metadata(self):
        """Get database metadata (labels, relationship types, property keys)"""
        with self.driver.session() as session:
            # Get node labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]

            # Get relationship types
            rel_result = session.run("CALL db.relationshipTypes()")
            relationships = [record["relationshipType"] for record in rel_result]

            # Get property keys
            prop_result = session.run("CALL db.propertyKeys()")
            properties = [record["propertyKey"] for record in prop_result]

            return {
                "labels": labels,
                "relationship_types": relationships,
                "property_keys": properties,
            }

    def get_schema(self) -> str:
        """Get the database schema information"""
        metadata = self.get_metadata()

        schema_info = "Database Schema:\n"
        schema_info += f"Node Labels: {', '.join(metadata['labels'])}\n"
        schema_info += (
            f"Relationship Types: {', '.join(metadata['relationship_types'])}\n"
        )
        schema_info += f"Property Keys: {', '.join(metadata['property_keys'])}\n\n"

        # Get sample data structure for each node label
        with self.driver.session() as session:
            schema_info += "Node Structure Examples:\n"
            for label in metadata["labels"]:
                sample_result = session.run(f"MATCH (n:{label}) RETURN n LIMIT 1")
                for record in sample_result:
                    node = record["n"]
                    schema_info += f"Label '{label}' properties: {dict(node.items())}\n"

            # Get sample relationship structure
            schema_info += "\nRelationship Structure Examples:\n"
            for rel_type in metadata["relationship_types"]:
                rel_sample = session.run(
                    f"MATCH ()-[r:{rel_type}]->() RETURN r LIMIT 1"
                )
                for record in rel_sample:
                    rel = record["r"]
                    schema_info += (
                        f"Relationship '{rel_type}' properties: {dict(rel.items())}\n"
                    )

            return schema_info

    def execute_query(
        self, cypher_query: str
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Execute a Cypher query and return the results and error if any"""
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query)
                records = [
                    {
                        key: self._convert_neo4j_types(record[key])
                        for key in record.keys()
                    }
                    for record in result
                ]
                return records, None  # Return results and no error
            except Exception as e:
                print(f"Error executing query: {e}")
                # Return empty results and the error message
                return [], str(e)

    def _convert_neo4j_types(self, value):
        """Convert Neo4j types to Python types"""
        if hasattr(value, "items") and callable(getattr(value, "items")):
            # Convert Neo4j Node, Relationship to dict
            return dict(value.items())
        elif hasattr(value, "__iter__") and not isinstance(value, str):
            # Convert iterables
            return [self._convert_neo4j_types(v) for v in value]
        else:
            # Return value as is
            return value

    def expand_entity(self, entity_name: str) -> Dict:
        """Get entities related to the specified entity and their relationships"""
        # Query to get all direct relationships
        cypher_query = """
        MATCH (center:Entity {name: $entity_name})-[r]-(related)
        RETURN center, r, related
        LIMIT 20
        """

        with self.driver.session() as session:
            try:
                result = session.run(cypher_query, entity_name=entity_name)

                # Process results to create graph data
                nodes = {}
                edges = []

                # First add the center entity
                for record in result:
                    center = record["center"]
                    if center["name"] not in nodes:
                        nodes[center["name"]] = {
                            "id": center["name"],
                            "label": center["name"],
                            "type": center.get("type", "entity"),
                            "properties": dict(center),
                        }

                    related = record["related"]
                    # Determine node type based on labels
                    rel_type = related.get("type", "unknown")
                    if "episode_id" in related:
                        rel_type = "episode"
                    elif "podcast_id" in related:
                        rel_type = "podcast"

                    if related.get("name", "") not in nodes:
                        nodes[
                            related.get(
                                "name", str(related.get("episode_id", "unknown"))
                            )
                        ] = {
                            "id": related.get(
                                "name", str(related.get("episode_id", "unknown"))
                            ),
                            "label": related.get(
                                "name", related.get("episode_title", "Unknown")
                            ),
                            "type": rel_type,
                            "properties": dict(related),
                        }

                    # Extract relationship
                    rel = record["r"]
                    rel_type = rel.get("type", type(rel).__name__)

                    # Create edge
                    edges.append(
                        {
                            "from": center["name"],
                            "to": related.get(
                                "name", str(related.get("episode_id", "unknown"))
                            ),
                            "label": rel_type,
                            "properties": dict(rel),
                        }
                    )

                return {"nodes": list(nodes.values()), "edges": edges}

            except Exception as e:
                print(f"Error expanding entity: {e}")
                return {"nodes": [], "edges": []}


def generate_fallback_query(db_metadata) -> str:
    """Generate a simple but valid fallback query based on the database metadata"""
    # Try to find Entity label first
    if "Entity" in db_metadata["labels"]:
        entity_label = "Entity"
    else:
        # Otherwise use the first available label
        entity_label = db_metadata["labels"][0] if db_metadata["labels"] else "Entity"

    # Use MENTIONED_IN relationship if available
    if "MENTIONED_IN" in db_metadata["relationship_types"]:
        rel_type = "MENTIONED_IN"
    # Or CEO_OF if available
    elif "CEO_OF" in db_metadata["relationship_types"]:
        rel_type = "CEO_OF"
    # Or the first available relationship
    elif db_metadata["relationship_types"]:
        rel_type = db_metadata["relationship_types"][0]
    else:
        rel_type = "RELATED_TO"

    return f"""
    MATCH (e:{entity_label})-[r]->(m)
    RETURN e, r, m
    LIMIT 10
    """


def clean_cypher_query(raw_query: str) -> str:
    """Clean and extract a valid Cypher query from raw text"""
    # Remove markdown code blocks
    query = raw_query
    if "```" in query:
        # Extract content from code block
        pattern = r"```(?:cypher)?(.*?)```"
        matches = re.findall(pattern, query, re.DOTALL)
        if matches:
            query = matches[0]

    # Remove explanatory text and comments
    cleaned_lines = []
    for line in query.strip().split("\n"):
        line = line.strip()
        # Skip empty lines or lines that are clearly explanations or comments
        if (
            not line
            or line.startswith("Here")
            or line.startswith("//")
            or line.startswith("--")
        ):
            continue
        if "corrected query" in line.lower() or "fixed query" in line.lower():
            continue
        cleaned_lines.append(line)

    # Join the remaining lines
    cleaned_query = "\n".join(cleaned_lines)

    # If there appears to be explanatory text after a complete query, truncate it
    # Look for common patterns that indicate the end of a query
    end_patterns = [r"LIMIT\s+\d+", r"RETURN.*?;", r"RETURN[^;]*$"]

    for pattern in end_patterns:
        match = re.search(pattern, cleaned_query, re.IGNORECASE)
        if match:
            end_pos = match.end()
            if end_pos < len(cleaned_query):
                # There's text after the query end, truncate it
                cleaned_query = cleaned_query[:end_pos]
            break

    # Remove trailing semicolons and whitespace
    cleaned_query = cleaned_query.rstrip(";").strip()

    return cleaned_query


def get_cypher_from_question(
    question: str, schema_info: str, db_metadata: dict, error_message: str = None
) -> str:
    """Generate a Cypher query from a natural language question using Llama"""

    # Base prompt for initial query generation
    base_prompt = f"""### SYSTEM
You are a Neo4j Cypher query generation expert. Your job is to convert natural language questions 
about podcast data into precise Cypher queries.

Here is the database schema information:
{schema_info}

Available node labels: {', '.join(db_metadata['labels'])}
Available relationship types: {', '.join(db_metadata['relationship_types'])}

You must ONLY use these exact labels and relationship types in your query.

Generate a Cypher query that would answer the user's question. Your query should:
1. Only use node labels that exist in the database
2. Only use relationship types that exist in the database
3. Return complete nodes rather than just individual properties
4. Include variables for both nodes and the relationships between them
5. Always return the relationship in the RETURN clause

IMPORTANT: Return ONLY the Cypher query itself, with no additional explanation, comments, or text before or after.
DO NOT include any markdown formatting.

### USER
{question}
"""

    # If we have an error message, create a correction prompt
    if error_message:
        prompt = f"""### SYSTEM
You are a Neo4j Cypher query debugging expert. A previous query attempt failed with the following error:

ERROR: {error_message}

Please fix the query to make it syntactically correct and executable in Neo4j. Pay close attention to:
1. Only use node labels that exist in the database: {', '.join(db_metadata['labels'])}
2. Only use relationship types that exist in the database: {', '.join(db_metadata['relationship_types'])}
3. Make sure all variables used in the RETURN clause are defined in the MATCH pattern
4. Ensure proper relationship direction in MATCH patterns
5. Use single quotes for string literals
6. Do not include any explanatory text in the query itself

IMPORTANT: Return ONLY the fixed Cypher query itself, with NO explanation or commentary.
Simply output the exact query that should be executed.

### USER
Generate a working Cypher query to answer this question: {question}
"""
    else:
        prompt = base_prompt

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLAMA_API_KEY}",
    }

    data = {
        "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(LLAMA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()

        if (
            "completion_message" in response_data
            and "content" in response_data["completion_message"]
        ):
            content = response_data["completion_message"]["content"]["text"]

            # Clean and extract the actual Cypher query
            cleaned_query = clean_cypher_query(content)

            # If the query is too short, it's probably incomplete or invalid
            if len(cleaned_query) < 10:
                # Fall back to a simple but likely valid query based on the schema
                return generate_fallback_query(db_metadata)

            return cleaned_query

        return generate_fallback_query(db_metadata)  # Default fallback query
    except Exception as e:
        print(f"Error calling Llama API for query generation: {e}")
        return generate_fallback_query(db_metadata)  # Default fallback query


def generate_answer(
    question: str, query_results: List[Dict[str, Any]], error: Optional[str] = None
) -> str:
    """Generate a natural language answer from query results using Llama"""
    # Convert results to a readable format
    results_str = json.dumps(query_results, indent=2)

    # Include error information if applicable
    error_context = ""
    if error:
        error_context = f"\nNote: The query encountered an error: {error}\n"

    prompt = f"""### SYSTEM
You are a helpful podcast data analysis assistant. Your task is to answer user questions based on 
the results of a database query. Provide a conversational, informative response that addresses the 
user's question directly.

### USER
My question was: "{question}"
{error_context}
The database returned these results:
{results_str}

Please explain what these results mean in relation to my question. If the results are empty or contain
an error, please explain that the data might not be available or suggest how I might rephrase my question.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLAMA_API_KEY}",
    }

    data = {
        "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(LLAMA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()

        if (
            "completion_message" in response_data
            and "content" in response_data["completion_message"]
        ):
            return response_data["completion_message"]["content"]["text"]

        return "I couldn't interpret the results. Please try asking your question differently."
    except Exception as e:
        print(f"Error calling Llama API for answer generation: {e}")
        return "Sorry, I encountered an error while generating your answer. Please try again later."


def process_results_for_visualization(
    query_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Transform query results into graph visualization format"""
    nodes = {}
    edges = []

    # Extract nodes and edges from results
    for record in query_results:
        # Process each key in the record
        for key, value in record.items():
            # Skip if not a dict or None
            if not isinstance(value, dict) or value is None:
                continue

            # Identify nodes
            if key != "r" and isinstance(value, dict):  # 'r' is typically relationship
                # Extract node properties
                node_id = value.get(
                    "name", value.get("episode_id", value.get("podcast_id", key))
                )

                if not node_id:
                    continue

                # Determine node type
                node_type = value.get("type", "unknown")
                if "episode_id" in value:
                    node_type = "episode"
                elif "podcast_id" in value:
                    node_type = "podcast"

                # Create node label
                node_label = value.get(
                    "name",
                    value.get(
                        "episode_title", value.get("podcast_title", str(node_id))
                    ),
                )

                # Skip if we already have this node
                if node_id in nodes:
                    continue

                # Add node
                nodes[node_id] = {
                    "id": node_id,
                    "label": node_label,
                    "type": node_type,
                    "properties": value,
                }

    # Extract relationships
    for record in query_results:
        # Find source and target nodes
        source = None
        target = None
        relationship = None

        for key, value in record.items():
            if not isinstance(value, dict) or value is None:
                continue

            if key == "r":
                relationship = value
                continue

            # First found node is source, second is target
            if source is None:
                source_id = value.get(
                    "name", value.get("episode_id", value.get("podcast_id", "unknown"))
                )
                if source_id in nodes:
                    source = source_id
            elif target is None:
                target_id = value.get(
                    "name", value.get("episode_id", value.get("podcast_id", "unknown"))
                )
                if target_id in nodes and target_id != source:
                    target = target_id

        # If we found source, target and they're different, create edge
        if source and target and source != target:
            # Determine relationship type
            rel_type = "RELATED_TO"
            if relationship and "type" in relationship:
                rel_type = relationship["type"]

            # Create unique edge ID
            edge_id = f"{source}_{rel_type}_{target}"

            # Add edge
            edges.append(
                {
                    "id": edge_id,
                    "from": source,
                    "to": target,
                    "label": rel_type,
                    "properties": relationship if relationship else {},
                }
            )

    return {"nodes": list(nodes.values()), "edges": edges}


# Initialize Neo4j connection
db = Neo4jDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Get the schema info once at startup
schema_info = db.get_schema()
db_metadata = db.get_metadata()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Maximum number of query generation attempts
    max_attempts = 3
    attempt = 0
    error_message = None
    error = None  # Initialize error
    query_results = []  # Initialize query_results

    while attempt < max_attempts:
        attempt += 1

        # Generate Cypher query (with error feedback if this is a retry)
        cypher_query = get_cypher_from_question(
            question, schema_info, db_metadata, error_message
        )

        # Execute the query against Neo4j
        query_results, error = db.execute_query(cypher_query)

        # If no error or we've reached max attempts, exit the loop
        if not error or attempt >= max_attempts:
            break

        # Update error message for the next attempt
        error_message = error
        print(f"Query attempt {attempt} failed. Error: {error}")
        print(f"Retrying with error feedback...")

    # If all attempts failed, try a simpler fallback query
    if error and attempt >= max_attempts:
        print("All query attempts failed, trying a simple fallback query")
        cypher_query = generate_fallback_query(db_metadata)
        query_results, error = db.execute_query(cypher_query)

    # Generate a natural language answer from the results
    answer = generate_answer(question, query_results, error)

    # Process results for visualization
    graph_data = (
        process_results_for_visualization(query_results)
        if query_results
        else {"nodes": [], "edges": []}
    )

    return jsonify(
        {
            "question": question,
            "cypher_query": cypher_query,
            "results": query_results,
            "answer": answer,
            "graph_data": graph_data,
            "error": error,
        }
    )


@app.route("/schema")
def get_db_schema():
    return jsonify({"schema": schema_info})


@app.route("/expand_entity/<entity_name>")
def expand_entity_route(entity_name):
    """Expand a specific entity and its relationships"""
    graph_data = db.expand_entity(entity_name)
    return jsonify(graph_data)


if __name__ == "__main__":
    try:
        app.run(debug=True, port=8080)
    finally:
        db.close()
