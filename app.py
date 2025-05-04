import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API and connection constants
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "https://api.llama.com/v1/chat/completions")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Check for required environment variables
if not all([LLAMA_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    missing = []
    if not LLAMA_API_KEY: missing.append("LLAMA_API_KEY")
    if not NEO4J_URI: missing.append("NEO4J_URI")
    if not NEO4J_USER: missing.append("NEO4J_USER")
    if not NEO4J_PASSWORD: missing.append("NEO4J_PASSWORD")
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

# Initialize Flask app
app = Flask(__name__)


class Neo4jDatabase:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._labels = None
        self._relationship_types = None
        self._property_keys = None
        logger.info("Neo4j database connection initialized")

    def close(self):
        self.driver.close()
        logger.info("Neo4j database connection closed")


    def get_metadata(self):
        """Get database metadata (labels, relationship types, property keys) without APOC"""
        logger.info("Fetching database metadata")
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

            # Get property keys by label (extended metadata)
            label_properties = {}
            for label in labels:
                try:
                    # Sample a node to get its properties
                    sample_result = session.run(f"MATCH (n:{label}) RETURN properties(n) as props LIMIT 1")
                    for record in sample_result:
                        props = record["props"]
                        label_properties[label] = list(props.keys())
                except Exception as e:
                    logger.warning(f"Error getting properties for label {label}: {e}")
                    label_properties[label] = []

            # Get relationship properties by type
            rel_properties = {}
            for rel_type in relationships:
                try:
                    # Sample a relationship to get its properties
                    rel_sample = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN properties(r) as props LIMIT 1")
                    for record in rel_sample:
                        props = record["props"]
                        rel_properties[rel_type] = list(props.keys())
                except Exception as e:
                    logger.warning(f"Error getting properties for relationship {rel_type}: {e}")
                    rel_properties[rel_type] = []

            # Get common relationships between node types using direct Cypher queries
            schema_relationships = []
            try:
                logger.info("Discovering schema relationships using Cypher queries")
                # For each pair of labels and relationship type, check if there are connections
                for start_label in labels:
                    for rel_type in relationships:
                        for end_label in labels:
                            # Check if this relationship pattern exists (limit search to avoid performance issues)
                            count_query = f"""
                            MATCH (start:{start_label})-[r:{rel_type}]->(end:{end_label})
                            RETURN count(r) as rel_count LIMIT 1
                            """
                            try:
                                count_result = session.run(count_query)
                                record = count_result.single()
                                if record and record["rel_count"] > 0:
                                    schema_relationships.append({
                                        "start": start_label,
                                        "relationship": rel_type,
                                        "end": end_label
                                    })
                                    logger.debug(f"Found relationship: ({start_label})-[{rel_type}]->({end_label})")
                            except Exception as e:
                                logger.warning(f"Error checking relationship {start_label}-{rel_type}->{end_label}: {e}")
                                continue
            except Exception as e:
                logger.warning(f"Error discovering schema relationships: {e}")

            logger.info(f"Retrieved metadata: {len(labels)} labels, {len(relationships)} relationship types, {len(properties)} property keys")
            logger.info(f"Discovered {len(schema_relationships)} relationship patterns in schema")
            
            return {
                "labels": labels,
                "relationship_types": relationships,
                "property_keys": properties,
                "label_properties": label_properties,
                "relationship_properties": rel_properties,
                "schema_relationships": schema_relationships
            }

    def get_schema(self) -> str:
        """Get the database schema information"""
        logger.info("Building database schema information")
        metadata = self.get_metadata()

        schema_info = "Database Schema:\n"
        schema_info += f"Node Labels: {', '.join(metadata['labels'])}\n"
        schema_info += (
            f"Relationship Types: {', '.join(metadata['relationship_types'])}\n"
        )
        schema_info += f"Property Keys: {', '.join(metadata['property_keys'])}\n\n"

        # Add label properties information
        schema_info += "Node Label Properties:\n"
        for label, props in metadata.get('label_properties', {}).items():
            schema_info += f"- {label}: {', '.join(props)}\n"
        
        # Add relationship properties information
        schema_info += "\nRelationship Properties:\n"
        for rel, props in metadata.get('relationship_properties', {}).items():
            schema_info += f"- {rel}: {', '.join(props)}\n"
            
        # Add schema relationships
        schema_info += "\nCommon Relationships:\n"
        for rel in metadata.get('schema_relationships', []):
            schema_info += f"- ({rel['start']})-[:{rel['relationship']}]->({rel['end']})\n"

        # Get sample data structure for each node label
        with self.driver.session() as session:
            schema_info += "\nNode Structure Examples:\n"
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
            
            logger.info("Database schema information built successfully")
            return schema_info

    def execute_query(
        self, cypher_query: str, params: Dict[str, Any] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Execute a Cypher query and return the results and error if any"""
        params = params or {}
        logger.info(f"Executing Cypher query: {cypher_query[:100]}...")
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query, **params)
                records = [
                    {
                        key: self._convert_neo4j_types(record[key])
                        for key in record.keys()
                    }
                    for record in result
                ]
                logger.info(f"Query executed successfully, returned {len(records)} records")
                return records, None  # Return results and no error
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                # Return empty results and the error message
                return [], str(e)

    def _convert_neo4j_types(self, value):
        """Convert Neo4j types to Python types"""
        if hasattr(value, "items") and callable(getattr(value, "items")):
            # Convert Neo4j Node, Relationship to dict
            result = dict(value.items())
            
            # For nodes, add type information
            if hasattr(value, "labels"):
                result["_labels"] = list(value.labels)
                # Add a single type field for easier processing
                result["_type"] = list(value.labels)[0] if value.labels else "Unknown"
            
            # For relationships, add type information
            if hasattr(value, "type"):
                result["_type"] = value.type
            
            # Add Neo4j ID for reference
            if hasattr(value, "id"):
                result["_id"] = value.id
                
            return result
        elif hasattr(value, "__iter__") and not isinstance(value, str):
            # Convert iterables
            return [self._convert_neo4j_types(v) for v in value]
        else:
            # Return value as is
            return value

    def expand_entity(self, entity_name: str, node_type: str = None) -> Dict:
        """Get entities related to the specified entity and their relationships"""
        logger.info(f"Expanding entity: {entity_name}, type: {node_type}")
        
        # Build a query based on available information
        conditions = []
        params = {"entity_name": entity_name}
        
        # Try to use the node type if provided
        if node_type:
            cypher_query = f"""
            MATCH (center:{node_type})-[r]-(related)
            WHERE center.name = $entity_name OR center.title = $entity_name OR 
                  center.podcast_title = $entity_name OR center.episode_title = $entity_name OR
                  center.id = $entity_name OR ID(center) = $entity_id
            RETURN center, r, related
            LIMIT 30
            """
        else:
            # More generic query if no type provided
            cypher_query = """
            MATCH (center)-[r]-(related)
            WHERE center.name = $entity_name OR center.title = $entity_name OR 
                  center.podcast_title = $entity_name OR center.episode_title = $entity_name OR
                  center.id = $entity_name OR ID(center) = $entity_id
            RETURN center, r, related
            LIMIT 30
            """

        # Try to parse entity_name as integer for ID lookup
        entity_id = None
        try:
            entity_id = int(entity_name)
            params["entity_id"] = entity_id
        except (ValueError, TypeError):
            pass

        with self.driver.session() as session:
            try:
                result = session.run(cypher_query, **params)

                # Process results to create graph data
                nodes = {}
                edges = []

                # Process the results
                for record in result:
                    center = record["center"]
                    related = record["related"]
                    rel = record["r"]
                    
                    # Get center node properties with type info
                    center_props = self._convert_neo4j_types(center)
                    center_type = center_props.get("_type", list(center.labels)[0] if center.labels else "Unknown")
                    
                    # Get center node ID
                    center_id = str(center.id)  # Use Neo4j ID as primary identifier
                    
                    # Get display label based on node type
                    center_label = self._get_best_display_label(center_props, center_type)
                    
                    if center_id not in nodes:
                        nodes[center_id] = {
                            "id": center_id,
                            "label": center_label,
                            "type": center_type,
                            "properties": center_props,
                        }

                    # Process related node
                    related_props = self._convert_neo4j_types(related)
                    related_type = related_props.get("_type", list(related.labels)[0] if related.labels else "Unknown")
                    
                    # Get related node ID
                    related_id = str(related.id)  # Use Neo4j ID as primary identifier
                    
                    # Get display label based on node type
                    related_label = self._get_best_display_label(related_props, related_type)
                    
                    if related_id not in nodes:
                        nodes[related_id] = {
                            "id": related_id,
                            "label": related_label,
                            "type": related_type,
                            "properties": related_props,
                        }

                    # Extract relationship type correctly
                    rel_type = rel.type
                    rel_props = self._convert_neo4j_types(rel)
                    
                    # Create edge
                    edge_id = f"{center_id}_{rel_type}_{related_id}"
                    edges.append({
                        "id": edge_id,
                        "from": center_id,
                        "to": related_id,
                        "label": rel_type,
                        "properties": rel_props,
                    })
                
                logger.info(f"Entity expansion complete. Found {len(nodes)} nodes and {len(edges)} edges")
                return {"nodes": list(nodes.values()), "edges": edges}

            except Exception as e:
                logger.error(f"Error expanding entity: {e}")
                return {"nodes": [], "edges": []}
    
    def _get_best_display_label(self, props, node_type):
        """Get the best property to use as a display label based on node type"""
        # Priority order of properties to use as labels
        if node_type == "Podcast":
            for prop in ["podcast_title", "title", "name", "id"]:
                if prop in props and props[prop]:
                    return str(props[prop])
        elif node_type == "Episode":
            for prop in ["episode_title", "title", "name", "id"]:
                if prop in props and props[prop]:
                    return str(props[prop])
        elif node_type == "Person":
            for prop in ["name", "full_name", "id"]:
                if prop in props and props[prop]:
                    return str(props[prop])
        elif node_type == "Topic" or node_type == "Concept":
            for prop in ["name", "title", "id"]:
                if prop in props and props[prop]:
                    return str(props[prop])
        
        # Generic fallback
        for prop in ["name", "title", "podcast_title", "episode_title", "id"]:
            if prop in props and props[prop]:
                return str(props[prop])
                
        # Last resort: use the node ID
        return f"{node_type}_{props.get('_id', 'unknown')}"


class QueryProcessor:
    """Processes queries with adaptive exploration based on results"""
    
    def __init__(self, db: Neo4jDatabase, schema_info: str, db_metadata: dict):
        self.db = db
        self.schema_info = schema_info
        self.db_metadata = db_metadata
        logger.info("QueryProcessor initialized")
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process a query with unified adaptive approach:
        1. Generate and execute the best direct query
        2. Perform dynamic exploration based on results
        3. Aggregate findings into a comprehensive answer
        """
        logger.info(f"Processing query: {question}")
        
        # Generate Cypher query
        cypher_query = self._generate_cypher_query(question)
        
        # Execute the query
        results, error = self.db.execute_query(cypher_query)
        
        # Try once more with error feedback if needed
        if error:
            logger.info(f"Initial query failed with error: {error}. Retrying with error feedback.")
            cypher_query = self._generate_cypher_query(question, error)
            results, error = self.db.execute_query(cypher_query)
        
        # Always perform dynamic exploration to enrich results
        logger.info("Performing dynamic exploration to enhance results")
        exploration_data = self._perform_dynamic_exploration(question, results, error)
        
        # Generate answer with all available data
        logger.info("Generating comprehensive answer")
        answer = self._generate_answer(question, results, exploration_data, error)
        
        # Process results for visualization
        graph_data = self._process_graph_from_all_data(results, exploration_data)
        
        logger.info(f"Query processing complete. Generated answer length: {len(answer)}")
        return {
            "question": question,
            "cypher_query": cypher_query,
            "results": results,
            "enhanced_data": exploration_data,
            "answer": answer,
            "graph_data": graph_data,
            "error": error
        }
    
    def _generate_cypher_query(self, question: str, error_message: str = None) -> str:
        """Generate a Cypher query from a natural language question using Llama with structured output"""
        logger.info(f"Generating Cypher query for: {question}")
        
        # Create context based on whether we're handling an error
        if error_message:
            logger.info(f"Including previous error feedback: {error_message}")
            system_content = f"""You are a Neo4j Cypher query debugging expert. A previous query attempt failed with the following error:

ERROR: {error_message}

Please fix the query to make it syntactically correct and executable in Neo4j. Pay close attention to:
1. Only use node labels that exist in the database: {', '.join(self.db_metadata['labels'])}
2. Only use relationship types that exist in the database: {', '.join(self.db_metadata['relationship_types'])}
3. Make sure all variables used in the RETURN clause are defined in the MATCH pattern
4. Ensure proper relationship direction in MATCH patterns
5. Use single quotes for string literals
6. For complex queries, ensure all aggregations are properly defined and grouped

Generate a working Cypher query to answer this question: {question}

CRITICAL REQUIREMENT:
Always return complete node objects and relationships in your query, not just properties.
This ensures proper visualization and more complete data for analysis.

For example:
- INSTEAD OF: RETURN n.name, n.description
- USE: RETURN n, n.name, n.description

- INSTEAD OF: MATCH (p)-[:HAS_EPISODE]->(e) RETURN e.title
- USE: MATCH (p)-[r:HAS_EPISODE]->(e) RETURN p, r, e, e.title"""
        else:
            system_content = f"""You are a Neo4j Cypher query generation expert. Your job is to convert natural language questions 
about podcast data into precise Cypher queries.

Here is the database schema information:
{self.schema_info[:1000]}  # Truncated to avoid token limits

Available node labels: {', '.join(self.db_metadata['labels'])}
Available relationship types: {', '.join(self.db_metadata['relationship_types'])}

You must ONLY use these exact labels and relationship types in your query.

Generate a Cypher query that would answer the user's question. Your query should be optimized for:

1. Complex pattern matching: Use multiple MATCH clauses for complex relationships
2. Return complete graph objects: Always return full node and relationship objects
3. Aggregation: Use COUNT, SUM, AVG, MIN, MAX when appropriate
4. Sorting: Use ORDER BY when the question asks for "top", "most", "least", etc.
5. Filtering: Use WHERE clauses with complex conditions when needed

CRITICAL REQUIREMENT:
Always return complete node objects and relationships in your query, not just properties.
This ensures proper visualization and more complete data for analysis.

Examples:
- INSTEAD OF: RETURN n.name, n.description
- USE: RETURN n, n.name, n.description

- INSTEAD OF: MATCH (p)-[:HAS_EPISODE]->(e) RETURN e.title
- USE: MATCH (p)-[r:HAS_EPISODE]->(e) RETURN p, r, e, e.title

The question is: {question}"""

        data = {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": f"Generate a Cypher query for: {question}"
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "cypher_query": {
                                "type": "string",
                                "description": "A valid executable Neo4j Cypher query that returns full node objects and relationships"
                            }
                        },
                        "required": ["cypher_query"]
                    }
                }
            },
            "temperature": 0.2  # Lower temperature for more deterministic output
        }

        try:
            logger.info("Sending Cypher generation request to Llama API")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLAMA_API_KEY}",
            }
            
            response = requests.post(LLAMA_API_URL, headers=headers, json=data)
            
            if response.status_code != 200:
                logger.error(f"Error from Llama API: Status {response.status_code}, Response: {response.text}")
                return self._generate_fallback_query()
            
            response_data = response.json()
            
            # Extract content from the response
            content = self._extract_content_from_llama_response(response_data)
            
            if content:
                try:
                    parsed_content = json.loads(content)
                    if "cypher_query" in parsed_content:
                        cypher_query = parsed_content["cypher_query"]
                        logger.info(f"Successfully generated Cypher query: {cypher_query[:100]}...")
                        return cypher_query
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}, content: {content}")
                    # Try to extract a Cypher query using regex as fallback
                    match = re.search(r"```cypher\s*(.*?)\s*```", content, re.DOTALL)
                    if match:
                        cypher_query = match.group(1).strip()
                        logger.info(f"Extracted Cypher query from code block: {cypher_query[:100]}...")
                        return cypher_query

            # If we couldn't get a structured response, fall back to a simple query
            logger.warning("Using fallback query")
            return self._generate_fallback_query()
        except Exception as e:
            logger.error(f"Error calling Llama API for query generation: {e}", exc_info=True)
            return self._generate_fallback_query()
    
    def _extract_content_from_llama_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Helper function to extract content from various Llama API response formats"""
        
        # Format 1: Standard OpenAI-like format
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            logger.debug(f"Extracted content from 'choices' format: {content[:100]}...")
            return content
        
        # Format 2: Llama-specific format
        if "completion_message" in response_data and "content" in response_data["completion_message"]:
            # Handle different content types
            content_obj = response_data["completion_message"]["content"]
            
            if isinstance(content_obj, dict):
                if "text" in content_obj:
                    logger.debug(f"Extracted content from 'completion_message.content.text' format: {content_obj['text'][:100]}...")
                    return content_obj["text"]
                elif "type" in content_obj and content_obj["type"] == "text":
                    logger.debug(f"Extracted content from 'completion_message.content' with type 'text': {content_obj['text'][:100]}...")
                    return content_obj["text"]
            elif isinstance(content_obj, str):
                logger.debug(f"Extracted string content from 'completion_message.content': {content_obj[:100]}...")
                return content_obj
        
        logger.warning(f"Could not extract content from response with keys: {list(response_data.keys())}")
        return None
    
    def _generate_fallback_query(self) -> str:
        """Generate a simple but valid fallback query based on the database metadata"""
        logger.info("Generating fallback query")
        
        # Try to find a suitable node label (prefer common entity-like labels)
        preferred_labels = ["Podcast", "Episode", "Entity", "Person", "Topic"]
        available_labels = self.db_metadata["labels"]
        
        chosen_label = None
        for label in preferred_labels:
            if label in available_labels:
                chosen_label = label
                break
        
        # If no preferred label found, use the first available one
        if not chosen_label and available_labels:
            chosen_label = available_labels[0]
        else:
            # Fallback with a generic label
            chosen_label = "Node"
        
        # Get a relationship type to use
        relationship_types = self.db_metadata.get("relationship_types", [])
        rel_type = relationship_types[0] if relationship_types else "RELATED_TO"
        
        # Create a query that returns full node objects and their relationships
        fallback_query = f"""
        MATCH (n:{chosen_label})
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT 15
        """
        
        logger.info(f"Generated fallback query: {fallback_query}")
        return fallback_query
    
    def _perform_dynamic_exploration(self, question: str, initial_results: List[Dict[str, Any]], 
                                   error: Optional[str]) -> Dict[str, Any]:
        """
        Use LLM to dynamically plan and execute an exploration strategy
        based on the question and initial results
        """
        logger.info(f"Performing dynamic exploration for: {question}")
        # Format initial results
        results_str = json.dumps(initial_results[:5], indent=2) if initial_results else "No results"
        if len(results_str) > 2000:
            results_str = results_str[:2000] + "... [truncated]"
        
        error_str = f"Error: {error}" if error else "No errors"
        
        # Create a context of available node labels and relationship types
        schema_context = f"""
        Available node labels: {', '.join(self.db_metadata['labels'])}
        Available relationship types: {', '.join(self.db_metadata['relationship_types'])}
        Common property keys: {', '.join(self.db_metadata['property_keys'][:20])}
        """
        
        # Use LLM to create an exploration plan
        logger.info("Generating exploration plan")
        exploration_plan = self._generate_exploration_plan(question, results_str, error_str, schema_context)
        
        # Execute the exploration plan
        logger.info(f"Executing exploration plan with {len(exploration_plan)} steps")
        exploration_results = self._execute_exploration_plan(exploration_plan)
        
        return {
            "exploration_plan": exploration_plan,
            "exploration_results": exploration_results
        }
    
    def _generate_exploration_plan(self, question: str, results_str: str, 
                                 error_str: str, schema_context: str) -> List[Dict[str, Any]]:
        """Generate a dynamic exploration plan using LLM with structured output"""
        logger.info(f"Generating exploration plan for: {question}")
        
        system_content = f"""You are an expert Neo4j graph database explorer. Your task is to create a dynamic exploration plan
to answer a user's question when the initial query results are insufficient or contain NULL values.

Question: "{question}"

Initial query results:
{results_str}

{error_str}

Database schema information:
{schema_context}

Create an exploration plan with 2-5 Cypher queries that will:
1. Identify key entities mentioned in the question
2. Explore their properties and relationships
3. Find alternative paths that might answer the question
4. Look for related information that fills in missing data

The exploration should be adaptive - if we don't find data one way, try another approach.

CRITICAL QUERY GUIDANCE:
When writing Cypher queries, ALWAYS return complete node and relationship objects in addition to any properties.
This ensures proper visualization of results and complete data for analysis.

Examples:
- INSTEAD OF: MATCH (n) RETURN n.name, n.description
- USE: MATCH (n) RETURN n, n.name, n.description

- INSTEAD OF: MATCH (p)-[:HAS_RELATIONSHIP]->(e) RETURN e.title
- USE: MATCH (p)-[r:HAS_RELATIONSHIP]->(e) RETURN p, r, e, e.title

For specific pattern matching:
- MATCH (n) WHERE n.name CONTAINS 'keyword' RETURN n
- MATCH (n1)-[r]->(n2) WHERE n1.property = 'value' RETURN n1, r, n2

For your exploration plan, each step should have a clear purpose that builds on previous steps.
"""

        data = {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": f"Create an exploration plan for: {question}"
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "exploration_plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {
                                            "type": "string",
                                            "description": "Description of what we're trying to find"
                                        },
                                        "query": {
                                            "type": "string",
                                            "description": "A complete Cypher query that returns full node objects and relationships for proper visualization"
                                        },
                                        "purpose": {
                                            "type": "string",
                                            "description": "What to do with the results"
                                        }
                                    },
                                    "required": ["description", "query", "purpose"]
                                }
                            }
                        },
                        "required": ["exploration_plan"]
                    }
                }
            },
            "temperature": 0.3  # Lower temperature for more focused results
        }

        try:
            logger.info("Sending exploration plan request to Llama API")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLAMA_API_KEY}",
            }
            
            response = requests.post(LLAMA_API_URL, headers=headers, json=data)
            
            if response.status_code != 200:
                logger.error(f"Error from Llama API: Status {response.status_code}, Response: {response.text}")
                return self._create_fallback_exploration_plan(question)
                
            response_data = response.json()
            content = self._extract_content_from_llama_response(response_data)

            if content:
                try:
                    parsed_content = json.loads(content)
                    if "exploration_plan" in parsed_content:
                        logger.info(f"Successfully generated exploration plan with {len(parsed_content['exploration_plan'])} steps")
                        return parsed_content["exploration_plan"]
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}, content: {content}")

            # Fallback
            logger.warning("Using fallback exploration plan")
            return self._create_fallback_exploration_plan(question)
        except Exception as e:
            logger.error(f"Error generating exploration plan: {e}", exc_info=True)
            return self._create_fallback_exploration_plan(question)
    
    def _create_fallback_exploration_plan(self, question: str) -> List[Dict[str, Any]]:
        """Create a smart fallback exploration plan when LLM plan generation fails"""
        logger.info(f"Creating fallback exploration plan for: {question}")
        
        # Extract key terms from the question
        terms = re.findall(r'\b\w{3,}\b', question.lower())
        keywords = []
        
        # Filter out common stop words
        stop_words = ["the", "and", "what", "where", "when", "who", "how", "about", "for", "from", "with", "this", "that"]
        for term in terms:
            if term not in stop_words:
                keywords.append(term)
        
        logger.info(f"Extracted keywords: {keywords}")
        
        # Identify potential entity names (capitalized phrases)
        entity_names = re.findall(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+\b|\b[A-Z][a-z]+\b', question)
        logger.info(f"Extracted entity names: {entity_names}")
        
        # Build exploration steps based on database schema
        exploration_plan = []
        
        # Step 1: Check for podcasts matching keywords or entity names
        podcast_conditions = []
        
        # Add entity name conditions
        for entity in entity_names:
            podcast_conditions.append(f"p.podcast_title CONTAINS '{entity}'")
        
        # Add keyword conditions (if not covered by entity names)
        for keyword in keywords:
            if not any(keyword.lower() in entity.lower() for entity in entity_names):
                podcast_conditions.append(f"toLower(p.podcast_title) CONTAINS toLower('{keyword}')")
        
        podcast_where_clause = " OR ".join(podcast_conditions) if podcast_conditions else "true"
        
        exploration_plan.append({
            "description": "Find podcasts matching the keywords or names in the question",
            "query": f"""
            MATCH (p:Podcast)
            WHERE {podcast_where_clause}
            RETURN p
            LIMIT 10
            """,
            "purpose": "Identify relevant podcasts to explore further"
        })
        
        # Step 2: Explore episodes for any matching podcasts
        exploration_plan.append({
            "description": "Explore episodes of relevant podcasts",
            "query": f"""
            MATCH (p:Podcast)-[r:HAS_EPISODE]->(e:Episode)
            WHERE {podcast_where_clause}
            RETURN p, r, e
            LIMIT 20
            """,
            "purpose": "Get the episode structure and content for relevant podcasts"
        })
        
        # Step 3: Look for specific episodes matching keywords
        episode_conditions = []
        
        # Add entity name conditions
        for entity in entity_names:
            episode_conditions.append(f"e.episode_title CONTAINS '{entity}'")
        
        # Add keyword conditions
        for keyword in keywords:
            episode_conditions.append(f"toLower(e.episode_title) CONTAINS toLower('{keyword}')")
        
        episode_where_clause = " OR ".join(episode_conditions) if episode_conditions else "true"
        
        exploration_plan.append({
            "description": "Find specific episodes matching keywords",
            "query": f"""
            MATCH (p:Podcast)-[r:HAS_EPISODE]->(e:Episode)
            WHERE {episode_where_clause}
            RETURN p, r, e
            LIMIT 15
            """,
            "purpose": "Find episodes directly related to the question topics"
        })
        
        # Step 4: Explore mentions/entities in relevant episodes
        if "Topic" in self.db_metadata["labels"] or "Entity" in self.db_metadata["labels"] or "Concept" in self.db_metadata["labels"]:
            entity_label = "Topic" if "Topic" in self.db_metadata["labels"] else "Entity" if "Entity" in self.db_metadata["labels"] else "Concept"
            
            # Find relationship type from Episodes to Entities/Topics
            rel_types = self.db_metadata.get("relationship_types", [])
            entity_rel = None
            
            for rel in rel_types:
                if "MENTIONS" in rel or "HAS_TOPIC" in rel or "ABOUT" in rel:
                    entity_rel = rel
                    break
            
            if entity_rel:
                exploration_plan.append({
                    "description": "Find topics/entities discussed in relevant episodes",
                    "query": f"""
                    MATCH (p:Podcast)-[r1:HAS_EPISODE]->(e:Episode)-[r2:{entity_rel}]->(t:{entity_label})
                    WHERE {episode_where_clause} OR {podcast_where_clause}
                    RETURN p, r1, e, r2, t
                    LIMIT 15
                    """,
                    "purpose": "Discover the topics and entities discussed in relevant episodes"
                })
        
        logger.info(f"Created fallback exploration plan with {len(exploration_plan)} steps")
        return exploration_plan
    
    def _execute_exploration_plan(self, exploration_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute each step in the exploration plan and gather results"""
        logger.info(f"Executing exploration plan with {len(exploration_plan)} steps")
        exploration_results = []
        
        for i, step in enumerate(exploration_plan):
            description = step.get("description", "Exploration step")
            query = step.get("query", "")
            purpose = step.get("purpose", "")
            
            logger.info(f"Executing exploration step {i+1}: {description}")
            
            if not query:
                logger.warning(f"Skipping step {i+1} with empty query")
                continue
                
            # Execute the query
            results, error = self.db.execute_query(query)
            
            # Log results
            if error:
                logger.warning(f"Error in exploration step {i+1}: {error}")
            else:
                result_count = len(results)
                logger.info(f"Step {i+1} returned {result_count} results")
                
                # Add some logging about the types of nodes found
                if result_count > 0:
                    # Extract node types to summarize what was found
                    node_types = {}
                    
                    for record in results:
                        for key, value in record.items():
                            if isinstance(value, dict) and "_type" in value:
                                node_type = value["_type"]
                                if node_type in node_types:
                                    node_types[node_type] += 1
                                else:
                                    node_types[node_type] = 1
                    
                    if node_types:
                        logger.info(f"Node types found in step {i+1}: {node_types}")
            
            # Process the step results
            graph_data = self._process_results_for_visualization(results)
            
            step_result = {
                "description": description,
                "query": query,
                "purpose": purpose,
                "results": results,
                "error": error,
                "graph_data": graph_data
            }
            
            exploration_results.append(step_result)
            
            # If this step had an error, add recovery step
            if error:
                logger.info(f"Generating recovery step for failed exploration step {i+1}")
                recovery_step = self._generate_recovery_step(step, error)
                
                if recovery_step:
                    recovery_query = recovery_step.get("query", "")
                    if recovery_query:
                        logger.info(f"Executing recovery step: {recovery_step.get('description', 'Recovery')}")
                        recovery_results, recovery_error = self.db.execute_query(recovery_query)
                        
                        # Process recovery results
                        recovery_graph_data = self._process_results_for_visualization(recovery_results)
                        
                        recovery_result = {
                            "description": recovery_step.get("description", "Recovery step"),
                            "query": recovery_query,
                            "purpose": recovery_step.get("purpose", "Recover from error"),
                            "results": recovery_results,
                            "error": recovery_error,
                            "graph_data": recovery_graph_data
                        }
                        
                        exploration_results.append(recovery_result)
                        
                        # Log recovery results
                        if recovery_error:
                            logger.warning(f"Recovery step also failed with error: {recovery_error}")
                        else:
                            logger.info(f"Recovery step returned {len(recovery_results)} results")
            
            # Adaptive exploration: if we find no results, try a broader search
            if not error and len(results) == 0 and i < len(exploration_plan) - 1:
                logger.info(f"Step {i+1} found no results, generating adaptive broader search")
                
                # Generate an adaptive broader search step
                broader_step = self._generate_broader_search_step(step)
                
                if broader_step:
                    broader_query = broader_step.get("query", "")
                    if broader_query:
                        logger.info(f"Executing broader search: {broader_step.get('description', 'Broader search')}")
                        broader_results, broader_error = self.db.execute_query(broader_query)
                        
                        # Process broader search results
                        broader_graph_data = self._process_results_for_visualization(broader_results)
                        
                        broader_result = {
                            "description": broader_step.get("description", "Broader search"),
                            "query": broader_query,
                            "purpose": broader_step.get("purpose", "Find results with broader criteria"),
                            "results": broader_results,
                            "error": broader_error,
                            "graph_data": broader_graph_data
                        }
                        
                        exploration_results.append(broader_result)
                        
                        # Log broader search results
                        if broader_error:
                            logger.warning(f"Broader search failed with error: {broader_error}")
                        else:
                            logger.info(f"Broader search returned {len(broader_results)} results")
        
        logger.info(f"Exploration plan execution complete with {len(exploration_results)} steps (including recovery and adaptive steps)")
        return exploration_results
    
    def _generate_recovery_step(self, failed_step: Dict[str, Any], error: str) -> Optional[Dict[str, Any]]:
        """Generate a recovery step when an exploration step fails"""
        logger.info(f"Generating recovery step for failed step: {failed_step.get('description')}")
        
        system_content = f"""You are a Neo4j Cypher query debugging expert. An exploration query has failed with an error.
Fix the query to make it work with the database schema.

Original query description: {failed_step.get('description', 'Exploration step')}
Original query: {failed_step.get('query', '')}
Error: {error}

Available node labels: {', '.join(self.db_metadata['labels'])}
Available relationship types: {', '.join(self.db_metadata['relationship_types'])}

Create a fixed version of this query that avoids the error and achieves a similar purpose.
If the error indicates a missing label or relationship type, try using a different approach or a more generic query.

CRITICAL REQUIREMENT:
Always return complete node objects and relationships in your query, not just properties.
This ensures proper visualization and more complete data for analysis.

Fix the query and create a simpler alternative that will work with the database."""

        data = {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": f"Fix this query: {failed_step.get('query', '')}"
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Description of the fixed query"
                            },
                            "query": {
                                "type": "string",
                                "description": "The fixed Cypher query"
                            },
                            "purpose": {
                                "type": "string",
                                "description": "Purpose of the query"
                            }
                        },
                        "required": ["description", "query", "purpose"]
                    }
                }
            },
            "temperature": 0.3
        }

        try:
            logger.info("Sending recovery step request to Llama API")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLAMA_API_KEY}",
            }
            
            response = requests.post(LLAMA_API_URL, headers=headers, json=data)
            
            if response.status_code != 200:
                logger.error(f"Error from Llama API: Status {response.status_code}, Response: {response.text}")
                return self._create_fallback_recovery_step(failed_step)
                
            response_data = response.json()
            content = self._extract_content_from_llama_response(response_data)

            if content:
                try:
                    parsed_content = json.loads(content)
                    if all(k in parsed_content for k in ["description", "query", "purpose"]):
                        logger.info(f"Successfully generated recovery step: {parsed_content['description']}")
                        return parsed_content
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}, content: {content}")
                    # Try to extract query using regex as fallback
                    match = re.search(r"```cypher\s*(.*?)\s*```", content, re.DOTALL)
                    if match:
                        cypher_query = match.group(1).strip()
                        return {
                            "description": "Fixed query (extracted from code block)",
                            "query": cypher_query,
                            "purpose": "Alternative approach to handle the error"
                        }

            # Fallback
            logger.warning("Using fallback recovery step")
            return self._create_fallback_recovery_step(failed_step)
        except Exception as e:
            logger.error(f"Error generating recovery step: {e}", exc_info=True)
            return self._create_fallback_recovery_step(failed_step)
    
    def _create_fallback_recovery_step(self, failed_step: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback recovery step based on the failed step"""
        original_query = failed_step.get("query", "")
        original_description = failed_step.get("description", "")
        
        # Basic pattern matching to fix common issues
        fixed_query = original_query
        
        # Try to fix common errors
        
        # 1. Undefined variable in RETURN clause
        return_pattern = re.search(r"RETURN\s+([^,]+)(?:,|$)", original_query)
        if return_pattern:
            return_var = return_pattern.group(1).strip()
            if not re.search(fr"\b{return_var}\b", original_query[:return_pattern.start()]):
                # Variable not defined before RETURN, make a simple query
                fixed_query = f"MATCH (n) RETURN n LIMIT 10"
        
        # 2. Invalid label or relationship type
        # Simplify to basic node query if it looks like label/relationship issue
        if "Invalid input" in str(failed_step.get("error", "")) or "Unknown function" in str(failed_step.get("error", "")):
            fixed_query = f"MATCH (n) WHERE n:Podcast OR n:Episode RETURN n LIMIT 10"
        
        # 3. Syntax error, revert to very basic query
        if "SyntaxError" in str(failed_step.get("error", "")):
            fixed_query = "MATCH (n) RETURN n LIMIT 10"
        
        return {
            "description": f"Fallback recovery for: {original_description}",
            "query": fixed_query,
            "purpose": "Simplified query to get basic results after error"
        }
    
    def _generate_broader_search_step(self, previous_step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a broader search step when a specific search returns no results"""
        original_query = previous_step.get("query", "")
        original_description = previous_step.get("description", "")
        
        # Extract patterns from the original query
        match_patterns = re.findall(r"MATCH\s+([^WHERE]+)", original_query)
        where_clauses = re.findall(r"WHERE\s+([^RETURN]+)", original_query)
        
        # If we can't parse the query, create a generic broader search
        if not match_patterns:
            # Create a basic query based on likely entities in the database
            if "Podcast" in self.db_metadata["labels"]:
                return {
                    "description": "Broader search: List all podcasts",
                    "query": "MATCH (p:Podcast) RETURN p LIMIT 20",
                    "purpose": "Get an overview of available podcasts"
                }
            elif "Episode" in self.db_metadata["labels"]:
                return {
                    "description": "Broader search: List episodes",
                    "query": "MATCH (e:Episode) RETURN e LIMIT 20",
                    "purpose": "Get an overview of available episodes"
                }
            else:
                return {
                    "description": "Broader search: Overview of database entities",
                    "query": "MATCH (n) RETURN n LIMIT 15",
                    "purpose": "Get a general overview of database contents"
                }
        
        # Create a broader search by relaxing WHERE conditions
        if where_clauses:
            # Get the original WHERE clause
            where_clause = where_clauses[0]
            
            # Look for exact matches to make more fuzzy
            exact_matches = re.findall(r"(\w+\s*=\s*['\"]\w+['\"])", where_clause)
            contains_matches = re.findall(r"(\w+\s+CONTAINS\s+['\"].+?['\"])", where_clause)
            
            # Create broader version
            broader_query = original_query
            
            # Replace all = with CONTAINS for text fields
            for exact_match in exact_matches:
                if "'" in exact_match or '"' in exact_match:  # Only for string values
                    property_name = re.findall(r"(\w+)\s*=", exact_match)[0]
                    value = re.findall(r"=\s*['\"](.+?)['\"]", exact_match)[0]
                    broader_match = f"toLower({property_name}) CONTAINS toLower('{value}')"
                    broader_query = broader_query.replace(exact_match, broader_match)
            
            # Make AND conditions into OR for broader matches
            broader_query = broader_query.replace(" AND ", " OR ")
            
            # Increase LIMIT if present
            limit_match = re.search(r"LIMIT\s+(\d+)", broader_query)
            if limit_match:
                original_limit = int(limit_match.group(1))
                new_limit = min(original_limit * 2, 50)  # Double the limit, max 50
                broader_query = broader_query.replace(f"LIMIT {original_limit}", f"LIMIT {new_limit}")
            
            return {
                "description": f"Broader search for: {original_description}",
                "query": broader_query,
                "purpose": "Find related results with relaxed criteria"
            }
        
        # If we can't generate a better broader search, return a basic one
        return {
            "description": "Generic broader search",
            "query": "MATCH (n) WHERE n:Podcast OR n:Episode RETURN n LIMIT 15",
            "purpose": "Get an overview of main entity types"
        }
    
    def _generate_answer(self, question: str, query_results: List[Dict[str, Any]], 
                        exploration_data: Dict[str, Any], error: Optional[str] = None) -> str:
        """Generate a comprehensive answer using all available data"""
        logger.info(f"Generating answer for: {question}")
        
        # Format query results for LLM consumption
        results_summary = self._prepare_results_summary(query_results, max_length=2000)
        
        # Format exploration results
        exploration_summary = self._prepare_exploration_summary(exploration_data)
        
        # Build a structured prompt
        system_content = f"""You are a podcast data analysis assistant with expertise in graph databases. Your task is to
    answer the user's question based on query results and additional exploration data.

    When analyzing the results:
    1. Start with the direct query results if they contain useful information
    2. Fill in gaps using the additional data from graph exploration steps
    3. Synthesize information from multiple sources to give a complete picture
    4. When the main results have NULL values, look at the exploration data to explain why and provide alternative information
    5. Be honest about limitations in the data
    6. Describe the relationships between entities (like podcasts and episodes) when relevant

    THE MOST IMPORTANT POINT: Combine information from both the direct query results AND the exploration data
    to provide the most comprehensive answer possible. Don't just focus on one source."""

        user_content = f"""My question was: "{question}"

    DIRECT QUERY RESULTS:
    {results_summary}

    EXPLORATION RESULTS:
    {exploration_summary}

    Please answer my question by synthesizing information from both the main results and the exploration data.
    Highlight any interesting relationships or connections discovered.
    Be specific about which podcasts, episodes, or other entities were found.
    """

        data = {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "temperature": 0.2
        }

        try:
            logger.info("Sending answer generation request to Llama API")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLAMA_API_KEY}",
            }
            
            response = requests.post(LLAMA_API_URL, headers=headers, json=data)
            
            if response.status_code != 200:
                logger.error(f"Error from Llama API: Status {response.status_code}, Response: {response.text}")
                return "I couldn't generate a complete answer. Please try a different question."
                
            response_data = response.json()
            content = self._extract_content_from_llama_response(response_data)

            if content:
                logger.info(f"Generated initial answer of length: {len(content)}")
                
                # Now apply the Podcast Explorer styling to the content
                formatted_answer = self._apply_podcast_explorer_styling(question, content)
                
                logger.info(f"Generated final formatted answer of length: {len(formatted_answer)}")
                return formatted_answer

            # If we couldn't get a response
            logger.warning("Could not extract content from Llama API response")
            return "I couldn't interpret the results. Please try asking your question differently."
        except Exception as e:
            logger.error(f"Error calling Llama API for answer generation: {e}", exc_info=True)
            return f"I encountered a problem while processing your question: {str(e)}. Please try again later."

    def _apply_podcast_explorer_styling(self, question: str, initial_answer: str) -> str:
        """Apply Podcast Explorer styling to the answer"""
        logger.info("Applying Podcast Explorer styling to answer")
        
        podcast_explorer_system_content = """You are **Podcast Explorer**, an AI concierge that surfaces clear, engaging insights from a podcast knowledge graph.

    STYLE RULES  
    1. Write for curious podcast fans, not engineers.  
    2. Lead with the direct answer; keep it under 120 words before any extras.  
    3. Use everyday language, short sentences, and active voice.  
    4. Never mention "steps," "sub‑queries," or how the graph/search was run.  
    5. Cite evidence naturally:  
       • "In episode #123 (May 2023) the host discussed..."  
       • "A podcast published in Jan 2022 covered this topic extensively."  
    6. If data are missing, say so briefly ("No relevant episodes found in the current dataset").  
    7. After the answer, offer **1‑3 bullets** ("Dig deeper") pointing to episodes / nodes the user can click.  
    8. No apologies, no speculation beyond the data, no internal reasoning.
    9. Use markdown to format the answer nicely

    OUTPUT FORMAT  
    - **Answer**: one‑paragraph summary (≤ 120 words)  
    - **Dig deeper**: 1‑3 bullets, each "•" + clickable title (Episode, Guest, Concept) + 1‑line tease."""

        podcast_explorer_user_content = f"""The user asked: "{question}"

    Here is my draft answer that needs to be reformatted according to Podcast Explorer style:

    {initial_answer}

    Reformat my answer following the Podcast Explorer style guide. Keep the core insights but make it clearer, more direct, and add "Dig deeper" bullet points at the end.
    """

        data = {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8", 
            "messages": [
                {
                    "role": "system",
                    "content": podcast_explorer_system_content
                },
                {
                    "role": "user", 
                    "content": podcast_explorer_user_content
                }
            ],
            "temperature": 0.3
        }

        try:
            logger.info("Sending Podcast Explorer formatting request to Llama API")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLAMA_API_KEY}",
            }
            
            response = requests.post(LLAMA_API_URL, headers=headers, json=data)
            
            if response.status_code != 200:
                logger.error(f"Error from Llama API during formatting: Status {response.status_code}, Response: {response.text}")
                return initial_answer  # Return the unformatted answer as fallback
                
            response_data = response.json()
            formatted_content = self._extract_content_from_llama_response(response_data)

            if formatted_content:
                logger.info(f"Successfully formatted answer with Podcast Explorer styling")
                return formatted_content
            else:
                logger.warning("Failed to extract formatted content, returning original answer")
                return initial_answer
                
        except Exception as e:
            logger.error(f"Error applying Podcast Explorer styling: {e}", exc_info=True)
            return initial_answer  # Return the unformatted answer as fallback

    def _prepare_results_summary(self, results: List[Dict[str, Any]], max_length: int = 2000) -> str:
        """Format query results for LLM consumption with intelligent summarization"""
        if not results:
            return "No direct results found from the main query."
        
        # Get sample of results (all if under 5, otherwise a representative sample)
        if len(results) <= 5:
            sample_results = results
        else:
            # Take first, middle, and last for better representation
            sample_results = [
                results[0],
                results[len(results)//2],
                results[-1]
            ]
            
            # Add note about how many more results there are
            additional_note = f"\n... and {len(results) - 3} more results (showing 3 representative examples)"
        
        # Convert results to string, focusing on the most relevant properties for each type
        summary_parts = []
        
        for i, result in enumerate(sample_results):
            summary_parts.append(f"Result {i+1}:")
            
            # Process each field in the result
            for key, value in result.items():
                # Skip system fields
                if key.startswith("_"):
                    continue
                    
                if isinstance(value, dict):
                    # Identify the type of entity to highlight relevant properties
                    entity_type = value.get("_type", "Unknown")
                    
                    summary_parts.append(f"  {key} ({entity_type}):")
                    
                    # For Podcast nodes, focus on podcast-specific properties
                    if entity_type == "Podcast":
                        relevant_props = ["podcast_title", "name", "title", "description", "publisher"]
                        for prop in relevant_props:
                            if prop in value and value[prop]:
                                summary_parts.append(f"    {prop}: {value[prop]}")
                    
                    # For Episode nodes, focus on episode-specific properties
                    elif entity_type == "Episode":
                        relevant_props = ["episode_title", "title", "release_date", "description"]
                        for prop in relevant_props:
                            if prop in value and value[prop]:
                                summary_parts.append(f"    {prop}: {value[prop]}")
                    
                    # For other node types, include most value-contributing properties
                    else:
                        # Filter out system properties and null values
                        for prop, prop_value in value.items():
                            if not prop.startswith("_") and prop_value is not None:
                                summary_parts.append(f"    {prop}: {prop_value}")
                else:
                    # For non-dict values, include directly
                    summary_parts.append(f"  {key}: {value}")
            
            summary_parts.append("")  # Empty line between results
        
        # Add note about additional results
        if len(results) > 5:
            summary_parts.append(additional_note)
        
        # Join all parts and trim if needed
        summary = "\n".join(summary_parts)
        
        if len(summary) > max_length:
            # Include first and last part of summary if too long
            first_part = summary[:max_length//2]
            last_part = summary[-max_length//2:]
            summary = f"{first_part}\n...[truncated for brevity]...\n{last_part}"
        
        return summary
    
    def _prepare_exploration_summary(self, exploration_data: Dict[str, Any]) -> str:
        """Format exploration results with intelligent summarization"""
        exploration_results = exploration_data.get("exploration_results", [])
        
        if not exploration_results:
            return "No additional exploration results."
        
        summary_parts = []
        
        # Process each exploration step
        for i, step in enumerate(exploration_results):
            # Skip steps with errors or no results
            if step.get("error") or not step.get("results"):
                continue
                
            step_description = step.get("description", f"Exploration step {i+1}")
            step_purpose = step.get("purpose", "")
            
            summary_parts.append(f"Step {i+1}: {step_description}")
            if step_purpose:
                summary_parts.append(f"Purpose: {step_purpose}")
            
            # Get entities and relationships found
            entity_counts = {}
            relationship_counts = {}
            
            # Process results to count entities by type and relationships
            for result in step.get("results", []):
                for key, value in result.items():
                    if isinstance(value, dict) and "_type" in value:
                        entity_type = value.get("_type")
                        if entity_type in entity_counts:
                            entity_counts[entity_type] += 1
                        else:
                            entity_counts[entity_type] = 1
                        
                        # For relationships, track types
                        if entity_type and "type" in value:
                            rel_type = value["type"]
                            if rel_type in relationship_counts:
                                relationship_counts[rel_type] += 1
                            else:
                                relationship_counts[rel_type] = 1
            
            # Summarize by entity type
            if entity_counts:
                summary_parts.append("Entities found:")
                for entity_type, count in entity_counts.items():
                    summary_parts.append(f"  - {count} {entity_type}")
            
            # Sample of specific results
            results = step.get("results", [])
            sample_size = min(3, len(results))
            
            if sample_size > 0:
                summary_parts.append(f"\nSample results ({sample_size} of {len(results)}):")
                
                for j in range(sample_size):
                    result = results[j]
                    summary_parts.append(f"  Result {j+1}:")
                    
                    # Extract key data based on entity type
                    for key, value in result.items():
                        if isinstance(value, dict) and "_type" in value:
                            entity_type = value.get("_type")
                            
                            # Format based on entity type
                            if entity_type == "Podcast":
                                title = value.get("podcast_title", value.get("title", "Untitled podcast"))
                                summary_parts.append(f"    {key}: Podcast \"{title}\"")
                            
                            elif entity_type == "Episode":
                                title = value.get("episode_title", value.get("title", "Untitled episode"))
                                date = value.get("release_date", "Unknown date")
                                summary_parts.append(f"    {key}: Episode \"{title}\" ({date})")
                            
                            else:
                                # Generic entity
                                name = value.get("name", value.get("title", f"{entity_type} entity"))
                                summary_parts.append(f"    {key}: {entity_type} \"{name}\"")
                        
                        elif key != "properties" and key != "graph_data" and not key.startswith("_"):
                            # Include non-entity fields except metadata
                            summary_parts.append(f"    {key}: {value}")
                
                if len(results) > sample_size:
                    summary_parts.append(f"  ... and {len(results) - sample_size} more results")
            
            summary_parts.append("")  # Empty line between steps
        
        # Join and return
        return "\n".join(summary_parts)
    
    def _process_results_for_visualization(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Transform query results into graph visualization format with improved handling"""
        logger.info(f"Processing {len(query_results)} results for visualization")
        nodes = {}
        edges = []
        
        # Track which properties were found to help with node identification
        found_properties = {
            "podcast": False,
            "episode": False,
            "entity": False,
            "relationship": False
        }
        
        # Process each result record
        for record in query_results:
            for key, value in record.items():
                # Skip null values and non-dict values (we're looking for node-like objects)
                if value is None or not isinstance(value, dict):
                    continue
                
                # Extract type information if available
                node_type = value.get("_type", "unknown")
                node_id = self._get_node_id(value)
                
                # Track property types found
                if node_type == "Podcast":
                    found_properties["podcast"] = True
                elif node_type == "Episode":
                    found_properties["episode"] = True
                elif node_type and node_type not in ["Podcast", "Episode"]:
                    found_properties["entity"] = True
                
                # Get appropriate label for the node
                node_label = self._get_node_label(value, node_type)
                
                # Add the node if not already added
                if node_id not in nodes:
                    # Determine color based on node type
                    color = self._get_node_color(node_type)
                    
                    nodes[node_id] = {
                        "id": node_id,
                        "label": node_label,
                        "type": node_type,
                        "properties": value,
                        "color": color
                    }
        
        # Second pass: extract relationships by matching nodes found in the first pass
        for record in query_results:
            for key, value in record.items():
                # Skip non-relationship values
                if value is None or not isinstance(value, dict) or not value.get("_type"):
                    continue
                
                # Check if this is a relationship
                rel_type = value.get("_type")
                
                # Track if we found relationships
                if rel_type and not key.startswith("_"):
                    found_properties["relationship"] = True
                
                # Find source and target nodes in this record
                source_id = None
                target_id = None
                
                # Get other keys in this record to find connected nodes
                other_keys = [k for k in record.keys() if k != key]
                
                if len(other_keys) >= 2:
                    # Find which nodes are connected by this relationship
                    for other_key in other_keys:
                        if other_key not in ["properties", "graph_data"] and record[other_key] and isinstance(record[other_key], dict):
                            node_id = self._get_node_id(record[other_key])
                            if node_id in nodes:
                                if source_id is None:
                                    source_id = node_id
                                elif target_id is None:
                                    target_id = node_id
                                    break
                    
                    # Create the edge
                    if source_id and target_id and source_id != target_id:
                        edge_id = f"{source_id}_{rel_type}_{target_id}"
                        
                        # Avoid adding duplicate edges
                        if not any(edge["id"] == edge_id for edge in edges):
                            edges.append({
                                "id": edge_id,
                                "from": source_id,
                                "to": target_id,
                                "label": rel_type,
                                "properties": value
                            })
        
        # If we didn't identify any real nodes or relationships but have results,
        # try to infer structure from the results
        if len(nodes) == 0 and len(query_results) > 0:
            # Check if we have podcast-related data
            podcast_candidates = []
            episode_candidates = []
            
            for record in query_results:
                # Look for podcast-like fields
                if any(key.endswith('podcast_title') or key == 'podcast_title' for key in record.keys()):
                    found_properties["podcast"] = True
                    
                    for key, value in record.items():
                        if 'podcast_title' in key and value:
                            podcast_candidates.append({
                                'id': f"podcast_{value}",
                                'title': value
                            })
                
                # Look for episode-like fields
                if any(key.endswith('episode_title') or key == 'episode_title' for key in record.keys()):
                    found_properties["episode"] = True
                    
                    for key, value in record.items():
                        if 'episode_title' in key and value:
                            episode_candidates.append({
                                'id': f"episode_{value}",
                                'title': value
                            })
            
            # Add inferred podcast nodes
            for podcast in podcast_candidates:
                if podcast['id'] not in nodes:
                    nodes[podcast['id']] = {
                        "id": podcast['id'],
                        "label": podcast['title'],
                        "type": "Podcast",
                        "properties": {"podcast_title": podcast['title']},
                        "color": self._get_node_color("Podcast"),
                        "inferredNode": True
                    }
            
            # Add inferred episode nodes and connect to podcasts
            for episode in episode_candidates:
                if episode['id'] not in nodes:
                    nodes[episode['id']] = {
                        "id": episode['id'],
                        "label": episode['title'],
                        "type": "Episode",
                        "properties": {"episode_title": episode['title']},
                        "color": self._get_node_color("Episode"),
                        "inferredNode": True
                    }
                    
                    # Connect to a podcast if we have any
                    if podcast_candidates:
                        # Connect to the first podcast for simplicity
                        podcast_id = podcast_candidates[0]['id']
                        
                        if podcast_id in nodes:
                            edge_id = f"{podcast_id}_HAS_EPISODE_{episode['id']}"
                            edges.append({
                                "id": edge_id,
                                "from": podcast_id,
                                "to": episode['id'],
                                "label": "HAS_EPISODE",
                                "inferredEdge": True
                            })
        
        # If we still have no nodes, create a fallback node with the results count
        if len(nodes) == 0:
            results_node_id = "results_overview"
            nodes[results_node_id] = {
                "id": results_node_id,
                "label": f"Results ({len(query_results)} items)",
                "type": "Results",
                "properties": {"count": len(query_results)},
                "color": {"background": "#607D8B", "border": "#455A64"},
                "shape": "box",
                "summary": True
            }
        
        logger.info(f"Visualization processing complete: {len(nodes)} nodes and {len(edges)} edges")
        return {"nodes": list(nodes.values()), "edges": edges}
    
    def _get_node_id(self, node_data: Dict[str, Any]) -> str:
        """Get a consistent node ID from node data"""
        # First try to use Neo4j ID
        if "_id" in node_data:
            return f"node_{node_data['_id']}"
        
        # Try common ID fields
        for id_field in ["id", "uuid", "node_id"]:
            if id_field in node_data and node_data[id_field]:
                return f"{id_field}_{node_data[id_field]}"
        
        # For podcast nodes, use podcast_title
        if node_data.get("_type") == "Podcast" and "podcast_title" in node_data:
            return f"podcast_{node_data['podcast_title']}"
        
        # For episode nodes, use episode_title
        if node_data.get("_type") == "Episode" and "episode_title" in node_data:
            return f"episode_{node_data['episode_title']}"
        
        # For named entities, use name or title
        for name_field in ["name", "title"]:
            if name_field in node_data and node_data[name_field]:
                return f"{node_data.get('_type', 'node')}_{node_data[name_field]}"
        
        # Fallback to a hash of the object
        return f"node_{hash(frozenset(node_data.items()))}"
    
    def _get_node_label(self, node_data: Dict[str, Any], node_type: str) -> str:
        """Get appropriate display label for a node"""
        # Type-specific label selection
        if node_type == "Podcast":
            for field in ["podcast_title", "title", "name"]:
                if field in node_data and node_data[field]:
                    return str(node_data[field])
        
        elif node_type == "Episode":
            # For episodes, use episode_title
            for field in ["episode_title", "title", "name"]:
                if field in node_data and node_data[field]:
                    return str(node_data[field])
            
            # If we have both release_date and number, use that format
            if "release_date" in node_data and "number" in node_data:
                return f"Episode {node_data['number']} ({node_data['release_date']})"
        
        elif node_type in ["Person", "Organization", "Topic", "Entity", "Concept"]:
            for field in ["name", "full_name", "title"]:
                if field in node_data and node_data[field]:
                    return str(node_data[field])
        
        # Generic fallback - try common name fields
        for field in ["name", "title", "podcast_title", "episode_title", "id"]:
            if field in node_data and node_data[field]:
                return str(node_data[field])
        
        # Last resort
        return f"{node_type} node"
    
    def _get_node_color(self, node_type: str) -> Dict[str, str]:
        """Get appropriate color for a node based on its type"""
        colors = {
            "podcast": {"background": "#9c27b0", "border": "#7b1fa2"},  # Purple
            "episode": {"background": "#ff9800", "border": "#f57c00"},  # Orange
            "person": {"background": "#e91e63", "border": "#c2185b"},  # Pink
            "organization": {"background": "#2196f3", "border": "#1976d2"},  # Blue
            "product": {"background": "#4caf50", "border": "#388e3c"},  # Green
            "concept": {"background": "#009688", "border": "#00796b"},  # Teal
            "entity": {"background": "#3f51b5", "border": "#303f9f"},  # Indigo
            "location": {"background": "#ffeb3b", "border": "#fdd835"},  # Yellow
            "unknown": {"background": "#607d8b", "border": "#455a64"}  # Gray
        }
        
        # Convert node_type to lowercase for case-insensitive matching
        node_type_lower = node_type.lower() if node_type else "unknown"
        return colors.get(node_type_lower, colors["unknown"])

    def _process_graph_from_all_data(self, main_results: List[Dict[str, Any]], 
                                    exploration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and combine all data sources into a rich graph visualization"""
        logger.info("Processing complete graph from all data sources")
        
        # Process main query results
        main_graph = self._process_results_for_visualization(main_results)
        
        # Track existing nodes and edges
        nodes_by_id = {node["id"]: node for node in main_graph["nodes"]}
        edges_by_id = {edge["id"]: edge for edge in main_graph["edges"]}
        
        # Process and merge exploration results
        exploration_results = exploration_data.get("exploration_results", [])
        
        # Keep track of how many items we add from each source
        addition_stats = {
            "main_query": {
                "nodes": len(nodes_by_id),
                "edges": len(edges_by_id)
            },
            "exploration": {
                "nodes": 0,
                "edges": 0
            }
        }
        
        # Merge nodes and edges from exploration steps
        for step in exploration_results:
            if step.get("error") or not step.get("graph_data"):
                continue
                
            step_graph = step.get("graph_data", {})
            step_desc = step.get("description", "Exploration")
            
            # Add unique nodes from this step
            for node in step_graph.get("nodes", []):
                if node["id"] not in nodes_by_id:
                    # Mark as coming from exploration
                    node["source"] = "exploration"
                    node["exploration_step"] = step_desc
                    
                    # Add to combined graph
                    nodes_by_id[node["id"]] = node
                    addition_stats["exploration"]["nodes"] += 1
            
            # Add unique edges from this step
            for edge in step_graph.get("edges", []):
                if edge["id"] not in edges_by_id:
                    # Only add if both source and target nodes exist
                    if edge["from"] in nodes_by_id and edge["to"] in nodes_by_id:
                        # Mark as coming from exploration
                        edge["source"] = "exploration"
                        edge["exploration_step"] = step_desc
                        
                        # Add to combined graph
                        edges_by_id[edge["id"]] = edge
                        addition_stats["exploration"]["edges"] += 1

# Create final graph
        final_graph = {
            "nodes": list(nodes_by_id.values()),
            "edges": list(edges_by_id.values())
        }
        
        logger.info(f"Final graph contains {len(final_graph['nodes'])} nodes and {len(final_graph['edges'])} edges")
        logger.info(f"Added {addition_stats['exploration']['nodes']} nodes and {addition_stats['exploration']['edges']} edges from exploration")
        
        # Enrich the graph with additional relationship inference if needed
        final_graph = self._enrich_graph_structure(final_graph)
        
        return final_graph
    
    def _enrich_graph_structure(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add inferred relationships and structural elements to enhance visualization"""
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        
        # Track node types present in the graph
        node_types = {}
        for node in nodes:
            node_type = node.get("type", "unknown")
            if node_type in node_types:
                node_types[node_type].append(node)
            else:
                node_types[node_type] = [node]
        
        # Special handling for Podcast/Episode relationships
        if "Podcast" in node_types and "Episode" in node_types:
            # Check if we already have podcast->episode connections
            has_podcast_episode_connections = False
            for edge in edges:
                if edge.get("label") == "HAS_EPISODE":
                    has_podcast_episode_connections = True
                    break
            
            # If no connections, try to infer them
            if not has_podcast_episode_connections:
                podcasts = node_types["Podcast"]
                episodes = node_types["Episode"]
                
                # If only one podcast, connect all episodes to it
                if len(podcasts) == 1:
                    podcast = podcasts[0]
                    for episode in episodes:
                        edge_id = f"{podcast['id']}_HAS_EPISODE_{episode['id']}"
                        
                        # Add the edge if not already present
                        if not any(e["id"] == edge_id for e in edges):
                            edges.append({
                                "id": edge_id,
                                "from": podcast["id"],
                                "to": episode["id"],
                                "label": "HAS_EPISODE",
                                "inferred": True,
                                "dashes": True  # Visual indication this is inferred
                            })
                            logger.info(f"Added inferred HAS_EPISODE relationship from {podcast['label']} to {episode['label']}")
        
        # Connect orphaned nodes to a central node if needed
        orphaned_nodes = []
        connected_node_ids = set()
        
        # Find all connected node IDs
        for edge in edges:
            connected_node_ids.add(edge["from"])
            connected_node_ids.add(edge["to"])
        
        # Find orphaned nodes
        for node in nodes:
            if node["id"] not in connected_node_ids:
                orphaned_nodes.append(node)
        
        # If we have orphaned nodes, connect them to the most relevant central node
        if orphaned_nodes and len(nodes) > len(orphaned_nodes):
            # Find best central node (prefer Podcast or concept node)
            central_node = None
            
            # Priority: Podcast > topic > any connected node
            if "Podcast" in node_types and node_types["Podcast"]:
                central_node = node_types["Podcast"][0]
            elif "Topic" in node_types and node_types["Topic"]:
                central_node = node_types["Topic"][0]
            else:
                # Find the node with most connections
                node_connections = {}
                for edge in edges:
                    for node_id in [edge["from"], edge["to"]]:
                        if node_id in node_connections:
                            node_connections[node_id] += 1
                        else:
                            node_connections[node_id] = 1
                
                # Find node with most connections
                most_connections = 0
                for node_id, count in node_connections.items():
                    if count > most_connections:
                        most_connections = count
                        for node in nodes:
                            if node["id"] == node_id:
                                central_node = node
                                break
            
            # Connect orphaned nodes to central node
            if central_node:
                for orphan in orphaned_nodes:
                    edge_id = f"{central_node['id']}_RELATED_TO_{orphan['id']}"
                    
                    # Add the edge if not already present
                    if not any(e["id"] == edge_id for e in edges):
                        edges.append({
                            "id": edge_id,
                            "from": central_node["id"],
                            "to": orphan["id"],
                            "label": "RELATED_TO",
                            "inferred": True,
                            "dashes": True,  # Visual indication this is inferred
                            "color": {"color": "#aaaaaa", "opacity": 0.5}  # Light gray, semi-transparent
                        })
                        logger.info(f"Added connection from central node to orphaned node {orphan['label']}")
        
        # Return the enriched graph
        return {
            "nodes": nodes,
            "edges": edges
        }


@app.route("/")
def index():
    logger.info("Rendering index page")
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")
    logger.info(f"Received question: {question}")

    if not question:
        logger.warning("Empty question received")
        return jsonify({"error": "Question is required"}), 400

    # Process the query using the QueryProcessor
    try:
        # Initialize query processor
        query_processor = QueryProcessor(db, schema_info, db_metadata)
        
        # Process the query
        result = query_processor.process_query(question)
        
        logger.info(f"Processed question successfully, returning response with answer length: {len(result.get('answer', ''))}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        return jsonify({
            "error": f"An error occurred: {str(e)}",
            "question": question,
            "answer": "I encountered an error while processing your question. Please try a different question or try again later."
        })


@app.route("/schema")
def get_db_schema():
    logger.info("Returning database schema")
    return jsonify({"schema": schema_info})


@app.route("/expand_entity/<entity_name>")
def expand_entity_route(entity_name):
    """Expand a specific entity and its relationships"""
    logger.info(f"Received request to expand entity: {entity_name}")
    
    # Get optional type parameter
    node_type = request.args.get('type', None)
    
    # Sanitize input to prevent Cypher injection
    if not re.match(r'^[a-zA-Z0-9_ -]+$', entity_name):
        logger.warning(f"Invalid entity name received: {entity_name}")
        return jsonify({"error": "Invalid entity name"}), 400
        
    graph_data = db.expand_entity(entity_name, node_type)
    logger.info(f"Entity expansion complete. Returning graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    return jsonify(graph_data)


@app.route("/refresh_schema")
def refresh_schema():
    """Refresh the database schema"""
    logger.info("Refreshing database schema")
    global schema_info, db_metadata
    
    try:
        # Use a synchronized approach to update schema
        new_schema_info = db.get_schema()
        new_db_metadata = db.get_metadata()
        
        # Update global variables atomically
        schema_info = new_schema_info
        db_metadata = new_db_metadata
        
        logger.info("Database schema refreshed successfully")
        return jsonify({"status": "success", "schema": schema_info})
    except Exception as e:
        logger.error(f"Error refreshing schema: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# Initialize database connection and schema at startup
with app.app_context():
    # Initialize database connection and schema information at startup
    logger.info("Initializing application...")
    db = Neo4jDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    schema_info = db.get_schema()
    db_metadata = db.get_metadata()
    logger.info("Application initialization complete")


if __name__ == "__main__":
    try:
        logger.info("Starting Flask application")
        # Use reloader=False to avoid duplicate driver initializations
        app.run(debug=True, port=8080, use_reloader=False)
    finally:
        if 'db' in globals():
            logger.info("Shutting down application, closing database connection")
            db.close()
