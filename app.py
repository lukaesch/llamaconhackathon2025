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
        """Get database metadata (labels, relationship types, property keys)"""
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

            logger.info(f"Retrieved metadata: {len(labels)} labels, {len(relationships)} relationship types, {len(properties)} property keys")
            return {
                "labels": labels,
                "relationship_types": relationships,
                "property_keys": properties,
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
            return dict(value.items())
        elif hasattr(value, "__iter__") and not isinstance(value, str):
            # Convert iterables
            return [self._convert_neo4j_types(v) for v in value]
        else:
            # Return value as is
            return value

    def expand_entity(self, entity_name: str) -> Dict:
        """Get entities related to the specified entity and their relationships"""
        logger.info(f"Expanding entity: {entity_name}")
        # Query to get all direct relationships
        cypher_query = """
        MATCH (center)-[r]-(related)
        WHERE center.name = $entity_name OR center.id = $entity_name OR ID(center) = $entity_id
        RETURN center, r, related
        LIMIT 20
        """

        with self.driver.session() as session:
            try:
                # Try to parse entity_name as integer for ID lookup
                entity_id = None
                try:
                    entity_id = int(entity_name)
                except (ValueError, TypeError):
                    pass

                params = {"entity_name": entity_name}
                if entity_id is not None:
                    params["entity_id"] = entity_id

                result = session.run(cypher_query, **params)

                # Process results to create graph data
                nodes = {}
                edges = []

                # Process the results
                for record in result:
                    center = record["center"]
                    related = record["related"]
                    rel = record["r"]
                    
                    # Get center node ID
                    center_id = center.get("name", center.get("id", str(center.id)))
                    
                    # Get center node type from labels
                    center_type = list(center.labels)[0] if center.labels else "unknown"
                    
                    if center_id not in nodes:
                        nodes[center_id] = {
                            "id": center_id,
                            "label": center.get("name", center.get("title", str(center_id))),
                            "type": center_type,
                            "properties": dict(center),
                        }

                    # Get related node ID
                    related_id = related.get("name", related.get("id", str(related.id)))
                    
                    # Get related node type from labels
                    related_type = list(related.labels)[0] if related.labels else "unknown"
                    
                    if related_id not in nodes:
                        nodes[related_id] = {
                            "id": related_id,
                            "label": related.get("name", related.get("title", str(related_id))),
                            "type": related_type,
                            "properties": dict(related),
                        }

                    # Extract relationship type correctly
                    rel_type = rel.type
                    
                    # Create edge
                    edge_id = f"{center_id}_{rel_type}_{related_id}"
                    edges.append({
                        "id": edge_id,
                        "from": center_id,
                        "to": related_id,
                        "label": rel_type,
                        "properties": dict(rel),
                    })
                
                logger.info(f"Entity expansion complete: found {len(nodes)} nodes and {len(edges)} edges")
                return {"nodes": list(nodes.values()), "edges": edges}

            except Exception as e:
                logger.error(f"Error expanding entity: {e}")
                return {"nodes": [], "edges": []}


class QueryProcessor:
    """Processes complex queries by breaking them down and aggregating results"""
    
    def __init__(self, db: Neo4jDatabase, schema_info: str, db_metadata: dict):
        self.db = db
        self.schema_info = schema_info
        self.db_metadata = db_metadata
        logger.info("QueryProcessor initialized")
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process a complex query by:
        1. Analyzing the query complexity
        2. Breaking it down if needed
        3. Executing sub-queries
        4. Aggregating results
        5. Generating a comprehensive answer
        """
        logger.info(f"Processing query: {question}")
        # First, analyze the query to determine if it's complex
        query_analysis = self._analyze_query_complexity(question)
        
        if query_analysis["is_complex"]:
            logger.info(f"Complex query detected. Complexity score: {query_analysis['complexity_score']}")
            return self._handle_complex_query(question, query_analysis)
        else:
            logger.info("Simple query detected, using standard processing")
            return self._handle_simple_query(question)
    
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
    
    def _analyze_query_complexity(self, question: str) -> Dict[str, Any]:
        """Analyze the complexity of a query using LLM with structured output"""
        logger.info(f"Analyzing complexity of query: {question}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}",
        }

        data = {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are a query complexity analyzer for a Neo4j graph database of podcast data. 
Your task is to analyze the complexity of user questions and determine if they require:
1. Multiple relationship traversals
2. Aggregations (count, sum, avg, etc.)
3. Ordering or sorting
4. Complex filtering conditions
5. Pattern matching across multiple entity types

Rate the complexity on a scale of 1-10 and analyze the following question: "{question}" """
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_complex": {
                                "type": "boolean"
                            },
                            "complexity_score": {
                                "type": "integer"
                            },
                            "requires_multiple_traversals": {
                                "type": "boolean"
                            },
                            "requires_aggregation": {
                                "type": "boolean"
                            },
                            "requires_ordering": {
                                "type": "boolean"
                            },
                            "requires_complex_filtering": {
                                "type": "boolean"
                            },
                            "sub_queries": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["is_complex", "complexity_score"]
                    }
                }
            }
        }

        try:
            logger.info("Sending query complexity analysis request to Llama API")
            response = requests.post(LLAMA_API_URL, headers=headers, json=data)
            
            if response.status_code != 200:
                logger.error(f"Error from Llama API: Status {response.status_code}, Response: {response.text}")
                # Fall back to a simple analysis if API call fails
                return {
                    "is_complex": False,
                    "complexity_score": 3,
                    "requires_multiple_traversals": False,
                    "requires_aggregation": False,
                    "requires_ordering": False,
                    "requires_complex_filtering": False,
                    "sub_queries": []
                }
            
            response_data = response.json()
            logger.debug(f"Response structure: {list(response_data.keys())}")
            
            content = self._extract_content_from_llama_response(response_data)
            
            if content:
                try:
                    logger.debug(f"Parsing content: {content}")
                    parsed_content = json.loads(content)
                    logger.info(f"Successfully parsed query complexity analysis: {parsed_content}")
                    return parsed_content
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}, content: {content}")
            else:
                logger.warning(f"Could not extract content from response: {response_data}")

            # Fallback to a simple analysis
            return {
                "is_complex": False,
                "complexity_score": 3,
                "requires_multiple_traversals": False,
                "requires_aggregation": False,
                "requires_ordering": False,
                "requires_complex_filtering": False,
                "sub_queries": []
            }
        except Exception as e:
            logger.error(f"Error analyzing query complexity: {e}", exc_info=True)
            return {
                "is_complex": False,
                "complexity_score": 3,
                "requires_multiple_traversals": False,
                "requires_aggregation": False,
                "requires_ordering": False,
                "requires_complex_filtering": False,
                "sub_queries": []
            }
    
    def _handle_complex_query(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a complex query through decomposition and aggregation"""
        logger.info(f"Handling complex query: {question}")
        # Step 1: Generate sub-queries if not provided in analysis
        sub_queries = analysis.get("sub_queries", [])
        if not sub_queries:
            logger.info("Generating sub-queries")
            sub_queries = self._generate_sub_queries(question, analysis)
        
        # Step 2: Process each sub-query
        sub_results = []
        all_cypher_queries = []
        all_results = []
        errors = []
        combined_graph_data = {"nodes": [], "edges": []}
        node_ids_set = set()
        edge_ids_set = set()
        
        for i, sub_query in enumerate(sub_queries):
            logger.info(f"Processing sub-query {i+1}/{len(sub_queries)}: {sub_query}")
            
            # Generate and execute Cypher for this sub-query
            cypher_query = get_cypher_from_question(sub_query, self.schema_info, self.db_metadata)
            results, error = self.db.execute_query(cypher_query)
            
            # Store the results
            sub_results.append({
                "sub_query": sub_query,
                "cypher_query": cypher_query,
                "results": results,
                "error": error
            })
            
            all_cypher_queries.append(cypher_query)
            all_results.extend(results)
            if error:
                errors.append(error)
                logger.warning(f"Error in sub-query {i+1}: {error}")
            
            # Add to combined graph data
            graph_data = process_results_for_visualization(results)
            
            # Merge nodes without duplicates
            for node in graph_data["nodes"]:
                if node["id"] not in node_ids_set:
                    combined_graph_data["nodes"].append(node)
                    node_ids_set.add(node["id"])
            
            # Merge edges without duplicates
            for edge in graph_data["edges"]:
                if edge["id"] not in edge_ids_set:
                    combined_graph_data["edges"].append(edge)
                    edge_ids_set.add(edge["id"])
        
        # Step 3: Aggregate the results
        final_cypher = "\n\n--- Sub-queries ---\n" + "\n\n".join(all_cypher_queries)
        
        # Step 4: Generate a comprehensive answer
        logger.info("Generating comprehensive answer")
        answer = self._generate_aggregated_answer(question, sub_results)
        
        logger.info(f"Complex query processing complete. Generated answer length: {len(answer)}")
        return {
            "question": question,
            "cypher_query": final_cypher,
            "results": all_results,
            "answer": answer,
            "graph_data": combined_graph_data,
            "error": "; ".join(errors) if errors else None,
            "sub_queries": sub_queries,
            "is_complex": True
        }
    
    def _handle_simple_query(self, question: str) -> Dict[str, Any]:
        """Handle a simple query through direct execution with adaptive exploration"""
        logger.info(f"Handling simple query: {question}")
        # Generate Cypher query
        cypher_query = get_cypher_from_question(question, self.schema_info, self.db_metadata)
        
        # Execute the query
        results, error = self.db.execute_query(cypher_query)
        
        # Try once more with error feedback if needed
        if error:
            logger.info(f"Initial query failed with error: {error}. Retrying with error feedback.")
            cypher_query = get_cypher_from_question(question, self.schema_info, self.db_metadata, error)
            results, error = self.db.execute_query(cypher_query)
        
        # NEW: Dynamic exploration only if needed (null values, limited results, or errors)
        if error or not results or self._needs_exploration(results):
            logger.info("Results need additional exploration. Performing dynamic exploration.")
            enhanced_data = self._perform_dynamic_exploration(question, results, error)
        else:
            enhanced_data = None
        
        # Generate answer with all available data
        logger.info("Generating answer")
        answer = generate_answer(question, results, error, enhanced_data)
        
        # Process results for visualization
        graph_data = process_results_for_visualization(results)
        
        # Add enhanced exploration data to graph visualization if available
        if enhanced_data and enhanced_data.get("exploration_results"):
            logger.info("Merging exploration results into graph data")
            graph_data = self._merge_graph_data(graph_data, enhanced_data.get("exploration_results", []))
        
        logger.info(f"Simple query processing complete. Generated answer length: {len(answer)}")
        return {
            "question": question,
            "cypher_query": cypher_query,
            "results": results,
            "enhanced_data": enhanced_data,
            "answer": answer,
            "graph_data": graph_data,
            "error": error,
            "is_complex": False
        }
    
    def _needs_exploration(self, results: List[Dict[str, Any]]) -> bool:
        """Determine if results need additional exploration"""
        if not results or len(results) < 3:
            logger.info("Few or no results, exploration needed")
            return True
            
        # Check for null values in key fields
        null_values = 0
        total_fields = 0
        
        for result in results:
            for key, value in result.items():
                total_fields += 1
                if value is None:
                    null_values += 1
        
        # If more than 30% of fields are null, exploration is needed
        null_percentage = (null_values / total_fields) if total_fields > 0 else 0
        logger.info(f"Result quality check: {null_values}/{total_fields} null fields ({null_percentage:.2%})")
        return null_percentage > 0.3
    
    def _perform_dynamic_exploration(self, question: str, initial_results: List[Dict[str, Any]], 
                                    error: Optional[str]) -> Dict[str, Any]:
        """
        Use LLM to dynamically plan and execute an exploration strategy
        based on the question and initial results
        """
        logger.info(f"Performing dynamic exploration for: {question}")
        # Format initial results
        results_str = json.dumps(initial_results[:5], indent=2) if initial_results else "No results"
        if len(results_str) > 1000:
            results_str = results_str[:1000] + "... [truncated]"
        
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

The exploration should be adaptive - if we don't find data one way, try another approach."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}",
        }

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
                                            "description": "A complete Cypher query to execute"
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
            }
        }

        try:
            logger.info("Sending exploration plan request to Llama API")
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
        """Create a simple fallback exploration plan when LLM plan generation fails"""
        logger.info(f"Creating fallback exploration plan for: {question}")
        # Extract potential entities from question (simple approach)
        entity_terms = re.findall(r'\b[A-Z][a-z]+\b', question)
        entity_terms.extend(re.findall(r'"([^"]+)"', question))
        
        # If no entities found, use all words longer than 4 characters
        if not entity_terms:
            entity_terms = [word for word in question.split() if len(word) > 4]
        
        logger.info(f"Extracted terms for exploration: {entity_terms}")
        
        # Create LIKE patterns for each term
        entity_patterns = " OR ".join([f"toLower(e.name) CONTAINS toLower('{term}')" for term in entity_terms])
        if not entity_patterns:
            entity_patterns = "true"  # Fallback if no terms found
        
        return [
            {
                "description": "Find entities related to the question",
                "query": f"MATCH (e) WHERE {entity_patterns} RETURN e LIMIT 10",
                "purpose": "Identify key entities to explore"
            },
            {
                "description": "Explore relationships of found entities",
                "query": f"MATCH (e)-[r]-(related) WHERE {entity_patterns} "
                        f"RETURN e.name as entity, type(r) as relationship, "
                        f"labels(related)[0] as related_type, count(*) as count "
                        f"ORDER BY count DESC LIMIT 20",
                "purpose": "Discover available connections"
            },
            {
                "description": "Find connected entities with their properties",
                "query": f"MATCH (e)-[r]-(related) WHERE {entity_patterns} "
                        f"RETURN e.name as entity, type(r) as relationship, related "
                        f"LIMIT 15",
                "purpose": "Get detailed information about related entities"
            }
        ]
    
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
            
            if error:
                logger.warning(f"Error in exploration step {i+1}: {error}")
            else:
                logger.info(f"Step {i+1} returned {len(results)} results")
            
            # Process the step results
            step_result = {
                "description": description,
                "query": query,
                "purpose": purpose,
                "results": results,
                "error": error,
                "graph_data": process_results_for_visualization(results)
            }
            
            exploration_results.append(step_result)
            
            # If this step had an error, add an additional recovery step
            if error:
                logger.info(f"Generating recovery step for failed exploration step {i+1}")
                recovery_step = self._generate_recovery_step(step, error)
                if recovery_step:
                    recovery_query = recovery_step.get("query", "")
                    if recovery_query:
                        logger.info(f"Executing recovery step: {recovery_step.get('description', 'Recovery')}")
                        recovery_results, recovery_error = self.db.execute_query(recovery_query)
                        recovery_step["results"] = recovery_results
                        recovery_step["error"] = recovery_error
                        recovery_step["graph_data"] = process_results_for_visualization(recovery_results)
                        
                        if recovery_error:
                            logger.warning(f"Recovery step failed with error: {recovery_error}")
                        else:
                            logger.info(f"Recovery step returned {len(recovery_results)} results")
                            
                        exploration_results.append(recovery_step)
        
        logger.info(f"Exploration plan execution complete with {len(exploration_results)} steps (including recovery steps)")
        return exploration_results
    
    def _generate_recovery_step(self, failed_step: Dict[str, Any], error: str) -> Optional[Dict[str, Any]]:
        """Generate a recovery step when an exploration step fails using structured output"""
        logger.info(f"Generating recovery step for failed step: {failed_step.get('description')}")
        
        system_content = f"""You are a Neo4j Cypher query debugging expert. An exploration query has failed with an error.
Fix the query to make it work with the database schema.

Original query description: {failed_step.get('description', 'Exploration step')}
Original query: {failed_step.get('query', '')}
Error: {error}

Available node labels: {', '.join(self.db_metadata['labels'])}
Available relationship types: {', '.join(self.db_metadata['relationship_types'])}

Create a fixed version of this query that avoids the error and achieves a similar purpose."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}",
        }

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
            }
        }

        try:
            logger.info("Sending recovery step request to Llama API")
            response = requests.post(LLAMA_API_URL, headers=headers, json=data)
            
            if response.status_code != 200:
                logger.error(f"Error from Llama API: Status {response.status_code}, Response: {response.text}")
                # Fall back to a simple recovery step
                return {
                    "description": f"Fallback for: {failed_step.get('description', 'Failed step')}",
                    "query": "MATCH (n) RETURN labels(n) as node_types, count(*) as count GROUP BY node_types LIMIT 10",
                    "purpose": "Get database overview after original query failed"
                }
                
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

            # Fallback
            logger.warning("Using fallback recovery step")
            return {
                "description": f"Fallback for: {failed_step.get('description', 'Failed step')}",
                "query": "MATCH (n) RETURN labels(n) as node_types, count(*) as count GROUP BY node_types LIMIT 10",
                "purpose": "Get database overview after original query failed"
            }
        except Exception as e:
            logger.error(f"Error generating recovery step: {e}", exc_info=True)
            return None
    
    def _merge_graph_data(self, primary_graph: Dict[str, Any], 
                         exploration_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge graph visualization data from multiple exploration steps"""
        logger.info("Merging graph data from exploration results")
        # Track existing nodes and edges to avoid duplicates
        node_ids = {node["id"] for node in primary_graph["nodes"]}
        edge_ids = {edge["id"] for edge in primary_graph["edges"]}
        
        # Create combined graph starting with primary graph
        combined_graph = {
            "nodes": primary_graph["nodes"].copy(),
            "edges": primary_graph["edges"].copy()
        }
        
        nodes_added = 0
        edges_added = 0
        
        # Add nodes and edges from each exploration step
        for step in exploration_results:
            graph_data = step.get("graph_data", {"nodes": [], "edges": []})
            
            # Add new nodes
            for node in graph_data.get("nodes", []):
                if node["id"] not in node_ids:
                    # Add a flag showing this came from exploration
                    node["fromExploration"] = True
                    node["explorationDescription"] = step.get("description", "Exploration")
                    combined_graph["nodes"].append(node)
                    node_ids.add(node["id"])
                    nodes_added += 1
            
            # Add new edges
            for edge in graph_data.get("edges", []):
                if edge["id"] not in edge_ids:
                    # Add a flag showing this came from exploration
                    edge["fromExploration"] = True
                    edge["explorationDescription"] = step.get("description", "Exploration")
                    combined_graph["edges"].append(edge)
                    edge_ids.add(edge["id"])
                    edges_added += 1
        
        logger.info(f"Merged graph data: added {nodes_added} nodes and {edges_added} edges from exploration results")
        return combined_graph
    
    def _generate_sub_queries(self, question: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate sub-queries for a complex question using structured output"""
        logger.info(f"Generating sub-queries for: {question}")
        
        system_content = f"""You are a query decomposition expert for a Neo4j graph database of podcast data.
Your task is to break down complex questions into simpler sub-queries that can be executed independently.

Original question: "{question}"

Complexity analysis:
- Requires multiple traversals: {analysis.get('requires_multiple_traversals', False)}
- Requires aggregation: {analysis.get('requires_aggregation', False)}
- Requires ordering: {analysis.get('requires_ordering', False)}
- Requires complex filtering: {analysis.get('requires_complex_filtering', False)}
- Complexity score: {analysis.get('complexity_score', 5)}/10

Break this question down into 2-5 simpler sub-queries that can be executed independently.
Each sub-query should focus on a specific aspect and be executable as a standalone query."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}",
        }

        data = {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": f"Break down this question into sub-queries: {question}"
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sub_queries": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "A list of 2-5 sub-queries that can be executed independently"
                            }
                        },
                        "required": ["sub_queries"]
                    }
                }
            }
        }

        try:
            logger.info("Sending sub-query generation request to Llama API")
            response = requests.post(LLAMA_API_URL, headers=headers, json=data)
            
            if response.status_code != 200:
                logger.error(f"Error from Llama API: Status {response.status_code}, Response: {response.text}")
                # Fall back to default sub-queries
                return [
                    f"What entities are mentioned in: {question}",
                    f"What relationships exist between entities in: {question}",
                    f"What are the key properties of entities in: {question}"
                ]
                
            response_data = response.json()
            content = self._extract_content_from_llama_response(response_data)

            if content:
                try:
                    parsed_content = json.loads(content)
                    if "sub_queries" in parsed_content and isinstance(parsed_content["sub_queries"], list):
                        logger.info(f"Successfully generated {len(parsed_content['sub_queries'])} sub-queries")
                        return parsed_content["sub_queries"]
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}, content: {content}")

            # Fallback to basic sub-queries if parsing fails
            logger.warning("Using fallback sub-queries")
            return [
                f"What entities are mentioned in: {question}",
                f"What relationships exist between entities in: {question}",
                f"What are the key properties of entities in: {question}"
            ]
        except Exception as e:
            logger.error(f"Error generating sub-queries: {e}", exc_info=True)
            return [
                f"What entities are mentioned in: {question}",
                f"What relationships exist between entities in: {question}",
                f"What are the key properties of entities in: {question}"
            ]
    
    def _generate_aggregated_answer(self, question: str, sub_results: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive answer by aggregating sub-query results"""
        logger.info(f"Generating aggregated answer for: {question}")
        # Format sub-results for the prompt
        formatted_results = ""
        for i, result in enumerate(sub_results):
            formatted_results += f"\nSub-Query {i+1}: {result['sub_query']}\n"
            if result['error']:
                formatted_results += f"Error: {result['error']}\n"
            else:
                # Truncate results if too large
                result_str = json.dumps(result['results'], indent=2)
                if len(result_str) > 2000:  # Limit length to avoid token limits
                    result_str = result_str[:2000] + "... [truncated]"
                formatted_results += f"Results: {result_str}\n"
        
        prompt = f"""### SYSTEM
You are a data analysis expert for a Neo4j graph database of podcast data.
Your task is to synthesize results from multiple sub-queries to answer a complex question.

Original complex question: "{question}"

I've broken this down into sub-queries and obtained the following results:
{formatted_results}

Based on all these results, please provide a comprehensive, well-structured answer to the original question.
Your answer should:
1. Address all aspects of the original question
2. Synthesize information from the different sub-queries
3. Present a coherent narrative that is easy to understand
4. Acknowledge any limitations in the data or errors encountered

### USER
Please provide a comprehensive answer to: "{question}"
"""

        return self._call_llm_api(prompt)
    
    def _call_llm_api(self, prompt: str) -> str:
        """Make a call to the LLama API"""
        logger.info("Calling Llama API for text generation")
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

            # Extract content from response
            content = self._extract_content_from_llama_response(response_data)
            if content:
                logger.info(f"Received response of length: {len(content)}")
                return content

            logger.warning("Could not extract content from Llama API response")
            return "I couldn't process this request. Please try again with a simpler question."
        except Exception as e:
            logger.error(f"Error calling Llama API: {e}", exc_info=True)
            return f"Error processing request: {str(e)}"


def generate_fallback_query(db_metadata) -> str:
    """Generate a simple but valid fallback query based on the database metadata"""
    logger.info("Generating fallback query")
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

    fallback_query = f"""
    MATCH (e:{entity_label})-[r]->(m)
    RETURN e, r, m
    LIMIT 10
    """
    
    logger.info(f"Generated fallback query: {fallback_query}")
    return fallback_query


def get_cypher_from_question(
    question: str, schema_info: str, db_metadata: dict, error_message: str = None
) -> str:
    """Generate a Cypher query from a natural language question using Llama with structured output"""
    logger.info(f"Generating Cypher query for: {question}")
    if error_message:
        logger.info(f"Previous error: {error_message}")

    # Create prompt based on whether we're handling an error
    if error_message:
        system_content = f"""You are a Neo4j Cypher query debugging expert. A previous query attempt failed with the following error:

ERROR: {error_message}

Please fix the query to make it syntactically correct and executable in Neo4j. Pay close attention to:
1. Only use node labels that exist in the database: {', '.join(db_metadata['labels'])}
2. Only use relationship types that exist in the database: {', '.join(db_metadata['relationship_types'])}
3. Make sure all variables used in the RETURN clause are defined in the MATCH pattern
4. Ensure proper relationship direction in MATCH patterns
5. Use single quotes for string literals
6. For complex queries, ensure all aggregations are properly defined and grouped

Generate a working Cypher query to answer this question: {question}"""
    else:
        system_content = f"""You are a Neo4j Cypher query generation expert. Your job is to convert natural language questions 
about podcast data into precise Cypher queries.

Here is the database schema information:
{schema_info[:1000]}  # Truncate to avoid token limits

Available node labels: {', '.join(db_metadata['labels'])}
Available relationship types: {', '.join(db_metadata['relationship_types'])}

You must ONLY use these exact labels and relationship types in your query.

Generate a Cypher query that would answer the user's question. Your query should be optimized for:

1. Complex pattern matching: Don't hesitate to use multiple MATCH clauses for complex relationships
2. Aggregation: Use COUNT, SUM, AVG, MIN, MAX when appropriate
3. Sorting: Use ORDER BY when the question asks for "top", "most", "least", etc.
4. Filtering: Use WHERE clauses with complex conditions when needed
5. Return complete paths: Use relationship variables and return full nodes

The question is: {question}"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLAMA_API_KEY}",
    }

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
                            "description": "A valid executable Neo4j Cypher query"
                        }
                    },
                    "required": ["cypher_query"]
                }
            }
        }
    }

    try:
        logger.info("Sending Cypher generation request to Llama API")
        response = requests.post(LLAMA_API_URL, headers=headers, json=data)
        
        if response.status_code != 200:
            logger.error(f"Error from Llama API: Status {response.status_code}, Response: {response.text}")
            return generate_fallback_query(db_metadata)
        
        response_data = response.json()
        logger.debug(f"Response keys: {list(response_data.keys())}")
        
        # Extract content from different response formats
        content = None
        
        # Format 1: Standard OpenAI-like format
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            logger.debug(f"Extracted content from 'choices' format: {content[:100]}...")
        
        # Format 2: Llama-specific format
        elif "completion_message" in response_data and "content" in response_data["completion_message"]:
            # Handle different content types
            content_obj = response_data["completion_message"]["content"]
            
            if isinstance(content_obj, dict):
                if "text" in content_obj:
                    content = content_obj["text"]
                    logger.debug(f"Extracted content from 'completion_message.content.text': {content[:100]}...")
                elif "type" in content_obj and content_obj["type"] == "text":
                    content = content_obj["text"]
                    logger.debug(f"Extracted content from 'completion_message.content' with type 'text': {content[:100]}...")
            elif isinstance(content_obj, str):
                content = content_obj
                logger.debug(f"Extracted string content from 'completion_message.content': {content[:100]}...")

        if content:
            try:
                parsed_content = json.loads(content)
                if "cypher_query" in parsed_content:
                    cypher_query = parsed_content["cypher_query"]
                    logger.info(f"Successfully generated Cypher query: {cypher_query[:100]}...")
                    return cypher_query
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}, content: {content}")

        # If we couldn't get a structured response, fall back to a simple query
        logger.warning("Using fallback query")
        return generate_fallback_query(db_metadata)
    except Exception as e:
        logger.error(f"Error calling Llama API for query generation: {e}", exc_info=True)
        return generate_fallback_query(db_metadata)


def generate_answer(
    question: str, 
    query_results: List[Dict[str, Any]], 
    error: Optional[str] = None,
    enhanced_data: Optional[Dict[str, Any]] = None
) -> str:
    """Generate a natural language answer from query results using Llama"""
    logger.info(f"Generating answer for: {question}")
    # Convert results to a readable format
    results_str = json.dumps(query_results, indent=2)
    
    # Truncate if too large to avoid token limits
    if len(results_str) > 4000:
        results_str = results_str[:4000] + "... [truncated for brevity]"

    # Include error information if applicable
    error_context = ""
    if error:
        error_context = f"\nNote: The query encountered an error: {error}\n"
        logger.info(f"Including error context in answer generation: {error}")
    
    # Format exploration results if available
    exploration_context = ""
    if enhanced_data and enhanced_data.get("exploration_results"):
        logger.info("Including exploration results in answer generation")
        exploration_context = "\n### Additional Data from Graph Exploration:\n"
        
        for i, step in enumerate(enhanced_data["exploration_results"]):
            # Only include successful steps with results
            if not step.get("error") and step.get("results"):
                exploration_context += f"\n{i+1}. {step['description']}:\n"
                
                # Limit the number of results shown
                results = step["results"][:3] if len(step["results"]) > 3 else step["results"]
                results_snippet = json.dumps(results, indent=2)
                
                # Truncate if too large
                if len(results_snippet) > 1000:
                    results_snippet = results_snippet[:1000] + "... [truncated]"
                
                exploration_context += f"{results_snippet}\n"
                
                if len(step["results"]) > 3:
                    exploration_context += f"... and {len(step['results']) - 3} more results\n"

    prompt = f"""### SYSTEM
You are a helpful podcast data analysis assistant with expertise in graph databases. Your task is to
answer the user's question based on query results and additional exploration data.

When analyzing the results, please:
1. Start with the direct query results if they contain useful information
2. Fill in gaps using the additional data from graph exploration
3. Synthesize information from multiple sources to give a complete picture
4. When the main results have NULL values, look at the exploration data to explain why and provide alternative information
5. Be honest about limitations in the data

THE MOST IMPORTANT POINT: Combine information from both the direct query results AND the exploration data
to provide the most comprehensive answer possible. Don't just focus on one source.

### USER
My question was: "{question}"
{error_context}

The main database query returned these results:
{results_str}

{exploration_context}

Please answer my question completely, using both the main results and the exploration data.
If something is missing or NULL in the main results, explain what we learned from the exploration.
Highlight any interesting relationships or connections discovered.
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
        logger.info("Sending answer generation request to Llama API")
        response = requests.post(LLAMA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()

        # Extract content from different response formats
        content = None
        
        # Format 1: Standard OpenAI-like format
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            logger.debug(f"Extracted answer from 'choices' format: {content[:100]}...")
        
        # Format 2: Llama-specific format
        elif "completion_message" in response_data and "content" in response_data["completion_message"]:
            # Handle different content types
            content_obj = response_data["completion_message"]["content"]
            
            if isinstance(content_obj, dict):
                if "text" in content_obj:
                    content = content_obj["text"]
                    logger.debug(f"Extracted answer from 'completion_message.content.text': {content[:100]}...")
                elif "type" in content_obj and content_obj["type"] == "text":
                    content = content_obj["text"]
                    logger.debug(f"Extracted answer from 'completion_message.content' with type 'text': {content[:100]}...")
            elif isinstance(content_obj, str):
                content = content_obj
                logger.debug(f"Extracted string answer from 'completion_message.content': {content[:100]}...")

        if content:
            logger.info(f"Generated answer of length: {len(content)}")
            return content

        logger.warning("Could not extract content from Llama API response")
        return "I couldn't interpret the results. Please try asking your question differently."
    except Exception as e:
        logger.error(f"Error calling Llama API for answer generation: {e}", exc_info=True)
        return "Sorry, I encountered an error while generating your answer. Please try again later."


def process_results_for_visualization(
    query_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Transform query results into graph visualization format with support for complex queries"""
    logger.info(f"Processing {len(query_results)} results for visualization")
    nodes = {}
    edges = []

    # Extract nodes and edges from results
    for record in query_results:
        # Process each key in the record
        for key, value in record.items():
            # Skip if not a dict, None, or a primitive type (handling aggregation results)
            if not isinstance(value, dict) or value is None:
                # Handle aggregation values by attaching them to a virtual node if needed
                if key.startswith('count') or key.startswith('sum') or key.startswith('avg'):
                    continue
                continue

            # Process nodes
            node_id = str(value.get("name", value.get("id", 
                     value.get("episode_id", value.get("podcast_id", key)))))
            
            if not node_id or node_id in nodes:
                continue
                
            # Determine node type
            node_type = value.get("type", "unknown")
            if "episode_id" in value:
                node_type = "episode"
            elif "podcast_id" in value:
                node_type = "podcast"
                
            # Create node label
            node_label = value.get("name", value.get("title", 
                          value.get("episode_title", value.get("podcast_title", str(node_id)))))
                
            # Add aggregation values to the node label if present in the record
            agg_info = []
            for agg_key, agg_value in record.items():
                if (agg_key.startswith('count') or agg_key.startswith('sum') or 
                    agg_key.startswith('avg') or agg_key.startswith('min') or 
                    agg_key.startswith('max')) and not isinstance(agg_value, dict):
                    agg_info.append(f"{agg_key}: {agg_value}")
                    
            if agg_info:
                node_label = f"{node_label}\n({', '.join(agg_info)})"
                
            nodes[node_id] = {
                "id": node_id,
                "label": node_label,
                "type": node_type,
                "properties": value,
            }

    # Extract relationships
    for record in query_results:
        relationship_found = False
        source = None
        target = None
        relationship = None
        
        # First try to find a direct relationship entry
        for key, value in record.items():
            if not isinstance(value, dict) or value is None:
                continue
                
            if key.lower() in ('r', 'rel', 'relationship'):
                relationship = value
                relationship_found = True
                break
        
        # If relationship was found, try to find source and target nodes
        if relationship_found:
            for key, value in record.items():
                if not isinstance(value, dict) or value is None or key.lower() in ('r', 'rel', 'relationship'):
                    continue
                
                node_id = str(value.get("name", value.get("id", 
                         value.get("episode_id", value.get("podcast_id", key)))))
                         
                if node_id in nodes:
                    if source is None:
                        source = node_id
                    elif target is None:
                        target = node_id
                        break
            
            # If we found source, target and relationship, create edge
            if source and target and source != target:
                # Get relationship type directly from the 'type' attribute
                rel_type = relationship.get("type", "RELATED_TO")
                
                edge_id = f"{source}_{rel_type}_{target}"
                
                edges.append({
                    "id": edge_id,
                    "from": source,
                    "to": target,
                    "label": rel_type,
                    "properties": relationship,
                })
        else:
            # Look for pairs of nodes that might be related
            node_keys = []
            for key, value in record.items():
                if isinstance(value, dict) and value is not None:
                    node_id = str(value.get("name", value.get("id", 
                             value.get("episode_id", value.get("podcast_id", key)))))
                    if node_id in nodes:
                        node_keys.append((key, node_id))
            
            # Create inferred edges between consecutive nodes, if we have at least 2
            if len(node_keys) >= 2:
                for i in range(len(node_keys) - 1):
                    source_key, source_id = node_keys[i]
                    target_key, target_id = node_keys[i + 1]
                    
                    # Infer relationship type from keys if possible
                    rel_type = f"RELATED_TO"
                    
                    edge_id = f"{source_id}_{rel_type}_{target_id}"
                    
                    edges.append({
                        "id": edge_id,
                        "from": source_id,
                        "to": target_id,
                        "label": rel_type,
                        "properties": {},
                    })

    # Handle aggregation results with virtual nodes if needed
    if len(nodes) == 0 and len(query_results) > 0:
        # Check if we have pure aggregation results
        has_aggregations = False
        for record in query_results:
            for key in record.keys():
                if (key.startswith('count') or key.startswith('sum') or key.startswith('avg') or 
                    key.startswith('min') or key.startswith('max')):
                    has_aggregations = True
                    break
            if has_aggregations:
                break
        
        if has_aggregations:
            # Create a virtual results node
            result_id = "aggregation_results"
            label = "Aggregation Results\n"
            
            # Add all aggregation values
            for i, record in enumerate(query_results[:5]):  # Limit to first 5 for readability
                label += f"Result {i+1}: "
                for key, value in record.items():
                    if not isinstance(value, dict):
                        label += f"{key}: {value}, "
                label = label.rstrip(", ") + "\n"
            
            if len(query_results) > 5:
                label += f"... and {len(query_results) - 5} more results"
                
            nodes[result_id] = {
                "id": result_id,
                "label": label,
                "type": "results",
                "properties": {"aggregations": True},
            }

    logger.info(f"Visualization processing complete: {len(nodes)} nodes and {len(edges)} edges")
    return {"nodes": list(nodes.values()), "edges": edges}


# Initialize database connection and schema at startup
with app.app_context():
    # Initialize database connection and schema information at startup
    logger.info("Initializing application...")
    db = Neo4jDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    schema_info = db.get_schema()
    db_metadata = db.get_metadata()
    query_processor = QueryProcessor(db, schema_info, db_metadata)
    logger.info("Application initialization complete")


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

    # Process the query using the QueryProcessor (handles both simple and complex queries)
    result = query_processor.process_query(question)
    
    # Add diagnostic info for debugging
    if "enhanced_data" in result:
        result["has_enhanced_data"] = bool(result["enhanced_data"])
    
    logger.info(f"Processed question successfully, returning response with answer length: {len(result.get('answer', ''))}")
    return jsonify(result)


@app.route("/schema")
def get_db_schema():
    logger.info("Returning database schema")
    global schema_info
    return jsonify({"schema": schema_info})


@app.route("/expand_entity/<entity_name>")
def expand_entity_route(entity_name):
    """Expand a specific entity and its relationships"""
    logger.info(f"Received request to expand entity: {entity_name}")
    # Sanitize input to prevent Cypher injection
    if not re.match(r'^[a-zA-Z0-9_ -]+$', entity_name):
        logger.warning(f"Invalid entity name received: {entity_name}")
        return jsonify({"error": "Invalid entity name"}), 400
        
    graph_data = db.expand_entity(entity_name)
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


if __name__ == "__main__":
    try:
        logger.info("Starting Flask application")
        # Use reloader=False to avoid duplicate driver initializations
        app.run(debug=True, port=8080, use_reloader=False)
    finally:
        if db:
            logger.info("Shutting down application, closing database connection")
            db.close()
