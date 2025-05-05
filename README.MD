# Podcast Knowledge Graph Explorer

A GraphRAG system that analyzes podcast conversations using AI to extract entities, relationships, and insights from transcripts.

## Overview

This project combines a knowledge graph with AI to explore and understand podcast content. It uses Llama 4 to extract named entities from podcast transcripts and builds a Neo4j graph database that can be queried through natural language.

## Key Components

### Data Ingestion Pipeline (`extract_entities.py`)

This script processes podcast transcriptions and extracts structured information:

1. Fetches podcast episodes and their transcripts from an SQLite database
2. Uses Llama 4 to identify entities, sentiment, and relationships mentioned in podcasts
3. Stores the extracted data in a Neo4j graph database with the following structure:
   - Nodes: Podcast, Episode, Person, Organization, Product, Concept, etc.
   - Relationships: HAS_EPISODE, MENTIONED_IN, RELATES_TO, etc.

### Query & Visualization Interface (`app.py`)

A Flask application that provides a natural language interface to the knowledge graph:

1. Converts user questions to Cypher queries using Llama 4
2. Implements an adaptive querying system that:
   - Generates initial graph query
   - Analyzes results for completeness
   - Creates exploration plans to find additional relevant information
   - Performs dynamic additional queries when needed
   - Synthesizes results into comprehensive answers
3. Visualizes results as an interactive graph

## Features

- **Natural Language Interface**: Ask questions about podcast content in plain English
- **Knowledge Graph Visualization**: Explore connections between entities mentioned in podcasts
- **Adaptive Query Processing**: Smart decomposition of complex queries into exploration plans
- **Entity Expansion**: Drill down into specific entities to discover relationships
- **Error Recovery**: Automatic retry strategies when initial queries fail

## Technical Architecture

### Data Extraction

The extraction pipeline reads podcast transcripts and uses Llama 4 to identify:
- Named entities (people, organizations, products, etc.)
- Sentiment analysis (-1.0 to 1.0) with explanations
- Relationships between entities

### Graph Database

Neo4j stores the extracted data in a rich graph structure:
- **Nodes**: Podcasts, Episodes, Entities (with type)
- **Relationships**: Connections between entities with relationship types
- **Properties**: Metadata, timestamps, sentiment scores, etc.

### Query Processing

The adaptive querying system:
1. Analyzes user questions and database schema
2. Generates optimal Cypher queries
3. Creates dynamic exploration plans for complex questions
4. Handles results or errors intelligently
5. Synthesizes comprehensive answers

## Getting Started

### Prerequisites

- Python 3.9+
- Neo4j database
- Llama API access ([https://llama.developer.meta.com/](https://llama.developer.meta.com/))
- Audioscrape API key for fetching podcast transcriptions and metadata ([www.audioscrape.com](https://www.audioscrape.com))

### Environment Variables

Create a `.env` file with:

```
LLAMA_API_KEY=your_api_key
LLAMA_API_URL=https://api.llama.com/v1/chat/completions
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

### Installation

```bash
pip install -r requirements.txt
```

### Running the Data Ingestion

```bash
python extract_entities.py
```

### Running the Web Application

```bash
python app.py
```

## Example Queries

- "What topics does Sam Altman discuss most frequently in podcasts?"
- "Which podcasters have talked about climate change and what do they say?"
- "How has the discussion of artificial intelligence evolved in podcasts over time?"
- "What are the most mentioned companies in tech podcasts?"
- "Find connections between cryptocurrency experts who appeared on the same podcasts"

## Key Innovation: Adaptive Query System

Our solution features a sophisticated adaptive query system that:

1. **Query Planning**: Decomposes complex questions into logical exploration steps
2. **Dynamic Exploration**: Adapts the query strategy based on initial results
3. **Error Recovery**: Automatically generates alternative approaches when queries fail
4. **Relationship Discovery**: Infers connections between entities even when not explicitly stored
5. **Context Enhancement**: Progressively builds context to provide comprehensive answers

This approach allows for much more robust knowledge extraction than simple one-shot querying, especially for complex questions where relevant information might be distributed across the graph.

## Conclusion

This project demonstrates the power of combining Large Language Models with graph databases for knowledge discovery in unstructured content. The adaptive querying approach enables deeper insights than traditional RAG systems by exploring relationship paths and inferring connections dynamically.

---

*This project was developed by team GraphLlamas for the #lamaconhackathon2025*
