"""
Answer Generator - Generate grounded answers from retrieved context with decomposition-aware prompting.

Uses LLM to generate answers based on graph and vector context.
Leverages query decomposition for more targeted and accurate responses.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class AnswerGenerator:
    """
    Generates answers using LLM with retrieved context.
    
    Optimizations:
    - Decomposition-aware prompting
    - Entity-focused context prioritization
    - Multi-intent answer formatting
    """
    
    def __init__(self):
        """Initialize LLM client."""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None
    
    def generate(
        self, 
        query: str, 
        graph_context: List[Dict[str, Any]], 
        vector_context: List[Dict[str, Any]],
        decomposed_query: Optional[Any] = None
    ) -> str:
        """
        Generate answer from query and context.
        
        Args:
            query: User's question
            graph_context: Retrieved graph nodes and relationships
            vector_context: Retrieved vector chunks
            decomposed_query: Optional DecomposedQuery for context-aware prompting
            
        Returns:
            Generated answer
        """
        if not self.model:
            return self._fallback_answer(query)
        
        try:
            prompt = self._build_answer_prompt(
                query, 
                graph_context, 
                vector_context,
                decomposed_query
            )
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error in answer generation: {e}")
            return self._fallback_answer(query)
    
    def _build_answer_prompt(
        self, 
        query: str, 
        graph_context: List[Dict[str, Any]], 
        vector_context: List[Dict[str, Any]],
        decomposed_query: Optional[Any] = None
    ) -> str:
        """Build the answer generation prompt with decomposition insights."""
        
        # Format graph context
        graph_context_str = self._format_graph_context(graph_context)
        
        # Format vector context
        vector_context_str = self._format_vector_context(vector_context)
        
        # NEW: Add decomposition context to prompt
        decomposition_hint = ""
        if decomposed_query:
            intent_list = ", ".join([i.value for i in decomposed_query.sub_intents])
            entity_list = ", ".join([f"{e['value']} ({e['type']})" for e in decomposed_query.entities])
            
            decomposition_hint = f"""
QUERY ANALYSIS:
- Primary Intent: {decomposed_query.primary_intent.value}
- Query Sub-Intents: {intent_list}
- Extracted Entities: {entity_list if entity_list else "None"}
- Temporal Constraint: {decomposed_query.temporal_constraint if decomposed_query.temporal_constraint else "None"}
- Decomposition Confidence: {decomposed_query.confidence}

Focus your answer on addressing the identified intents and entities.
"""
        
        prompt = f"""You are a financial assistant. Answer the user's question using ONLY the provided context.

{decomposition_hint}

RULES:
1. Use ONLY information from the context below
2. If the context doesn't contain enough information, say "I don't have enough information to answer that"
3. Be specific and cite facts from the context
4. Keep answers concise and actionable
5. If analyzing financial alignment, provide clear reasoning
6. Format multi-part answers with clear sections if multiple intents detected

GRAPH CONTEXT (Structured Financial Data):
{graph_context_str}

VECTOR CONTEXT (Text Memories):
{vector_context_str}

USER QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def _format_graph_context(self, graph_context: List[Dict[str, Any]]) -> str:
        """Format graph context for the prompt."""
        if not graph_context:
            return "No structured data available."
        
        formatted_sections = []
        
        # Separate nodes and relationships
        nodes = [item for item in graph_context if item.get("type") != "relationship"]
        relationships = [item for item in graph_context if item.get("type") == "relationship"]
        
        # Format nodes by type
        if nodes:
            nodes_by_type = {}
            for node in nodes:
                node_type = node.get("type", "Unknown")
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                nodes_by_type[node_type].append(node)
            
            for node_type, type_nodes in nodes_by_type.items():
                formatted_sections.append(f"\n{node_type} Entities:")
                for node in type_nodes:
                    props = node.get("properties", {})
                    if props:
                        prop_str = ", ".join([f"{k}: {v}" for k, v in props.items() if k != "id" and k != "user_id"])
                        formatted_sections.append(f"  - {prop_str}")
                    else:
                        formatted_sections.append(f"  - {node_type} (no properties)")
        
        # Format relationships
        if relationships:
            formatted_sections.append("\nRelationships:")
            for rel in relationships:
                rel_type = rel.get("relationship_type", "RELATED_TO")
                from_node = rel.get("from", {})
                to_node = rel.get("to", {})
                props = rel.get("properties", {})
                
                from_label = from_node.get("type", "Node")
                to_label = to_node.get("type", "Node")
                
                rel_str = f"  - {from_label} --[{rel_type}]--> {to_label}"
                if props:
                    prop_str = ", ".join([f"{k}: {v}" for k, v in props.items()])
                    rel_str += f" ({prop_str})"
                formatted_sections.append(rel_str)
        
        return "\n".join(formatted_sections) if formatted_sections else "No structured data available."
    
    def _format_vector_context(self, vector_context: List[Dict[str, Any]]) -> str:
        """Format vector context for the prompt."""
        if not vector_context:
            return "No text memories available."
        
        chunks = [item.get("text", "") for item in vector_context]
        return "\n".join(f"- {chunk}" for chunk in chunks if chunk)
    
    def _fallback_answer(self, query: str) -> str:
        """Fallback answer when LLM is unavailable."""
        return f"I received your query: '{query}'. However, I need proper configuration to generate a detailed answer."
