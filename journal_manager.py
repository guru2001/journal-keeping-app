"""
Journal Manager - Handles journal entries storage and querying with vector search
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import numpy as np
from openai import OpenAI
import os

class JournalManager:
    """Manages journal entries in memory with vector search capabilities"""
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        self.entries: List[Dict[str, Any]] = []
        self.next_id = 1
        self.openai_client = openai_client
        self.embedding_model = "text-embedding-3-small"  # Cost-effective embedding model
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI"""
        if not self.openai_client:
            return None
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def add_entry(self, content: str, category: str = "general", extracted_items: List[str] = None, embedding: Optional[List[float]] = None) -> int:
        """Add a new entry to the journal with optional embedding"""
        # Generate embedding if not provided
        if embedding is None and self.openai_client:
            embedding = self._get_embedding(content)
        
        entry = {
            "id": self.next_id,
            "content": content,
            "category": category,
            "extracted_items": extracted_items or [],
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding  # Store embedding for vector search
        }
        self.entries.append(entry)
        self.next_id += 1
        return entry["id"]
    
    def query_entries(self, query: str, category_filter: Optional[str] = None, use_vector_search: bool = True, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query journal entries using vector search and keyword matching"""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Generate query embedding for vector search
        query_embedding = None
        if use_vector_search and self.openai_client:
            query_embedding = self._get_embedding(query)
        
        scored_entries = []
        
        for entry in self.entries:
            # Filter by category if specified
            if category_filter and entry["category"] != category_filter:
                continue
            
            score = 0.0
            entry_embedding = entry.get("embedding")
            
            # Vector similarity score (0-1)
            if query_embedding and entry_embedding:
                similarity = self._cosine_similarity(query_embedding, entry_embedding)
                score += similarity * 0.7  # Weight vector search at 70%
            
            # Keyword matching score
            content_lower = entry["content"].lower()
            category_lower = entry.get("category", "").lower()
            
            keyword_score = 0.0
            # Exact match
            if query_lower in content_lower:
                keyword_score += 0.3
            # Word matches
            word_matches = sum(1 for word in query_words if len(word) > 2 and word in content_lower)
            if query_words:
                keyword_score += (word_matches / len(query_words)) * 0.2
            # Category match
            if query_lower in category_lower:
                keyword_score += 0.1
            
            score += keyword_score * 0.3  # Weight keyword search at 30%
            
            if score > 0:  # Only include entries with some relevance
                scored_entries.append((entry, score))
        
        # Sort by score (highest first)
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return [entry for entry, score in scored_entries[:top_k]]
    
    def get_shopping_items(self) -> List[str]:
        """Get all shopping items from journal entries - relies on LLM's extracted_items"""
        all_items = set()
        
        for entry in self.entries:
            # Primary source: LLM-extracted items (most reliable)
            if entry.get("extracted_items"):
                for item in entry["extracted_items"]:
                    if item and item.strip():
                        all_items.add(item.strip())
            
            # Fallback: Only for entries categorized as shopping but missing extracted_items
            # This is a safety net, but LLM should handle extraction
            if entry["category"] == "shopping" and not entry.get("extracted_items"):
                # Simple fallback extraction for edge cases
                content_lower = entry["content"].lower()
                if "buy" in content_lower or "purchase" in content_lower:
                    # Try basic extraction as last resort
                    patterns = [
                        r"remind.*?to\s+buy\s+([^,.!?]+)",
                        r"buy\s+([^,.!?]+)",
                    ]
                    for pattern in patterns:
                        matches = re.findall(pattern, content_lower)
                        for match in matches:
                            item = match.strip()
                            item = re.sub(r"^(me|to|the|a|an|some|next time|at the)\s+", "", item)
                            item = re.sub(r"\s+(next time|at the|supermarket|store).*$", "", item)
                            if item and len(item) > 2:
                                all_items.add(item.title())
        
        return sorted(list(all_items))
    
    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Get all journal entries"""
        return self.entries
    
    def get_entries_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all entries in a specific category"""
        return [entry for entry in self.entries if entry["category"] == category]
    
    def clear_all(self):
        """Clear all journal entries (for testing/reset)"""
        self.entries = []
        self.next_id = 1
    
    def update_embeddings(self):
        """Update embeddings for all entries that don't have them"""
        if not self.openai_client:
            return
        
        for entry in self.entries:
            if not entry.get("embedding"):
                embedding = self._get_embedding(entry["content"])
                if embedding:
                    entry["embedding"] = embedding

