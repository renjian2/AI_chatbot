"""
Utility functions for intelligent file chunking and selection
"""

def select_diverse_chunks(chunks, max_chunks):
    """
    Select diverse chunks from a document to maximize information coverage.
    
    Args:
        chunks (list): List of text chunks from the document
        max_chunks (int): Maximum number of chunks to select
        
    Returns:
        list: Selected chunks that provide good coverage of the document
    """
    if not chunks or max_chunks <= 0:
        return []
    
    if len(chunks) <= max_chunks:
        return chunks
    
    # Always include first and last chunks as they often contain important info
    selected = [chunks[0]]
    if len(chunks) > 1:
        selected.append(chunks[-1])
    
    # Fill remaining slots with strategically placed chunks
    remaining_slots = max_chunks - len(selected)
    if remaining_slots > 0:
        # Divide the document into segments and pick representative chunks
        segment_size = len(chunks) // (remaining_slots + 1)
        for i in range(1, remaining_slots + 1):
            # Pick chunk from the middle of each segment
            segment_start = i * segment_size
            segment_end = min((i + 1) * segment_size, len(chunks) - 1)
            chunk_index = (segment_start + segment_end) // 2
            if chunk_index < len(chunks):
                selected.append(chunks[chunk_index])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_selected = []
    for chunk in selected:
        if chunk not in seen:
            seen.add(chunk)
            unique_selected.append(chunk)
    
    return unique_selected[:max_chunks]


def rank_chunks_by_importance(chunks, query=None):
    """
    Rank chunks by importance for a given query or in general.
    
    Args:
        chunks (list): List of text chunks
        query (str, optional): Query to rank chunks by relevance to
        
    Returns:
        list: Chunks sorted by importance/relevance
    """
    # Simple heuristic ranking:
    # 1. Longer chunks might be more informative
    # 2. Chunks with more sentences might be more complete
    # 3. If query is provided, chunks containing query terms are ranked higher
    
    def chunk_score(chunk):
        score = 0
        # Prefer longer chunks (but not too long)
        chunk_length = len(chunk)
        if 100 <= chunk_length <= 2000:
            score += chunk_length * 0.1
        elif chunk_length > 2000:
            # Penalize very long chunks
            score += 200 - (chunk_length - 2000) * 0.05
            
        # Prefer chunks with more sentences
        sentence_count = len(chunk.split('.'))
        score += sentence_count * 2
        
        # If we have a query, boost chunks containing query terms
        if query:
            query_terms = query.lower().split()
            chunk_lower = chunk.lower()
            for term in query_terms:
                if term in chunk_lower:
                    score += 10
        
        return score
    
    # Sort chunks by score (descending)
    return sorted(chunks, key=chunk_score, reverse=True)


def adaptive_chunk_sizing(content_length, max_total_tokens, min_chunks=3, max_chunks=15):
    """
    Calculate adaptive chunk sizes based on content length and token limits.
    
    Args:
        content_length (int): Length of the content in characters
        max_total_tokens (int): Maximum tokens available for file context
        min_chunks (int): Minimum number of chunks to create
        max_chunks (int): Maximum number of chunks to create
        
    Returns:
        tuple: (chunk_size_tokens, num_chunks)
    """
    # Estimate average tokens per character (rough approximation)
    avg_tokens_per_char = 0.25
    
    # Estimate total tokens needed for the content
    estimated_total_tokens = int(content_length * avg_tokens_per_char)
    
    # If content fits within limits, use it all
    if estimated_total_tokens <= max_total_tokens:
        return min(max_total_tokens // max(1, min_chunks), 1024), min_chunks
    
    # Calculate optimal chunk size and number
    optimal_chunks = max(min_chunks, min(max_chunks, max_total_tokens // 256))  # At least 256 tokens per chunk
    optimal_chunk_size = max(256, min(1024, max_total_tokens // optimal_chunks))
    
    return optimal_chunk_size, optimal_chunks