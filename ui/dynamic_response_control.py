"""
Dynamic Response Control Module
"""

import re

def detect_repetitive_content(response_text, max_code_blocks=3, max_response_length=2000):
    """
    Detects if a response is becoming too repetitive or long.
    
    Args:
        response_text (str): The response text to analyze
        max_code_blocks (int): Maximum number of code blocks allowed
        max_response_length (int): Maximum length of response before truncation
        
    Returns:
        tuple: (is_repetitive, reason, truncated_response)
    """
    if not response_text:
        return False, "", response_text
    
    # Check if response is too long
    if len(response_text) > max_response_length:
        # Truncate to a reasonable length, preserving complete code blocks if possible
        truncated = truncate_long_response(response_text, max_response_length)
        return True, "Response too long", truncated
    
    # Count code blocks
    code_blocks = re.findall(r'```(?:\w+)?(.*?)```', response_text, re.DOTALL)
    if len(code_blocks) > max_code_blocks:
        # Keep only the first few code blocks
        truncated = truncate_excess_code_blocks(response_text, max_code_blocks)
        return True, f"Too many code blocks ({len(code_blocks)})", truncated
    
    # Check for repetitive patterns
    if detect_repetitive_patterns(response_text):
        truncated = remove_repetitive_patterns(response_text)
        return True, "Repetitive patterns detected", truncated
    
    return False, "", response_text

def truncate_long_response(response_text, max_length):
    """Truncates a long response while trying to preserve structure."""
    if len(response_text) <= max_length:
        return response_text
    
    # Try to find a good truncation point
    # Look for the last complete code block within the limit
    code_blocks = list(re.finditer(r'```(?:\w+)?(.*?)```', response_text, re.DOTALL))
    
    # If we have code blocks, try to preserve the last complete one
    if code_blocks:
        # Find the last code block that ends before max_length
        for match in reversed(code_blocks):
            if match.end() <= max_length:
                # Return everything up to the end of this code block
                return response_text[:match.end()] + "\n\n[Response truncated for brevity]"
    
    # If no suitable code block, just truncate at max_length
    return response_text[:max_length] + "\n\n[Response truncated for brevity]"

def truncate_excess_code_blocks(response_text, max_blocks):
    """Keeps only the first max_blocks code blocks."""
    # Find all code blocks
    pattern = r'```(?:\w+)?(.*?)```'
    matches = list(re.finditer(pattern, response_text, re.DOTALL))
    
    if len(matches) <= max_blocks:
        return response_text
    
    # Find the end position of the max_blocks-th code block
    end_pos = matches[max_blocks - 1].end()
    
    # Return everything up to that point plus a note
    return response_text[:end_pos] + "\n\n[Additional code blocks truncated for brevity]"

def detect_repetitive_patterns(response_text):
    """Detects repetitive patterns in the response."""
    lines = response_text.split('\n')
    
    # Check for consecutive duplicate lines (3 or more identical lines)
    for i in range(len(lines) - 2):
        if (lines[i].strip() and 
            lines[i].strip() == lines[i + 1].strip() and 
            lines[i].strip() == lines[i + 2].strip()):
            return True
    
    # Check for repeated blocks with minor variations (5 or more similar lines)
    if len(lines) > 5:
        similar_count = 0
        for i in range(len(lines) - 1):
            if similarity(lines[i].strip(), lines[i + 1].strip()) > 0.9:  # 90% similar
                similar_count += 1
        # If more than 60% of consecutive lines are very similar
        if similar_count > len(lines) * 0.6:
            return True
    
    return False

def remove_repetitive_patterns(response_text):
    """Removes repetitive patterns from the response."""
    lines = response_text.split('\n')
    unique_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        # Add line if it's not a duplicate of recent lines
        if not any(similarity(stripped_line, prev.strip()) > 0.95 for prev in unique_lines[-3:]):
            unique_lines.append(line)
    
    result = '\n'.join(unique_lines)
    if len(result) < len(response_text):
        result += "\n\n[Repetitive content removed]"
    
    return result

def similarity(a, b):
    """Simple similarity measure between two strings."""
    if not a or not b:
        return 0.0
    
    # Convert to sets of words
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    
    # Calculate Jaccard similarity
    intersection = words_a.intersection(words_b)
    union = words_a.union(words_b)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def enforce_response_limits(response_text, user_input=""):
    """
    Enforces dynamic limits on response length and repetitiveness.
    
    Args:
        response_text (str): The response to limit
        user_input (str): The original user input for context
        
    Returns:
        str: The limited response
    """
    # For technical queries, allow slightly longer responses
    is_technical = any(keyword in user_input.lower() for keyword in 
                      ['code', 'sql', 'query', 'select', 'create', 'procedure', 'function',
                       'class', 'method', 'algorithm', 'python', 'java', 'javascript'])
    
    max_length = 3000 if is_technical else 1500
    max_code_blocks = 5 if is_technical else 2
    
    is_repetitive, reason, truncated = detect_repetitive_content(
        response_text, max_code_blocks, max_length
    )
    
    if is_repetitive:
        # Add a note about why the response was truncated
        if reason:
            truncated += f"\n\n[Note: Response was truncated due to {reason}]"
        return truncated
    
    return response_text

# Export all functions
__all__ = [
    'detect_repetitive_content',
    'truncate_long_response',
    'truncate_excess_code_blocks',
    'detect_repetitive_patterns',
    'remove_repetitive_patterns',
    'similarity',
    'enforce_response_limits'
]