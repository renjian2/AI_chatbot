import csv
import io
from datetime import datetime


def export_chat_to_csv(chat_history, filename="chat_history.csv"):
    """
    Exports the chat history to a CSV file.
    
    Args:
        chat_history (list): List of chat messages with role and content
        filename (str): Name of the CSV file to create
        
    Returns:
        io.StringIO: CSV data as a string buffer
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["Timestamp", "Role", "Content", "Response Time (seconds)"])
    
    # Write chat history
    for i, message in enumerate(chat_history):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        role = message.get("role", "")
        content = message.get("content", "")
        response_time = message.get("time", "")
        
        # Clean content for CSV (remove newlines and excessive whitespace)
        if isinstance(content, str):
            # Handle encoding issues
            try:
                content = content.encode('ascii', 'ignore').decode('ascii')
            except:
                content = "Content could not be displayed due to encoding issues"
            content = content.replace('\n', ' ').replace('\r', ' ').strip()
        
        writer.writerow([timestamp, role, content, response_time])
    
    output.seek(0)
    return output


def export_chat_to_detailed_csv(chat_history, filename="chat_history_detailed.csv"):
    """
    Exports the chat history to a detailed CSV file with additional formatting.
    
    Args:
        chat_history (list): List of chat messages with role and content
        filename (str): Name of the CSV file to create
        
    Returns:
        str: CSV data as string
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        "Message Number",
        "Timestamp", 
        "Role",
        "Content", 
        "Response Time (seconds)",
        "Word Count",
        "Character Count"
    ])
    
    # Write chat history with additional details
    for i, message in enumerate(chat_history):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        role = message.get("role", "")
        content = message.get("content", "")
        response_time = message.get("time", "")
        
        # Calculate text statistics
        word_count = len(content.split()) if isinstance(content, str) else 0
        char_count = len(content) if isinstance(content, str) else 0
        
        # Clean content for CSV
        if isinstance(content, str):
            # Handle encoding issues
            try:
                content = content.encode('ascii', 'ignore').decode('ascii')
            except:
                content = "Content could not be displayed due to encoding issues"
            # Replace newlines with spaces for better CSV compatibility
            content = content.replace('\n', ' ').replace('\r', ' ').strip()
            # Handle quotes by escaping them
            content = content.replace('"', '""')
        
        writer.writerow([
            i + 1,
            timestamp, 
            role,
            content, 
            response_time,
            word_count,
            char_count
        ])
    
    output.seek(0)
    return output.getvalue()