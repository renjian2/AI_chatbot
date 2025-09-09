"""
PDF Export Module for Chat History
Provides enhanced PDF export functionality with better formatting and styling.
"""

from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import re
import tempfile
import os
import base64


def export_chat_to_pdf(chat_history, filename="chat_export.pdf"):
    """Exports the chat history to a well-formatted PDF file that preserves the chat format."""
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=25)
    
    # Set document title
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 20, "AArete AI Chat Conversation", 0, 1, "C")
    
    # Add subtitle with timestamp
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(100, 100, 100)
    timestamp = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    pdf.cell(0, 10, f"Exported on {timestamp}", 0, 1, "C")
    pdf.ln(15)
    
    # Add conversation
    for i, message in enumerate(chat_history):
        role = message["role"]
        content = message["content"]
        
        # Handle both string and list content types
        if isinstance(content, list):
            # If content is a list, join the elements
            content_str = "\n".join(str(item) for item in content)
        else:
            # If content is a string, use it directly
            content_str = str(content)
        
        # Add message header with better formatting
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(255, 255, 255)  # White text
        
        # Set different colors for user and assistant
        if role.lower() == "user":
            pdf.set_fill_color(70, 130, 180)  # Steel blue for user
            role_text = "You"
        else:
            pdf.set_fill_color(40, 167, 69)  # Green for assistant
            role_text = "Assistant"
            
        pdf.cell(0, 10, f"{role_text}", 0, 1, "L", True)
        pdf.ln(2)
        
        # Process content with better formatting
        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(0, 0, 0)  # Black text
        
        # Handle Unicode characters and special formatting
        try:
            if isinstance(content_str, str):
                # Replace common Unicode characters that might cause issues with FPDF
                content_str = content_str.replace('\u2019', "'")  # Right single quotation mark
                content_str = content_str.replace('\u2018', "'")  # Left single quotation mark
                content_str = content_str.replace('\u201c', '"')  # Left double quotation mark
                content_str = content_str.replace('\u201d', '"')  # Right double quotation mark
                content_str = content_str.replace('\u2013', '-')  # En dash
                content_str = content_str.replace('\u2014', '--')  # Em dash
                content_str = content_str.replace('\u2022', '*')   # Bullet point
                
                # Remove any remaining problematic Unicode characters
                encoded_content = content_str.encode('ascii', 'ignore').decode('ascii')
            else:
                encoded_content = str(content_str)
                # Also clean non-string content
                encoded_content = encoded_content.encode('ascii', 'ignore').decode('ascii')
        except Exception:
            # Fallback: convert to ASCII
            encoded_content = "Content could not be displayed due to encoding issues"
        
        # Process markdown-like content for better formatting
        lines = encoded_content.split('\n')
        in_code_block = False
        code_block_content = []
        
        for line in lines:
            # Handle code blocks
            if line.strip().startswith("```"):
                if in_code_block:
                    # End code block
                    in_code_block = False
                    # Format code block with better styling
                    if code_block_content:
                        pdf.ln(2)
                        pdf.set_font("Courier", "", 10)
                        pdf.set_fill_color(245, 245, 245)
                        for code_line in code_block_content:
                            # Clean code line
                            clean_code_line = code_line.encode('ascii', 'ignore').decode('ascii')
                            if clean_code_line.strip():  # Only add non-empty lines
                                pdf.cell(0, 6, "", 0, 1, "L", True)  # Background
                                pdf.set_xy(15, pdf.get_y() - 6)
                                pdf.cell(0, 6, clean_code_line[:180], 0, 1)  # Limit line length
                        pdf.ln(2)
                        code_block_content = []
                        pdf.set_font("Arial", "", 11)
                else:
                    # Start code block
                    in_code_block = True
                    code_block_content = []
            elif in_code_block:
                # Collect code block content
                code_block_content.append(line)
            else:
                # Regular text processing
                line = line.strip()
                
                # Skip empty lines at the beginning
                if not line and pdf.get_y() < 35:
                    continue
                
                # Handle headers (markdown style)
                if line.startswith("# "):
                    pdf.set_font("Arial", "B", 14)
                    line = line[2:]
                    pdf.ln(2)
                    clean_line = line.encode('ascii', 'ignore').decode('ascii')
                    pdf.cell(0, 8, clean_line, 0, 1)
                    pdf.ln(1)
                elif line.startswith("## "):
                    pdf.set_font("Arial", "B", 13)
                    line = line[3:]
                    pdf.ln(1)
                    clean_line = line.encode('ascii', 'ignore').decode('ascii')
                    pdf.cell(0, 7, clean_line, 0, 1)
                    pdf.ln(1)
                elif line.startswith("### "):
                    pdf.set_font("Arial", "B", 12)
                    line = line[4:]
                    clean_line = line.encode('ascii', 'ignore').decode('ascii')
                    pdf.cell(0, 6, clean_line, 0, 1)
                    pdf.ln(1)
                # Handle bullet points (improved)
                elif line.startswith("- ") or line.startswith("* "):
                    pdf.set_font("Arial", "", 11)
                    line = line[2:]  # Remove the marker
                    # Clean line content
                    clean_line = line.encode('ascii', 'ignore').decode('ascii')
                    # Add bullet point with proper indentation
                    pdf.ln(1)
                    pdf.set_x(15)
                    pdf.cell(0, 6, "* " + clean_line, 0, 1)
                    pdf.ln(1)
                # Handle numbered lists
                elif re.match(r'^\d+\.\s', line):
                    pdf.set_font("Arial", "", 11)
                    # Clean line content
                    clean_line = line.encode('ascii', 'ignore').decode('ascii')
                    # Add numbered point with proper indentation
                    pdf.ln(1)
                    pdf.set_x(15)
                    pdf.cell(0, 6, clean_line, 0, 1)
                    pdf.ln(1)
                else:
                    # Regular paragraph text
                    pdf.set_font("Arial", "", 11)
                    # Clean line content
                    clean_line = line.encode('ascii', 'ignore').decode('ascii')
                    
                    # Handle empty lines
                    if not clean_line:
                        pdf.ln(3)
                    else:
                        # Add text with proper wrapping
                        pdf.ln(1)
                        pdf.set_x(10)
                        pdf.multi_cell(0, 6, clean_line)
                        pdf.ln(1)
        
        # Add response time if available
        if "time" in message and message["time"]:
            pdf.ln(2)
            pdf.set_font("Arial", "I", 9)
            pdf.set_text_color(128, 128, 128)  # Gray color
            pdf.cell(0, 8, f"Response time: {message['time']:.2f} seconds", 0, 1, "R")
        
        pdf.ln(8)  # Add space after each message
        
        # Add a subtle separator (except for the last message)
        if i < len(chat_history) - 1:
            pdf.set_draw_color(230, 230, 230)  # Light gray line
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)
    
    # Add footer with page number
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, "C")
    
    # Create a temporary file to store the PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.close()
    
    # Output the PDF to the temporary file
    pdf.output(temp_file.name)
    
    # Read the PDF data
    with open(temp_file.name, 'rb') as f:
        pdf_data = f.read()
    
    # Clean up the temporary file
    os.unlink(temp_file.name)
    
    # Return as BytesIO for Streamlit download
    return BytesIO(pdf_data)


def get_base64_encoded_image(image_path):
    """Returns the base64 encoded string of an image."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


def apply_custom_css():
    """Applies custom CSS to the Streamlit app."""
    import streamlit as st
    st.markdown(
        """
        <style>
        /* Custom CSS Variables */
        :root {
          --bg-100: #fbfbfb;
          --bg-200: #f1f1f1;
          --bg-300: #c8c8c8;
          --text-100: #232121;
          --primary-100: #c21d03;
          --accent-100: #007acc;
          --accent-200: #005a9e;
        }

        /* Global Styles */
        body {
          background-color: var(--bg-100);
          color: var(--text-100);
          font-family: 'Arial', sans-serif;
        }

        /* Main App Container */
        .stApp {
          background-color: var(--bg-100);
        }

        /* Sidebar Styles */
        section[data-testid="stSidebar"] {
          background-color: var(--bg-200);
        }

        /* Chat Input Area */
        .stChatInputContainer {
          background-color: var(--bg-200);
          border-radius: 8px;
          padding: 10px;
        }

        .stChatInput {
          background-color: white;
          border: 1px solid var(--bg-300);
          border-radius: 4px;
          padding: 10px;
        }

        /* Chat Message Styles */
        .stChatMessage {
          background-color: white;
          border-radius: 8px;
          padding: 15px;
          margin-bottom: 10px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .stChatMessage.user {
          background-color: var(--bg-200);
        }

        .stChatMessage.assistant {
          background-color: white;
        }

        /* Button Styles */
        .stButton>button {
          background-color: var(--primary-100);
          color: white;
          border: none;
          border-radius: 4px;
          padding: 8px 16px;
          font-weight: bold;
          transition: background-color 0.3s;
        }

        .stButton>button:hover {
          background-color: #a01500;
        }

        .stButton>button:active {
          background-color: #801000;
        }

        /* Expander Styles */
        .stExpander {
          background-color: var(--bg-200);
          border-radius: 8px;
          margin-bottom: 10px;
        }

        .stExpander>details>summary {
          background-color: var(--bg-300);
          border-radius: 8px;
          padding: 10px;
          font-weight: bold;
        }

        /* Text Input Styles */
        .stTextInput>div>div>input {
          background-color: white;
          border: 1px solid var(--bg-300);
          border-radius: 4px;
          padding: 8px;
        }

        /* Selectbox Styles */
        .stSelectbox>div>div>select {
          background-color: white;
          border: 1px solid var(--bg-300);
          border-radius: 4px;
          padding: 8px;
        }

        /* Slider Styles */
        .stSlider>div>div>div {
          background-color: var(--bg-300);
        }

        .stSlider>div>div>div>div {
          background-color: var(--primary-100);
        }
        </style>
        """,
        unsafe_allow_html=True
    )