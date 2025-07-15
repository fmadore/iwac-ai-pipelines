"""
This script uses Google's Gemini models to correct and improve OCR text output.
It processes text files containing raw OCR output, applies AI-powered corrections,
and saves the improved versions while handling large texts through chunking.
"""

import os
from google import genai
from google.genai import types
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_api_key():
    """
    Retrieve the Gemini API key from environment variables.
    
    Returns:
        str: The Gemini API key
        
    Raises:
        ValueError: If the API key is not found in environment variables
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")
    return api_key

def initialize_client(api_key):
    """
    Initialize the Gemini client with the provided API key.
    
    Args:
        api_key (str): Gemini API key
        
    Returns:
        genai.Client: Initialized Gemini client
    """
    return genai.Client(api_key=api_key)

def get_generation_config():
    """
    Configure generation parameters for optimal OCR correction performance.
    
    Returns:
        types.GenerateContentConfig: Configured generation config
    """
    return types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=65535,
        response_mime_type="text/plain",
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
    )

def get_system_instruction():
    """
    Get the specialized system instructions for OCR text correction from markdown file.
    
    Returns:
        str: Detailed system instruction for OCR correction
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(script_dir, 'ocr_correction_prompt.md')
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the instruction text by removing the markdown header
        # Split by lines and filter out markdown formatting
        lines = content.split('\n')
        instruction_lines = []
        for line in lines:
            # Skip markdown headers and empty lines at the beginning
            if line.strip() and not line.startswith('#'):
                instruction_lines.append(line)
        
        return '\n'.join(instruction_lines).strip()
    
    except FileNotFoundError:
        # Fallback to a basic instruction if file is missing
        return "You are an expert OCR correction assistant. Correct OCR errors, fix formatting, and improve text structure while preserving the original meaning."

def correct_text_with_gemini(client, text):
    """
    Use Gemini to correct OCR errors and improve text formatting.
    
    Args:
        client (genai.Client): Initialized Gemini client
        text (str): Raw OCR text to be corrected
        
    Returns:
        str: Corrected and improved text
    """
    try:
        config = get_generation_config()
        config.system_instruction = get_system_instruction()
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=f"Here is the OCR text that needs correction. Analyze the structure, fix errors, and ensure proper formatting:\n\n{text}")
                    ]
                )
            ],
            config=config
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return text  # Return original text if correction fails

def split_text(text, max_tokens=65535):
    """
    Split long texts into manageable chunks for Gemini processing.
    
    This function implements intelligent text splitting that:
    - Respects sentence boundaries where possible
    - Handles extremely long sentences by splitting on word boundaries
    - Ensures chunks stay within token limits for Gemini processing
    
    Args:
        text (str): Input text to be split
        max_tokens (int): Maximum tokens per chunk (default: 65535 for Gemini)
        
    Returns:
        list: List of text chunks ready for processing
    """
    # Estimate: 1 token ≈ 4 characters, leaving room for system prompt and response
    # Using 50000 tokens as safe limit for content (leaving ~15000 tokens for prompt and overhead)
    max_chars = 50000 * 4
    
    # If text is shorter than limit, return as single chunk
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split into sentences (roughly) by splitting on periods followed by spaces
    sentences = text.replace('. ', '.|').split('|')
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed limit, save current chunk and start new one
        if current_length + sentence_length > max_chars:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Handle sentences that are longer than max_chars
            if sentence_length > max_chars:
                # Split long sentence into smaller pieces
                words = sentence.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    if temp_length + word_length > max_chars:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                            temp_chunk = []
                            temp_length = 0
                    temp_chunk.append(word)
                    temp_length += word_length
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            else:
                current_chunk.append(sentence)
                current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add any remaining text
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_file(client, input_file_path, output_file_path, max_length):
    """
    Process a single text file through the OCR correction pipeline.
    
    Args:
        client (genai.Client): Initialized Gemini client
        input_file_path (str): Path to input text file
        output_file_path (str): Path where corrected text will be saved
        max_length (int): Maximum text length before chunking is required
    """
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        original_text = infile.read()

    if len(original_text) > max_length:
        chunks = split_text(original_text, max_length)
        corrected_chunks = [correct_text_with_gemini(client, chunk) for chunk in chunks]
        corrected_text = ' '.join(corrected_chunks)
    else:
        corrected_text = correct_text_with_gemini(client, original_text)

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write(corrected_text)

def process_txt_files(client, input_dir, output_dir, max_length=200000):
    """
    Process all text files in a directory through the OCR correction pipeline.
    
    Args:
        client (genai.Client): Initialized Gemini client
        input_dir (str): Directory containing input text files
        output_dir (str): Directory where corrected files will be saved
        max_length (int): Maximum text length before chunking is required
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    for txt_file in tqdm(txt_files, desc="Processing files"):
        input_file_path = os.path.join(input_dir, txt_file)
        output_file_path = os.path.join(output_dir, txt_file)
        process_file(client, input_file_path, output_file_path, max_length)

def main():
    """
    Main execution function that orchestrates the OCR correction process.
    
    Sets up directories, initializes the Gemini client, and processes all text files
    while providing progress feedback through tqdm.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set input and output directories relative to the script directory
    input_dir = os.path.join(script_dir, 'TXT')  # Directory containing the input .txt files
    output_dir = os.path.join(script_dir, 'Corrected_TXT')  # Directory to save the corrected .txt files
    
    api_key = get_api_key()
    client = initialize_client(api_key)
    process_txt_files(client, input_dir, output_dir)

if __name__ == "__main__":
    main()
