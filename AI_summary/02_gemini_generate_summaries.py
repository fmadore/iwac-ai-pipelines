"""
This script uses Google's Gemini 2.5 Flash model to generate summaries of text files.
It processes text files from an input directory, generates a short French summary for each,
and saves these summaries to an output directory.
"""

import os
# Use the new SDK structure
from google import genai
# Add the types import as per the new SDK structure
from google.genai import types
# Use the new SDK's error handling
from google.genai import errors
from tqdm import tqdm
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Define the model name
# Using the preview model name mentioned in the blog post
MODEL_NAME = 'gemini-2.5-flash'

# Load the prompt template once at module level
def load_prompt_template():
    """
    Loads the prompt template from the markdown file.
    
    Returns:
        str: The prompt template with {text} placeholder.
        
    Raises:
        FileNotFoundError: If the prompt template file is not found.
        Exception: If there's an error reading the file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(script_dir, 'summary_prompt.md')
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Prompt template file not found: {prompt_file}")
        raise FileNotFoundError(f"Required prompt template file not found: {prompt_file}. Please ensure 'summary_prompt.md' exists in the same directory as this script.")
    except Exception as e:
        logging.error(f"Error reading prompt template: {e}")
        raise

# Load the prompt template once at startup
PROMPT_TEMPLATE = load_prompt_template()

def initialize_gemini_client():
    """
    Initializes the Google Generative AI Client using the new SDK.
    It tries to use Application Default Credentials (ADC) via GOOGLE_APPLICATION_CREDENTIALS first.
    If ADC is not configured or fails, it falls back to using GEMINI_API_KEY from the environment/.env file.

    Returns:
        genai.Client: Initialized Gemini client instance.

    Raises:
        ValueError: If neither ADC nor GEMINI_API_KEY allows successful client creation.
        errors.APIError: If there's an issue configuring the API.
    """
    google_api_key = None
    client = None
    auth_method = "Unknown"

    # 1. Try initializing client without API key (attempts ADC)
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if credentials_path:
        if os.path.exists(credentials_path):
            logging.info("Attempting client initialization using Google Application Default Credentials (ADC) from: %s", credentials_path)
            try:
                # The new SDK often picks up ADC automatically if GOOGLE_APPLICATION_CREDENTIALS is set
                # and no api_key is provided to Client()
                client = genai.Client()
                # Verify client works by listing models (optional, but good check)
                # client.list_models() # Uncomment to test connectivity
                auth_method = "ADC"
                logging.info("Gemini Client initialized successfully using ADC.")
            except Exception as e:
                logging.warning(f"ADC initialization failed: {e}. Falling back to API key.")
                client = None # Ensure client is None if ADC fails
        else:
            logging.warning(
                "GOOGLE_APPLICATION_CREDENTIALS is set, but file not found at: %s. "
                "Falling back to API key.", credentials_path
            )

    # 2. If client not initialized via ADC, try API Key
    if client is None:
        google_api_key = os.getenv('GEMINI_API_KEY')
        if google_api_key:
            logging.info("Attempting client initialization using GEMINI_API_KEY from environment.")
            try:
                # Pass the API key directly to the client constructor
                client = genai.Client(api_key=google_api_key)
                # Verify client works by listing models (optional, but good check)
                # client.list_models() # Uncomment to test connectivity
                auth_method = "API Key"
                logging.info("Gemini Client initialized successfully using API Key.")
            except Exception as e:
                 logging.error(f"Failed to initialize Gemini Client with API key: {e}")
                 raise ValueError("Failed to initialize Gemini Client with the provided GEMINI_API_KEY.") from e
        else:
            # 3. If neither method works, raise error
             raise ValueError(
                "Authentication failed. Gemini Client could not be initialized. "
                "Please configure ADC (GOOGLE_APPLICATION_CREDENTIALS) OR "
                "set the GEMINI_API_KEY in your environment/.env file."
            )

    # Return the initialized client
    if client:
        logging.info(f"Using model: {MODEL_NAME}")
        return client
    else:
        # This path should ideally not be reached due to prior checks/raises
        raise RuntimeError("Failed to initialize Gemini Client.")


def generate_summary_with_gemini(client, text):
    """
    Uses the Gemini client to generate a short summary of the text in French using the specified model.

    Args:
        client (genai.Client): Initialized Gemini client instance.
        text (str): The input text to summarize.

    Returns:
        str: The generated summary in French, or None if an error occurs.
    """
    # Use the pre-loaded prompt template
    prompt = PROMPT_TEMPLATE.format(text=text)

    try:
        # Define generation configuration with low temperature for factual summary
        # Corrected to types.GenerateContentConfig and parameter name to 'config'
        gen_config = types.GenerateContentConfig(
            temperature=0.2
            # Add other parameters like top_p, top_k here if needed
        )

        # Use client.models.generate_content directly as per the documentation
        response = client.models.generate_content(
            model=MODEL_NAME, # Pass model name directly
            contents=prompt,
            config=gen_config # Use 'config' parameter
        )
        # Ensure response and text attribute exist before accessing
        if response and hasattr(response, 'text'):
             # Add basic cleaning for potential markdown formatting if needed
            summary = response.text.strip().replace('*', '') # Example cleaning
            return summary
        else:
            logging.error("Received an unexpected response format from Gemini API.")
            # Log the full response for debugging if possible and privacy permits
            # logging.debug(f"Full Gemini response: {response}")
            return None # Indicate failure
    except errors.APIError as e:
        logging.error(f"Gemini API call failed: {e}")
        # You might want to inspect e.code or e.message for more info
        return None
    except Exception as e:
        # Catch other potential errors during generation or response processing
        logging.error(f"An unexpected error occurred during summary generation: {e}")
        return None

def process_file(client, input_file_path, output_file_path):
    """
    Reads a text file, generates its summary using the Gemini client, and saves the summary.

    Args:
        client (genai.Client): Initialized Gemini client instance.
        input_file_path (str): Path to the input text file.
        output_file_path (str): Path where the summary text file will be saved.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            original_text = infile.read()

        if not original_text.strip():
            logging.warning(f"Input file is empty, skipping: {input_file_path}")
            return # Skip empty files

        logging.info(f"Generating summary for: {os.path.basename(input_file_path)}")
        # Pass the client instance to the generation function
        summary_text = generate_summary_with_gemini(client, original_text)

        if summary_text:
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                outfile.write(summary_text)
            logging.info(f"Summary saved to: {os.path.basename(output_file_path)}")
        else:
            logging.error(f"Failed to generate summary for: {os.path.basename(input_file_path)}")

    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file_path}")
    except IOError as e:
        logging.error(f"Error processing file {input_file_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred processing file {input_file_path}: {e}")

def process_txt_files(client, input_dir, output_dir):
    """
    Processes all text files in an input directory, generating and saving summaries.

    Args:
        client (genai.Client): Initialized Gemini client instance.
        input_dir (str): Directory containing input .txt files.
        output_dir (str): Directory where summary .txt files will be saved.
    """
    if not os.path.exists(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Failed to create output directory {output_dir}: {e}")
            return

    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    if not txt_files:
        logging.warning(f"No .txt files found in input directory: {input_dir}")
        return

    logging.info(f"Found {len(txt_files)} .txt files to process.")

    # Pass the client instance when processing each file
    for txt_file in tqdm(txt_files, desc="Generating Summaries"):
        input_file_path = os.path.join(input_dir, txt_file)
        output_file_path = os.path.join(output_dir, txt_file) # Keep the same filename for the summary
        process_file(client, input_file_path, output_file_path)

def main():
    """
    Main execution function: sets up directories, initializes the Gemini client,
    and processes all text files to generate French summaries.
    """
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Set input and output directories relative to the script directory
        input_dir = os.path.join(script_dir, 'TXT')  # Directory containing the input .txt files
        output_dir = os.path.join(script_dir, 'Summaries_FR_TXT')  # Directory for French summaries

        # Initialize Gemini client (includes credential check)
        # Rename variable to reflect it's a client instance
        gemini_client = initialize_gemini_client()

        # Process the files using the client instance
        process_txt_files(gemini_client, input_dir, output_dir)

        logging.info("Summary generation process completed.")

    except ValueError as e:
        # Catch credential/initialization errors
        logging.error(f"Configuration or Initialization error: {e}")
    except FileNotFoundError as e:
        # Catch prompt template file missing error
        logging.error(f"Prompt template error: {e}")
    except errors.APIError as e:
        # Catch API call errors that might occur during initialization checks (if added)
        logging.error(f"Gemini API call error during setup: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}")

if __name__ == "__main__":
    main()
