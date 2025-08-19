"""Unified French Summary Generation (Gemini or OpenAI)

Generates French summaries for all .txt files in TXT/ and writes them to Summaries_FR_TXT/.
User selects provider interactively:
  1) OpenAI Responses API (model: gpt-5-mini)
  2) Google Gemini (model: gemini-2.5-flash)

OpenAI path uses the EXACT required responses.create structure (only the input list contents vary).
Gemini path uses google-genai SDK with GenerateContent.
"""

import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm

from google import genai
from google.genai import types, errors

try:  # OpenAI optional import
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-5-mini"
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"

# ------------------------------------------------------------------
# Prompt Loading
# ------------------------------------------------------------------
def load_prompt_template() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(script_dir, 'summary_prompt.md')
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if '{text}' not in content:
            logging.warning("Prompt template missing '{text}' placeholder.")
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read prompt template {prompt_file}: {e}")

PROMPT_TEMPLATE = load_prompt_template()

# ------------------------------------------------------------------
# Client Initialization
# ------------------------------------------------------------------
def initialize_gemini_client():
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    client = None
    auth_method = "API Key"
    if credentials_path and os.path.exists(credentials_path):
        try:
            client = genai.Client()
            auth_method = "ADC"
            logging.info("Gemini client initialized via ADC.")
        except Exception as e:
            logging.warning(f"ADC init failed: {e}; falling back to API key.")
            client = None
    if client is None:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set and ADC not available.")
        client = genai.Client(api_key=api_key)
        logging.info("Gemini client initialized via API key.")
    logging.info(f"Using Gemini model: {GEMINI_MODEL} ({auth_method})")
    return client

def initialize_openai_client() -> Optional[object]:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logging.error("OPENAI_API_KEY not set.")
        return None
    if OpenAI is None:
        logging.error("openai package not installed.")
        return None
    logging.info(f"Using OpenAI model: {OPENAI_MODEL}")
    return OpenAI()

# ------------------------------------------------------------------
# Generation Functions
# ------------------------------------------------------------------
def generate_summary_openai(client, text: str) -> Optional[str]:
    if not text.strip():
        return None
    system_prompt = PROMPT_TEMPLATE
    user_prompt = system_prompt.format(text=text)
    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            text={
                "format": {"type": "text"},
                "verbosity": "low"
            },
            reasoning={"effort": "low"},
            tools=[],
            store=True
        )
        raw_output = getattr(response, 'output_text', None)
        if not raw_output:
            segments = []
            for seg in getattr(response, 'output', []) or []:
                if isinstance(seg, dict):
                    c = seg.get('content')
                    if isinstance(c, str):
                        segments.append(c)
            raw_output = '\n'.join(filter(None, segments))
        return raw_output.strip() if raw_output else None
    except Exception as e:
        logging.error(f"OpenAI summary error: {e}")
        return None

def generate_summary_gemini(client, text: str) -> Optional[str]:
    if not text.strip():
        return None
    prompt = PROMPT_TEMPLATE.format(text=text)
    try:
        gen_config = types.GenerateContentConfig(temperature=0.2)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=gen_config
        )
        if response and hasattr(response, 'text'):
            return response.text.strip().replace('*', '')
        logging.error("Unexpected Gemini response format.")
        return None
    except errors.APIError as e:
        logging.error(f"Gemini API error: {e}")
        return None
    except Exception as e:
        logging.error(f"Gemini summary error: {e}")
        return None

# ------------------------------------------------------------------
# File Processing
# ------------------------------------------------------------------
def process_file(provider: str, client, input_file_path: str, output_file_path: str):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            original_text = infile.read()
        if not original_text.strip():
            logging.warning(f"Empty file skipped: {input_file_path}")
            return
        logging.info(f"Summarizing: {os.path.basename(input_file_path)}")
        if provider == PROVIDER_GEMINI:
            summary = generate_summary_gemini(client, original_text)
        else:
            summary = generate_summary_openai(client, original_text)
        if summary:
            with open(output_file_path, 'w', encoding='utf-8') as out:
                out.write(summary)
            logging.info(f"Saved: {os.path.basename(output_file_path)}")
        else:
            logging.error(f"No summary produced for {input_file_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {input_file_path}")
    except IOError as e:
        logging.error(f"IO error {input_file_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error {input_file_path}: {e}")

def process_txt_files(provider: str, client, input_dir: str, output_dir: str):
    if not os.path.exists(input_dir):
        logging.error(f"Input dir not found: {input_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if not txt_files:
        logging.warning("No .txt files to process.")
        return
    logging.info(f"Processing {len(txt_files)} files -> {output_dir}")
    for fname in tqdm(txt_files, desc="Generating Summaries"):
        process_file(provider,
                     client,
                     os.path.join(input_dir, fname),
                     os.path.join(output_dir, fname))
    logging.info("Batch complete.")

# ------------------------------------------------------------------
# User Interaction & Main
# ------------------------------------------------------------------
def select_provider() -> str:
    while True:
        choice = input("Select AI model: 1) ChatGPT (OpenAI)  2) Google Gemini  > ").strip()
        if choice == '1':
            return PROVIDER_OPENAI
        if choice == '2':
            return PROVIDER_GEMINI
        print("Invalid choice. Enter 1 or 2.")

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'TXT')
        output_dir = os.path.join(script_dir, 'Summaries_FR_TXT')
        logging.info("Starting summary generation pipeline")
        logging.info(f"Input: {input_dir}")
        logging.info(f"Output: {output_dir}")
        provider = select_provider()
        if provider == PROVIDER_GEMINI:
            client = initialize_gemini_client()
        else:
            client = initialize_openai_client()
            if client is None:
                return
        process_txt_files(provider, client, input_dir, output_dir)
        logging.info("Completed successfully")
    except FileNotFoundError as e:
        logging.error(e)
    except ValueError as e:
        logging.error(e)
    except errors.APIError as e:
        logging.error(f"Gemini API setup error: {e}")
    except Exception as e:
        logging.error(f"Unexpected failure: {e}")

if __name__ == '__main__':
    main()
