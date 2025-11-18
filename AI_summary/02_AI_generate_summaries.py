"""French Summary Generation using OpenAI GPT-5.1 mini or Gemini 2.5 Flash.

Optimized for cost-effective document summarization with low reasoning effort.
"""

import os
import sys
import logging
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from common.llm_provider import (  # noqa: E402
    BaseLLMClient,
    LLMConfig,
    build_llm_client,
    get_model_option,
    summary_from_option,
)

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

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
# Generation Helper
# ------------------------------------------------------------------
def generate_summary(llm_client: BaseLLMClient, text: str) -> Optional[str]:
    """Generate a summary using the configured LLM client."""
    if not text.strip():
        return None
    system_prompt = PROMPT_TEMPLATE
    user_prompt = system_prompt.format(text=text)
    try:
        raw_output = llm_client.generate(system_prompt, user_prompt)
        if raw_output:
            return raw_output.strip().replace('*', '')
        logging.error("Model returned empty summary.")
        return None
    except Exception as exc:
        logging.error(f"Summary generation error: {exc}")
        return None

# ------------------------------------------------------------------
# File Processing
# ------------------------------------------------------------------
def process_file(llm_client: BaseLLMClient, input_file_path: str, output_file_path: str):
    """Process a single text file and generate its summary."""
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            original_text = infile.read()
        if not original_text.strip():
            logging.warning(f"Empty file skipped: {input_file_path}")
            return
        logging.info(f"Summarizing: {os.path.basename(input_file_path)}")
        summary = generate_summary(llm_client, original_text)
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

def process_txt_files(llm_client: BaseLLMClient, input_dir: str, output_dir: str):
    """Process all text files in input directory."""
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
        process_file(
            llm_client,
            os.path.join(input_dir, fname),
            os.path.join(output_dir, fname),
        )
    logging.info("Batch complete.")

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'TXT')
        output_dir = os.path.join(script_dir, 'Summaries_FR_TXT')
        logging.info("Starting summary generation pipeline")
        logging.info(f"Input: {input_dir}")
        logging.info(f"Output: {output_dir}")
        
        # Get model selection (restricted to mini and flash)
        model_option = get_model_option(None, allowed_keys=["openai", "gemini-flash"])
        logging.info(f"Using AI model: {summary_from_option(model_option)}")
        
        # Configure for cost-effective summarization
        config = LLMConfig(
            reasoning_effort="low",      # OpenAI: quick summarization
            text_verbosity="low",        # OpenAI: concise output
            thinking_budget=0,           # Gemini Flash: no thinking needed for summaries
            temperature=0.3              # Gemini: some creativity for natural summaries
        )
        logging.info(f"Summary Config: reasoning_effort={config.reasoning_effort}, "
                    f"text_verbosity={config.text_verbosity}, thinking_budget={config.thinking_budget}")
        
        llm_client = build_llm_client(model_option, config=config)
        process_txt_files(llm_client, input_dir, output_dir)
        logging.info("Completed successfully")
    except (FileNotFoundError, ValueError) as err:
        logging.error(err)
    except Exception as exc:
        logging.error(f"Unexpected failure: {exc}")

if __name__ == '__main__':
    main()
