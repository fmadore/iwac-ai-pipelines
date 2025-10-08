# AI-Powered Handwritten Text Recognition (HTR)

A high-precision HTR system using Google's Gemini vision model for research-grade transcription of handwritten manuscripts in multiple languages. This tool is engineered for academic research and archival preservation, maintaining exact typography, reading order, and structural semantics.

## Features

- **Multi-language Support**: 
  - French manuscripts with proper typography (spacing rules, accents, de-hyphenation)
  - Arabic manuscripts with right-to-left text flow and traditional orthography
  - Multilingual/Auto-detect mode for other languages (Russian, Spanish, Persian, Chinese, etc.)
  
- **Research-Grade Quality**: Archival-quality transcription with zero summarization
- **Intelligent Zone Processing**: Recognizes columns, marginalia, headers, footers, and annotations
- **Semantic Line Joining**: Merges hyphenated words and joins lines within paragraphs
- **Multiple AI Models**: Choose between Gemini 2.5 Flash (fast) or Pro (more accurate)
- **Batch Processing**: Process multiple PDF files automatically
- **Robust Error Handling**: Retry logic, copyright detection handling, safety filtering

## Requirements

### Python Dependencies
```bash
pip install google-genai pdf2image pillow python-dotenv tqdm
```

### System Requirements

1. **Poppler PDF utilities** (for PDF to image conversion)
   - Windows: Install to `C:\Program Files\poppler\Library\bin`
   - Download from: https://github.com/oschwartz10612/poppler-windows/releases/
   - Linux: `sudo apt-get install poppler-utils`
   - macOS: `brew install poppler`

2. **Google Gemini API Key**
   - Get your API key from: https://aistudio.google.com/app/apikey
   - Create a `.env` file in this directory with:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fmadore/iwac-ai-pipelines.git
cd iwac-ai-pipelines/AI_htr_extraction
```

2. Install Python dependencies:
```bash
pip install -r ../requirements.txt
```

3. Install Poppler (see System Requirements above)

4. Create `.env` file with your Gemini API key:
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Usage

1. Place your PDF files in the `PDF/` directory

2. Run the script:
```bash
python 02_gemini_htr_processor.py
```

3. Follow the interactive prompts:
   - **Choose manuscript language:**
     - `1` - French handwritten manuscripts
     - `2` - Arabic handwritten manuscripts  
     - `3` - Multilingual/Other languages (AI auto-detects)
   
   - **Choose Gemini model:**
     - `1` - gemini-2.5-flash (Faster, good for most cases)
     - `2` - gemini-2.5-pro (More powerful, potentially more accurate)

4. Results will be saved in `OCR_Results/` as `.txt` files

## Output Format

### French & Arabic
Pure transcribed text with:
- Proper paragraph structure (double newlines between paragraphs)
- Language-specific typography rules applied
- De-hyphenated words (for French)
- Page markers for multi-page documents

### Multilingual Mode
Transcription begins with language detection header:
```
[LANGUAGE DETECTED: Russian]
[WRITING SYSTEM: Cyrillic]
[TEXT DIRECTION: Left-to-right]

<transcribed text follows>
```

## System Prompts

The system uses specialized prompts for each language mode:

- **`htr_system_prompt_french.md`**: French-specific rules (left-to-right, de-hyphenation, French spacing)
- **`htr_system_prompt_arabic.md`**: Arabic-specific rules (right-to-left, ligatures, traditional orthography)
- **`htr_system_prompt_multilingual.md`**: Auto-detection protocol with multi-script support

You can customize these prompts to adjust transcription behavior for your specific needs.

## Logging

Processing logs are saved to `log/ocr_gemini.log` with detailed information about:
- PDF conversion status
- API interactions
- Processing errors
- Retry attempts

## Project Structure

```
AI_htr_extraction/
├── 02_gemini_htr_processor.py          # Main HTR script
├── htr_system_prompt_french.md         # French transcription rules
├── htr_system_prompt_arabic.md         # Arabic transcription rules
├── htr_system_prompt_multilingual.md   # Auto-detect transcription rules
├── README.md                            # This file
├── PDF/                                 # Input directory (place PDFs here)
├── OCR_Results/                         # Output directory (transcriptions)
└── log/                                 # Processing logs
    └── ocr_gemini.log
```

## Advanced Configuration

### Adjusting Image Quality
Edit `pdf_to_images()` method in the script to modify:
- **DPI** (default: 300): Higher = better quality but slower
- **Grayscale** (default: True): Color images may help with some documents
- **Timeout** (default: 180s): Increase for very large PDFs

### Model Parameters
Edit `_setup_generation_config()` to adjust:
- **temperature** (default: 0.2): Lower = more consistent, higher = more creative
- **max_output_tokens** (default: 65535): Maximum length of transcription
- **top_p/top_k**: Adjust for different sampling behavior

## Troubleshooting

### PDF Conversion Fails
- Ensure Poppler is installed correctly
- Check PDF file is not corrupted
- Try increasing timeout value
- Check log file for specific errors

### API Errors
- Verify GEMINI_API_KEY is set correctly
- Check API quota/rate limits
- Wait a moment and retry (automatic retry is built-in)

### Empty Transcriptions
- Check if image quality is sufficient (try higher DPI)
- Review log file for safety/copyright blocks
- Try different model (Flash vs Pro)

### Language Detection Issues (Multilingual Mode)
- Switch to specific language mode if known
- Ensure manuscript has clear text visible
- Check log for any processing errors

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{iwac_htr_2025,
  title={AI-Powered Handwritten Text Recognition for Archival Documents},
  author={IWAC AI Pipelines Project},
  year={2025},
  url={https://github.com/fmadore/iwac-ai-pipelines}
}
```

## License

This project is part of the IWAC AI Pipelines suite. See the main repository for license information.

## Contributing

Contributions are welcome! Please:
1. Test your changes thoroughly
2. Update documentation as needed
3. Follow existing code style
4. Submit pull requests to the main repository

## Support

For issues, questions, or contributions:
- Open an issue on GitHub: https://github.com/fmadore/iwac-ai-pipelines/issues
- Review existing documentation in the main repository

## Acknowledgments

- Built with Google's Gemini AI vision models
- PDF processing via Poppler and pdf2image
- Designed for the IWAC (Islamic West Africa Collection) project
