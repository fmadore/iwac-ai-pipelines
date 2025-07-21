# AI Named Entity Recognition (NER) Module

This module provides a complete Named Entity Recognition pipeline using Google's Gemini AI for Omeka S Collections. The pipeline consists of three sequential scripts that extract, reconcile, and update named entities in the IWAC (Islam West Africa Collection) database.

## Pipeline Overview

The NER module operates as a three-stage pipeline:

1. **Entity Extraction** - Extract named entities from Omeka S collection items using AI
2. **Authority Reconciliation** - Match extracted entities against existing authority records
3. **Database Update** - Update Omeka S items with reconciled entity links

## Files

### Core Scripts
- `01_NER_AI_Gemini.py` - **Entity Extraction**: Uses Google Gemini AI to extract named entities (persons, organizations, locations, subjects) from Omeka S collection items
- `02_NER_reconciliation_Omeka.py` - **Authority Reconciliation**: Matches AI-extracted entities against existing Omeka S authority records (spatial coverage, subjects)
- `03_Omeka_update.py` - **Database Update**: Updates Omeka S items with reconciled entity links, preserving existing data and avoiding duplicates

### Configuration
- `ner_system_prompt.md` - **NER System Prompt**: Detailed French-language prompt template defining extraction rules for the IWAC collection context

## Features

### Script 1: Entity Extraction (`01_NER_AI_Gemini.py`)
- Asynchronous and synchronous processing modes
- Configurable batch size for optimized performance  
- Rate limiting to avoid API throttling
- Proper error handling and retry mechanisms
- Command-line argument support
- Detailed progress tracking and statistics
- **Modular prompt system using external markdown file**
- Spatial coverage filtering capabilities
- Structured output with Pydantic models

### Script 2: Authority Reconciliation (`02_NER_reconciliation_Omeka.py`)
- Multi-stage reconciliation process (spatial → subject/topic)
- Authority dictionary building from multiple item sets
- Ambiguous term detection and reporting
- Unreconciled entity tracking and export
- Comprehensive matching statistics
- Preserves original data through iterative processing

### Script 3: Database Update (`03_Omeka_update.py`)
- Safe item updating with duplicate prevention
- Preserves existing metadata and relationships
- Processes reconciled CSV output from script 2
- Comprehensive error handling and logging
- Updates both spatial coverage (dcterms:spatial) and subjects (dcterms:subject)

## NER System Prompt

The NER processing uses a sophisticated modular prompt system where detailed extraction instructions are stored in `ner_system_prompt.md`. This French-language prompt is specifically designed for the IWAC (Islam West Africa Collection) context.

### Prompt Benefits
- Easy modification of extraction rules without changing code
- Better version control for prompt changes
- Collaborative editing of prompts
- Clear separation of concerns between logic and instructions
- Context-specific rules for West African Islamic collections

### Prompt Structure

The prompt file contains detailed specifications for the IWAC collection:

1. **Context Definition**: Specific to Islam and Muslims in Burkina Faso, Benin, Niger, Nigeria, Togo, and Côte d'Ivoire
2. **Four Entity Categories** with precise formatting rules:
   - **Personnes (Persons)**: Full names without titles (religious, civil, honorific)
   - **Organisations**: Full organization names without acronyms, excluding parishes
   - **Lieux (Places)**: Simplified geographical names without political qualifiers
   - **Sujets (Subjects)**: Simple thematic keywords (max 8 per text) covering religion, society, politics, economy
3. **Extensive Examples**: Specific to West African context and Islamic terminology
4. **Output Format**: JSON structure specification
5. **Text Placeholder**: `{text_content}` for dynamic content insertion

### Subject Categories Covered
- **Religion**: islam, fiqh, sharia, hajj, ramadan, salafisme, soufisme, confrérie, etc.
- **Society**: mariage, éducation, famille, jeunesse, femmes, etc.
- **Politics**: laïcité, démocratie, élections, radicalisation, etc.
- **Economy**: commerce, agriculture, développement, etc.

### Modifying the Prompt

To modify the NER extraction behavior:
1. Edit the `ner_system_prompt.md` file
2. Update the formatting rules, examples, or subject categories as needed
3. The changes will be automatically picked up on the next script run

## Workflow

### Complete Pipeline Execution

1. **Run Entity Extraction**:
   ```bash
   python 01_NER_AI_Gemini.py --item-set-id 123 --async --batch-size 20 --timeout 30
   ```
   - Outputs: `item_set_123_processed.csv` in `/output` directory

2. **Run Authority Reconciliation**:
   ```bash
   python 02_NER_reconciliation_Omeka.py
   ```
   - Automatically finds latest CSV from step 1
   - Outputs: `*_reconciled.csv`, `*_unreconciled_*.csv`, `*_ambiguous_authorities_*.csv`

3. **Update Database**:
   ```bash
   python 03_Omeka_update.py
   ```
   - Automatically finds latest reconciled CSV from step 2
   - Updates Omeka S items with reconciled entity links

### Output Files

- **After Script 1**: `item_set_[ID]_processed.csv` - Raw AI-extracted entities
- **After Script 2**: 
  - `*_reconciled.csv` - Main file with reconciled entity IDs
  - `*_unreconciled_spatial.csv` - Unmatched spatial entities
  - `*_unreconciled_subject.csv` - Unmatched subject entities  
  - `*_ambiguous_authorities_*.csv` - Ambiguous authority matches
- **After Script 3**: Database updated, no additional files

## Requirements

### Environment Setup
- Python 3.7+
- Dependencies listed in `requirements.txt`
- Environment variables for API credentials:
  - `OMEKA_BASE_URL` - Base URL for Omeka S instance
  - `OMEKA_KEY_IDENTITY` - Omeka S API key identity  
  - `OMEKA_KEY_CREDENTIAL` - Omeka S API key credential
  - `GEMINI_API_KEY` - Google Gemini API key

### Required Files
- `ner_system_prompt.md` - Must be in the same directory as scripts
- `.env` file with API credentials

### Authority Item Sets (for Script 2)
The reconciliation script expects specific Omeka S item sets:
- **Spatial authorities**: Item set 268 (spatial coverage terms)
- **Subject authorities**: Item sets 854, 2, 266 (subject/topic terms)

## Notes

- All scripts include comprehensive logging and error handling
- The pipeline preserves existing Omeka S data and relationships
- Batch processing and rate limiting prevent API overload
- French language support throughout the pipeline
- Designed specifically for Islamic studies collections in West Africa
