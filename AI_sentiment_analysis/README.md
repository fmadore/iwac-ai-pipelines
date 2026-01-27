# AI Sentiment Analysis Pipeline

Analyze sentiment towards Islam and Muslims in French-language West African media articles using three AI models concurrently (Gemini, ChatGPT, Mistral).

## Overview

This pipeline:
1. Fetches items from Omeka S based on item set ID(s)
2. Extracts `bibo:content` text from each item
3. Runs sentiment analysis with all 3 AI models concurrently
4. Caches results to JSON for resumability
5. Updates Omeka S items with sentiment analysis results

## Analysis Dimensions

The analysis evaluates three dimensions for each article:

### Centralit√© (Centrality)
How central Islam/Muslims are to the article:
- **Tr√®s central** - Islam/Muslims are the main subject
- **Central** - Important theme shared with others
- **Secondaire** - Mentioned significantly but secondary
- **Marginal** - Brief or anecdotal mention
- **Non abord√©** - No mention

### Subjectivit√© (Subjectivity)
Score from 1-5 measuring objectivity:
1. **Tr√®s objectif** - Pure facts, no opinions
2. **Plut√¥t objectif** - Mostly factual with subtle opinions
3. **Mixte** - Balanced mix of facts and opinions
4. **Plut√¥t subjectif** - Clear opinions supported by facts
5. **Tr√®s subjectif** - Strongly biased, editorial style

### Polarit√© (Polarity)
Sentiment towards Islam/Muslims:
- **Tr√®s positif** - Extremely favorable portrayal
- **Positif** - Favorable, optimistic
- **Neutre** - Balanced or factual without emotion
- **N√©gatif** - Critical, pessimistic
- **Tr√®s n√©gatif** - Extremely unfavorable, alarmist
- **Non applicable** - Subject not addressed

## Usage

```bash
# Basic usage with item set ID
python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123

# Multiple item sets
python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123,456,789

# Skip Omeka update (analysis and cache only)
python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123 --skip-update

# Force re-analysis of cached items
python AI_sentiment_analysis/01_sentiment_analysis.py --item-set-id 123 --force-reanalyze

# Interactive mode (prompts for item set ID)
python AI_sentiment_analysis/01_sentiment_analysis.py
```

## Environment Variables

Required in `.env`:

```bash
# Omeka S API (required)
OMEKA_BASE_URL=https://your-omeka-instance.com/api
OMEKA_KEY_IDENTITY=your_key_identity
OMEKA_KEY_CREDENTIAL=your_key_credential

# AI APIs (at least one required)
GEMINI_API_KEY=your_gemini_api_key      # or GOOGLE_API_KEY
OPENAI_API_KEY=your_openai_api_key      # or CHATGPT
MISTRAL_API_KEY=your_mistral_api_key
```

## Omeka S Property Mappings

Results are stored in IWAC ontology properties:

| Model | Property | Type |
|-------|----------|------|
| Gemini | `iwac:geminiCentralite` | resource:item |
| Gemini | `iwac:geminiCentraliteJustification` | literal |
| Gemini | `iwac:geminiPolarite` | resource:item |
| Gemini | `iwac:geminiPolariteJustification` | literal |
| Gemini | `iwac:geminiSubjectiviteScore` | numeric:integer |
| Gemini | `iwac:geminiSubjectiviteJustification` | literal |
| ChatGPT | `iwac:chatgptCentralite` | resource:item |
| ChatGPT | `iwac:chatgptCentraliteJustification` | literal |
| ChatGPT | `iwac:chatgptPolarite` | resource:item |
| ChatGPT | `iwac:chatgptPolariteJustification` | literal |
| ChatGPT | `iwac:chatgptSubjectiviteScore` | numeric:integer |
| ChatGPT | `iwac:chatgptSubjectiviteJustification` | literal |
| Mistral | `iwac:mistralCentralite` | resource:item |
| Mistral | `iwac:mistralCentraliteJustification` | literal |
| Mistral | `iwac:mistralPolarite` | resource:item |
| Mistral | `iwac:mistralPolariteJustification` | literal |
| Mistral | `iwac:mistralSubjectiviteScore` | numeric:integer |
| Mistral | `iwac:mistralSubjectiviteJustification` | literal |

## Controlled Vocabulary Item IDs

Centralit√© and Polarit√© values link to Omeka items:

### Centralit√©
| Value | Omeka Item ID |
|-------|---------------|
| Tr√®s central | 78048 |
| Central | 78049 |
| Secondaire | 78050 |
| Marginal | 78051 |
| Non abord√© | 78052 |

### Polarit√©
| Value | Omeka Item ID |
|-------|---------------|
| Tr√®s positif | 78031 |
| Positif | 78038 |
| Neutre | 78039 |
| N√©gatif | 78040 |
| Tr√®s n√©gatif | 78041 |
| Non applicable | 78042 |

### Subjectivit√© (for reference)
| Score | Label | Omeka Item ID |
|-------|-------|---------------|
| 1 | Tr√®s objectif | 78043 |
| 2 | Plut√¥t objectif | 78044 |
| 3 | Mixte | 78045 |
| 4 | Plut√¥t subjectif | 78046 |
| 5 | Tr√®s subjectif | 78047 |

## Caching

Results are cached in `AI_sentiment_analysis/cache/sentiment_cache.json`. The cache:
- Stores results keyed by Omeka item ID
- Allows resuming interrupted processing
- Automatically re-analyzes items with errors
- Can be bypassed with `--force-reanalyze`

## AI Models Used

| Model | Provider | Notes |
|-------|----------|-------|
| `gemini-3-flash-preview` | Google | Gemini 3 Flash Preview with thinking_level support |
| `gpt-5-mini` | OpenAI | GPT-5 Mini with structured output support |
| `ministral-14b-2512` | Mistral | Ministral 14B (2025-12) with structured output |

## Dependencies

```bash
pip install requests python-dotenv pydantic rich google-genai openai mistralai
```
