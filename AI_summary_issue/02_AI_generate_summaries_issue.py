"""Pipeline d'Extraction d'Articles de Magazines Islamiques (en 2 Étapes)

Ce script implémente un pipeline en deux étapes pour extraire et consolider les articles
d'un magazine islamique à partir de fichiers PDF ou texte OCR page par page.

Étape 1 : Extraction page par page (Gemini 2.5 Pro)
- Analyse chaque page individuellement avec le modèle le plus performant
- Identifie les articles présents sur la page
- Extrait le titre exact et génère un résumé bref
- Détecte les indices de continuation

Étape 2 : Consolidation au niveau du magazine (Gemini 2.5 Flash)
- Fusionne les articles fragmentés sur plusieurs pages
- Élimine les doublons avec le modèle rapide
- Produit un résumé global par article
- Liste toutes les pages associées

Mécanismes de robustesse:
- Retry automatique en cas d'erreur (max 3 tentatives)
- Sauvegarde progressive des résultats
- Reprise possible à partir des fichiers déjà traités

Usage:
    python 02_AI_generate_summaries_issue.py
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import PyPDF2

from google import genai
from google.genai import types, errors

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Modèles Gemini pour chaque étape
GEMINI_MODEL_STEP1 = "gemini-2.5-pro"  # Extraction page par page - modèle performant
GEMINI_MODEL_STEP2 = "gemini-2.5-flash"  # Consolidation - modèle rapide

# Configuration retry
MAX_RETRIES = 3
RETRY_DELAY = 2  # secondes
RETRY_BACKOFF = 2  # multiplicateur pour backoff exponentiel

# ------------------------------------------------------------------
# Prompt Loading
# ------------------------------------------------------------------
def load_extraction_prompt() -> str:
    """Charge le prompt pour l'étape 1 (extraction page par page)."""
    script_dir = Path(__file__).parent
    prompt_file = script_dir / 'summary_prompt_issue.md'
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if '{text}' not in content and '{page_number}' not in content:
            logging.warning("Prompt template missing required placeholders.")
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read prompt template {prompt_file}: {e}")

def load_consolidation_prompt() -> str:
    """Charge le prompt pour l'étape 2 (consolidation)."""
    script_dir = Path(__file__).parent
    prompt_file = script_dir / 'consolidation_prompt_issue.md'
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if '{extracted_content}' not in content:
            logging.warning("Consolidation prompt template missing '{extracted_content}' placeholder.")
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Consolidation prompt template not found: {prompt_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read consolidation prompt template {prompt_file}: {e}")

# ------------------------------------------------------------------
# Client Initialization with Retry
# ------------------------------------------------------------------
def retry_on_error(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """
    Décorateur pour retry automatique en cas d'erreur.
    
    Args:
        max_retries: Nombre maximum de tentatives
        delay: Délai initial entre les tentatives (avec backoff exponentiel)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (errors.APIError, errors.ClientError, ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= RETRY_BACKOFF
                    else:
                        logging.error(f"All {max_retries} attempts failed.")
                except Exception as e:
                    # Pour les autres erreurs, ne pas retry
                    logging.error(f"Non-retryable error: {e}")
                    raise
            
            raise last_exception
        return wrapper
    return decorator

def initialize_gemini_client():
    """Initialise le client Gemini avec ADC ou API key."""
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
    
    logging.info(f"Using Gemini models: Step1={GEMINI_MODEL_STEP1}, Step2={GEMINI_MODEL_STEP2} ({auth_method})")
    return client

# ------------------------------------------------------------------
# PDF Processing
# ------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: Path) -> Dict[int, str]:
    """
    Extrait le texte de chaque page d'un PDF.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        Dictionnaire {numéro_page: texte}
    """
    pages_text = {}
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                pages_text[page_num + 1] = text  # Numérotation à partir de 1
                
        logging.info(f"Extracted text from {total_pages} pages in {pdf_path.name}")
        return pages_text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return {}

def load_txt_files_as_pages(txt_dir: Path) -> Dict[int, str]:
    """
    Charge les fichiers TXT numérotés comme pages.
    Attend des fichiers nommés comme: page_1.txt, page_2.txt, etc.
    Ou simplement tous les .txt dans l'ordre alphabétique.
    
    Args:
        txt_dir: Répertoire contenant les fichiers TXT
        
    Returns:
        Dictionnaire {numéro_page: texte}
    """
    pages_text = {}
    try:
        txt_files = sorted([f for f in txt_dir.glob('*.txt')])
        if not txt_files:
            logging.warning(f"No TXT files found in {txt_dir}")
            return {}
        
        for idx, txt_file in enumerate(txt_files, start=1):
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Essayer d'extraire le numéro de page du nom de fichier
            match = re.search(r'page[_\s]?(\d+)', txt_file.stem, re.IGNORECASE)
            if match:
                page_num = int(match.group(1))
            else:
                page_num = idx  # Utiliser l'ordre alphabétique
            
            pages_text[page_num] = text
        
        logging.info(f"Loaded {len(pages_text)} pages from TXT files in {txt_dir}")
        return pages_text
    except Exception as e:
        logging.error(f"Error loading TXT files from {txt_dir}: {e}")
        return {}

# ------------------------------------------------------------------
# AI Generation Functions with Retry
# ------------------------------------------------------------------
@retry_on_error(max_retries=MAX_RETRIES, delay=RETRY_DELAY)
def generate_with_gemini(client, prompt: str, model: str = GEMINI_MODEL_STEP1) -> Optional[str]:
    """
    Génère une réponse avec Gemini avec retry automatique.
    
    Args:
        client: Client Gemini
        prompt: Prompt à envoyer
        model: Modèle à utiliser (par défaut STEP1)
        
    Returns:
        Réponse générée ou None
    """
    if not prompt.strip():
        return None
    
    try:
        gen_config = types.GenerateContentConfig(temperature=0.2)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=gen_config
        )
        
        if response and hasattr(response, 'text'):
            return response.text.strip().replace('*', '')
        
        logging.error("Unexpected Gemini response format.")
        return None
    except Exception as e:
        logging.error(f"Gemini generation error: {e}")
        raise  # Laisser le décorateur retry gérer

# ------------------------------------------------------------------
# Pipeline Functions
# ------------------------------------------------------------------
def step1_extract_pages(client, pages_text: Dict[int, str], 
                        extraction_prompt: str, output_dir: Path, magazine_id: str) -> Path:
    """
    Étape 1 : Extraction page par page avec Gemini 2.0 Flash Exp.
    Sauvegarde progressive pour permettre la reprise en cas d'interruption.
    
    Args:
        client: Client Gemini
        pages_text: Dictionnaire {page_num: text}
        extraction_prompt: Template de prompt pour l'extraction
        output_dir: Répertoire de sortie
        magazine_id: Identifiant du magazine
        
    Returns:
        Chemin du fichier consolidé de l'étape 1
    """
    step1_dir = output_dir / "step1_page_extractions"
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    all_extractions = []
    
    logging.info(f"Step 1: Processing {len(pages_text)} pages with {GEMINI_MODEL_STEP1}...")
    
    for page_num in tqdm(sorted(pages_text.keys()), desc="Extracting articles per page"):
        page_file = step1_dir / f"page_{page_num:03d}.md"
        
        # Vérifier si la page a déjà été traitée
        if page_file.exists():
            logging.info(f"Page {page_num} already processed, loading from cache...")
            with open(page_file, 'r', encoding='utf-8') as f:
                extraction = f.read()
            all_extractions.append(f"\n{extraction}\n")
            continue
        
        page_text = pages_text[page_num]
        
        # Préparer le prompt avec le texte (pas de numéro de page)
        prompt = extraction_prompt.replace('{text}', page_text)
        
        # Générer l'extraction avec retry automatique
        try:
            extraction = generate_with_gemini(client, prompt, model=GEMINI_MODEL_STEP1)
            
            if extraction:
                # Ajouter le numéro de page au début de la réponse de l'IA
                extraction_with_page = f"## Page : {page_num}\n\n{extraction}"
                all_extractions.append(f"\n{extraction_with_page}\n")
                
                # Sauvegarder immédiatement l'extraction individuelle
                with open(page_file, 'w', encoding='utf-8') as f:
                    f.write(extraction_with_page)
                logging.info(f"✓ Page {page_num} processed and saved")
            else:
                logging.error(f"✗ No extraction generated for page {page_num}")
                # Créer un placeholder pour éviter de bloquer le pipeline
                placeholder = f"## Page : {page_num}\n\nErreur lors du traitement de cette page.\n"
                all_extractions.append(f"\n{placeholder}\n")
                with open(page_file, 'w', encoding='utf-8') as f:
                    f.write(placeholder)
        except Exception as e:
            logging.error(f"✗ Failed to process page {page_num} after retries: {e}")
            placeholder = f"## Page : {page_num}\n\nErreur: {str(e)}\n"
            all_extractions.append(f"\n{placeholder}\n")
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(placeholder)
    
    # Consolider toutes les extractions dans un seul fichier
    consolidated_file = output_dir / f"{magazine_id}_step1_consolidated.md"
    with open(consolidated_file, 'w', encoding='utf-8') as f:
        f.write(f"# Extraction page par page - Magazine {magazine_id}\n\n")
        f.write('\n---\n'.join(all_extractions))
    
    logging.info(f"Step 1 complete. Consolidated file: {consolidated_file}")
    
    # Supprimer les fichiers individuels pour économiser l'espace
    logging.info("Cleaning up individual page files...")
    for page_file in step1_dir.glob('page_*.md'):
        page_file.unlink()
    logging.info(f"Deleted {len(list(step1_dir.glob('page_*.md')))} individual page files")
    
    # Optionnel : supprimer le dossier s'il est vide
    try:
        step1_dir.rmdir()
        logging.info(f"Removed empty directory: {step1_dir}")
    except OSError:
        # Le dossier n'est pas vide, on le garde
        pass
    
    return consolidated_file

def step2_consolidate(client, step1_file: Path, 
                     output_dir: Path, magazine_id: str) -> Path:
    """
    Étape 2 : Consolidation au niveau du magazine avec Gemini 2.0 Flash Exp.
    
    Args:
        client: Client Gemini
        step1_file: Fichier consolidé de l'étape 1
        output_dir: Répertoire de sortie
        magazine_id: Identifiant du magazine
        
    Returns:
        Chemin du fichier final consolidé
    """
    logging.info(f"Step 2: Consolidating articles at magazine level with {GEMINI_MODEL_STEP2}...")
    
    # Lire le fichier consolidé de l'étape 1
    with open(step1_file, 'r', encoding='utf-8') as f:
        extracted_content = f.read()
    
    # Charger le prompt de consolidation
    consolidation_prompt = load_consolidation_prompt()
    full_prompt = consolidation_prompt.replace('{extracted_content}', extracted_content)
    
    # Générer la consolidation avec retry automatique
    try:
        consolidated = generate_with_gemini(client, full_prompt, model=GEMINI_MODEL_STEP2)
        
        if not consolidated:
            logging.error("Failed to generate consolidated output")
            raise RuntimeError("Step 2 consolidation failed - no output generated")
        
        # Sauvegarder le résultat final
        final_file = output_dir / f"{magazine_id}_final_index.md"
        with open(final_file, 'w', encoding='utf-8') as f:
            f.write(consolidated)
        
        logging.info(f"✓ Step 2 complete. Final index: {final_file}")
        return final_file
        
    except Exception as e:
        logging.error(f"✗ Step 2 failed after retries: {e}")
        raise

# ------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------
def process_magazine(client, input_path: Path, output_dir: Path, magazine_id: str = None):
    """
    Pipeline complet pour traiter un magazine avec Gemini.
    
    Args:
        client: Client Gemini
        input_path: Chemin vers le PDF ou répertoire TXT
        output_dir: Répertoire de sortie
        magazine_id: Identifiant du magazine (optionnel)
    """
    # Déterminer l'identifiant du magazine
    if magazine_id is None:
        magazine_id = input_path.stem
    
    logging.info(f"Processing magazine: {magazine_id}")
    logging.info(f"Input: {input_path}")
    logging.info(f"Output: {output_dir}")
    
    # Charger le prompt d'extraction
    extraction_prompt = load_extraction_prompt()
    
    # Extraire le texte page par page
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        logging.info("Input is PDF file - extracting text...")
        pages_text = extract_text_from_pdf(input_path)
    elif input_path.is_dir():
        logging.info("Input is directory - loading TXT files...")
        pages_text = load_txt_files_as_pages(input_path)
    else:
        raise ValueError(f"Invalid input path: {input_path}. Must be PDF file or directory with TXT files.")
    
    if not pages_text:
        raise RuntimeError("No pages extracted from input")
    
    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Étape 1 : Extraction page par page (Gemini 2.0 Flash Exp)
    step1_file = step1_extract_pages(
        client, pages_text, extraction_prompt, output_dir, magazine_id
    )
    
    # Étape 2 : Consolidation (Gemini 2.0 Flash Exp)
    final_file = step2_consolidate(
        client, step1_file, output_dir, magazine_id
    )
    
    logging.info(f"✓ Pipeline complete for magazine {magazine_id}")
    logging.info(f"✓ Final index: {final_file}")
    
    return final_file

# ------------------------------------------------------------------
# User Interaction
# ------------------------------------------------------------------
def get_input_path(script_dir: Path) -> Path:
    """Obtient le chemin d'entrée (PDF ou répertoire TXT)."""
    # Utiliser le dossier PDF par défaut
    default_pdf_dir = script_dir / "PDF"
    
    if default_pdf_dir.exists() and default_pdf_dir.is_dir():
        # Vérifier s'il y a des PDFs dans le dossier
        pdf_files = list(default_pdf_dir.glob('*.pdf'))
        if pdf_files:
            logging.info(f"Dossier PDF trouvé avec {len(pdf_files)} fichier(s)")
            # Si un seul PDF, retourner directement le fichier
            if len(pdf_files) == 1:
                return pdf_files[0]
            # Sinon retourner le dossier
            return default_pdf_dir
    
    # Sinon demander le chemin
    while True:
        path_str = input("Entrez le chemin vers le PDF ou le répertoire TXT (ou Entrée pour ./PDF): ").strip()
        
        # Si vide, utiliser le dossier PDF par défaut
        if not path_str:
            if default_pdf_dir.exists():
                return default_pdf_dir
            else:
                print(f"Le dossier PDF par défaut n'existe pas: {default_pdf_dir}")
                continue
        
        # Enlever les guillemets si présents
        path_str = path_str.strip('"').strip("'")
        path = Path(path_str)
        
        if path.exists():
            if path.is_file() and path.suffix.lower() == '.pdf':
                return path
            elif path.is_dir():
                return path
            else:
                print("Entrée invalide. Doit être un fichier PDF ou un répertoire contenant des fichiers TXT.")
        else:
            print(f"Le chemin n'existe pas: {path}")

def get_magazine_id() -> str:
    """Demande l'identifiant du magazine."""
    magazine_id = input("Entrez l'identifiant du magazine (ou Entrée pour utiliser le nom du fichier): ").strip()
    return magazine_id if magazine_id else None

# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------
def main():
    """Point d'entrée principal du script."""
    try:
        script_dir = Path(__file__).parent
        
        logging.info("=== Pipeline d'Extraction d'Articles de Magazines ===")
        logging.info(f"Étape 1: Extraction page par page ({GEMINI_MODEL_STEP1})")
        logging.info(f"Étape 2: Consolidation au niveau du magazine ({GEMINI_MODEL_STEP2})")
        logging.info("")
        
        # Initialisation du client Gemini
        client = initialize_gemini_client()
        
        # Obtenir le chemin d'entrée
        input_path = get_input_path(script_dir)
        
        # Utiliser le nom du fichier/dossier comme Omeka ID
        omeka_id = input_path.stem
        logging.info(f"Omeka ID: {omeka_id}")
        
        # Définir le répertoire de sortie
        output_dir = script_dir / "Magazine_Extractions" / omeka_id
        
        # Exécuter le pipeline
        process_magazine(client, input_path, output_dir, omeka_id)
        
        logging.info("=== Pipeline terminé avec succès ===")
        
    except KeyboardInterrupt:
        logging.info("Processus interrompu par l'utilisateur")
    except Exception as e:
        logging.error(f"Échec du pipeline: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
