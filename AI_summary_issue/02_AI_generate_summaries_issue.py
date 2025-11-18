"""Pipeline d'Extraction d'Articles de Magazines Islamiques (en 2 Étapes)

Ce script implémente un pipeline en deux étapes pour extraire et consolider les articles
d'un magazine islamique à partir de fichiers PDF ou texte OCR page par page.

Modèles supportés:
- Gemini: Pro (étape 1) + Flash (étape 2)
- OpenAI: GPT-5.1 full (étape 1) + GPT-5.1 mini (étape 2)

Étape 1 : Extraction page par page (modèle performant)
- Analyse chaque page individuellement avec le modèle le plus performant
- Identifie les articles présents sur la page
- Extrait le titre exact et génère un résumé bref
- Détecte les indices de continuation

Étape 2 : Consolidation au niveau du magazine (modèle rapide)
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
import sys
import re
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import PyPDF2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from common.llm_provider import (  # noqa: E402
    BaseLLMClient,
    LLMConfig,
    ModelOption,
    build_llm_client,
    get_model_option,
    summary_from_option,
)

try:
    from google import genai
    from google.genai import types, errors
except ImportError:
    genai = None
    types = None
    errors = None

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

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
# Client Initialization
# ------------------------------------------------------------------
def get_model_pair() -> Tuple[ModelOption, ModelOption]:
    """Permet à l'utilisateur de sélectionner la paire de modèles pour le pipeline.
    
    Returns:
        Tuple de (model_step1, model_step2) où step1 est le modèle performant
        et step2 est le modèle rapide.
    """
    print("\n=== Sélection des modèles pour le pipeline ===")
    print("Étape 1 (extraction page par page): Modèle performant")
    print("Étape 2 (consolidation): Modèle rapide\n")
    
    # Définir les paires disponibles
    model_pairs = {
        "gemini": {
            "step1": "gemini-pro",
            "step2": "gemini-flash",
            "name": "Gemini (Pro + Flash)"
        },
        "openai": {
            "step1": "openai-5.1",
            "step2": "openai",
            "name": "OpenAI (GPT-5.1 full + mini)"
        }
    }
    
    print("Paires de modèles disponibles:")
    print("  1) Gemini (Pro + Flash) - Pro pour extraction, Flash pour consolidation")
    print("  2) OpenAI (GPT-5.1 full + mini) - Full pour extraction, mini pour consolidation")
    
    while True:
        choice = input("\nChoisissez la paire de modèles (1 ou 2): ").strip()
        if choice == "1":
            pair_key = "gemini"
            break
        elif choice == "2":
            pair_key = "openai"
            break
        else:
            print("Choix invalide. Veuillez entrer 1 ou 2.")
    
    pair = model_pairs[pair_key]
    step1_option = get_model_option(pair["step1"])
    step2_option = get_model_option(pair["step2"])
    
    logging.info(f"\nModèles sélectionnés: {pair['name']}")
    logging.info(f"  Étape 1: {summary_from_option(step1_option)}")
    logging.info(f"  Étape 2: {summary_from_option(step2_option)}")
    
    return step1_option, step2_option

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
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= RETRY_BACKOFF
                    else:
                        logging.error(f"All {max_retries} attempts failed.")
            
            raise last_exception
        return wrapper
    return decorator

@retry_on_error(max_retries=MAX_RETRIES, delay=RETRY_DELAY)
def generate_with_llm(llm_client: BaseLLMClient, prompt: str) -> Optional[str]:
    """
    Génère une réponse avec le client LLM avec retry automatique.
    
    Args:
        llm_client: Client LLM (OpenAI ou Gemini)
        prompt: Prompt à envoyer
        
    Returns:
        Réponse générée ou None
    """
    if not prompt.strip():
        return None
    
    try:
        response = llm_client.generate(
            system_prompt="",  # Le prompt est complet dans user_prompt
            user_prompt=prompt
        )
        
        if response:
            return response.strip().replace('*', '')
        
        logging.error("Model returned empty response.")
        return None
    except Exception as e:
        logging.error(f"Generation error: {e}")
        raise  # Laisser le décorateur retry gérer

# ------------------------------------------------------------------
# Pipeline Functions
# ------------------------------------------------------------------
def step1_extract_pages(llm_client: BaseLLMClient, pages_text: Dict[int, str], 
                        extraction_prompt: str, output_dir: Path, magazine_id: str,
                        model_name: str) -> Path:
    """
    Étape 1 : Extraction page par page avec le modèle performant.
    Sauvegarde progressive pour permettre la reprise en cas d'interruption.
    
    Args:
        llm_client: Client LLM configuré
        pages_text: Dictionnaire {page_num: text}
        extraction_prompt: Template de prompt pour l'extraction
        output_dir: Répertoire de sortie
        magazine_id: Identifiant du magazine
        model_name: Nom du modèle pour logging
        
    Returns:
        Chemin du fichier consolidé de l'étape 1
    """
    step1_dir = output_dir / "step1_page_extractions"
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    all_extractions = []
    
    logging.info(f"Step 1: Processing {len(pages_text)} pages with {model_name}...")
    
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
            extraction = generate_with_llm(llm_client, prompt)
            
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

def step2_consolidate(llm_client: BaseLLMClient, step1_file: Path, 
                     output_dir: Path, magazine_id: str, model_name: str) -> Path:
    """
    Étape 2 : Consolidation au niveau du magazine avec le modèle rapide.
    
    Args:
        llm_client: Client LLM configuré
        step1_file: Fichier consolidé de l'étape 1
        output_dir: Répertoire de sortie
        magazine_id: Identifiant du magazine
        model_name: Nom du modèle pour logging
        
    Returns:
        Chemin du fichier final consolidé
    """
    logging.info(f"Step 2: Consolidating articles at magazine level with {model_name}...")
    
    # Lire le fichier consolidé de l'étape 1
    with open(step1_file, 'r', encoding='utf-8') as f:
        extracted_content = f.read()
    
    # Charger le prompt de consolidation
    consolidation_prompt = load_consolidation_prompt()
    full_prompt = consolidation_prompt.replace('{extracted_content}', extracted_content)
    
    # Générer la consolidation avec retry automatique
    try:
        consolidated = generate_with_llm(llm_client, full_prompt)
        
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
def process_magazine(model_step1: ModelOption, model_step2: ModelOption,
                    input_path: Path, output_dir: Path, magazine_id: str = None):
    """
    Pipeline complet pour traiter un magazine.
    
    Args:
        model_step1: Option de modèle pour l'étape 1 (performant)
        model_step2: Option de modèle pour l'étape 2 (rapide)
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
    
    # Configurer les clients LLM pour chaque étape
    # Étape 1: Modèle performant avec medium reasoning pour extraction détaillée
    config_step1 = LLMConfig(
        reasoning_effort="medium",
        text_verbosity="medium",
        thinking_budget=500,
        temperature=0.2
    )
    llm_client_step1 = build_llm_client(model_step1, config=config_step1)
    
    # Étape 2: Modèle rapide avec low reasoning pour consolidation simple
    config_step2 = LLMConfig(
        reasoning_effort="low",
        text_verbosity="low",
        thinking_budget=0,
        temperature=0.3
    )
    llm_client_step2 = build_llm_client(model_step2, config=config_step2)
    
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
    
    # Étape 1 : Extraction page par page (modèle performant)
    step1_file = step1_extract_pages(
        llm_client_step1, pages_text, extraction_prompt, output_dir, magazine_id,
        summary_from_option(model_step1)
    )
    
    # Étape 2 : Consolidation (modèle rapide)
    final_file = step2_consolidate(
        llm_client_step2, step1_file, output_dir, magazine_id,
        summary_from_option(model_step2)
    )
    
    logging.info(f"✓ Pipeline complete for magazine {magazine_id}")
    logging.info(f"✓ Final index: {final_file}")
    
    return final_file

# ------------------------------------------------------------------
# User Interaction
# ------------------------------------------------------------------
def get_input_pdfs(script_dir: Path) -> list[Path]:
    """Obtient la liste de tous les PDFs à traiter."""
    # Utiliser le dossier PDF par défaut
    default_pdf_dir = script_dir / "PDF"
    
    if not default_pdf_dir.exists():
        raise FileNotFoundError(f"Le dossier PDF n'existe pas: {default_pdf_dir}")
    
    # Récupérer tous les PDFs
    pdf_files = sorted(list(default_pdf_dir.glob('*.pdf')))
    
    if not pdf_files:
        raise FileNotFoundError(f"Aucun fichier PDF trouvé dans {default_pdf_dir}")
    
    logging.info(f"{len(pdf_files)} fichier(s) PDF trouvé(s) dans {default_pdf_dir}")
    return pdf_files

# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------
def main():
    """Point d'entrée principal du script."""
    try:
        script_dir = Path(__file__).parent
        
        logging.info("=== Pipeline d'Extraction d'Articles de Magazines ===")
        logging.info("Étape 1: Extraction page par page (modèle performant)")
        logging.info("Étape 2: Consolidation au niveau du magazine (modèle rapide)")
        logging.info("")
        
        # Sélection des modèles
        model_step1, model_step2 = get_model_pair()
        
        # Obtenir la liste des PDFs à traiter
        pdf_files = get_input_pdfs(script_dir)
        
        # Traiter chaque PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Traitement du PDF {i}/{len(pdf_files)}: {pdf_path.name}")
            logging.info(f"{'='*60}")
            
            # Utiliser le nom du fichier comme Omeka ID
            omeka_id = pdf_path.stem
            logging.info(f"Omeka ID: {omeka_id}")
            
            # Définir le répertoire de sortie
            output_dir = script_dir / "Magazine_Extractions" / omeka_id
            
            # Exécuter le pipeline pour ce PDF
            process_magazine(model_step1, model_step2, pdf_path, output_dir, omeka_id)
            
            logging.info(f"PDF {i}/{len(pdf_files)} terminé: {pdf_path.name}")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"=== Pipeline terminé avec succès - {len(pdf_files)} magazine(s) traité(s) ===")
        logging.info(f"{'='*60}")
        
    except KeyboardInterrupt:
        logging.info("Processus interrompu par l'utilisateur")
    except Exception as e:
        logging.error(f"Échec du pipeline: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
