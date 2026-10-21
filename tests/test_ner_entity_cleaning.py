"""Casing rules applied to every entity AI_NER/01 extracts."""

import importlib.util
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ner = _load("ner_cleaning", "AI_NER/01_NER_AI.py")


@pytest.mark.parametrize("raw, expected", [
    # Acronyms and inner capitals are kept as the model wrote them.
    ("UEMOA", "UEMOA"),
    ("IBK", "IBK"),
    ("RFI", "RFI"),
    ("RTB Télé", "RTB Télé"),
    ("G5 Sahel", "G5 Sahel"),
    ("McCarthy", "McCarthy"),
    # Each piece of a hyphenated word is capitalised.
    ("Joseph Ki-zerbo", "Joseph Ki-Zerbo"),
    ("bobo-dioulasso", "Bobo-Dioulasso"),
    ("Centre-nord", "Centre-Nord"),
    ("aïd el-fitr", "Aïd el-Fitr"),
    # Elided articles stay lowercase inside a name, capital at its start.
    ("Côte d'Ivoire", "Côte d'Ivoire"),
    ("côte d'ivoire", "Côte d'Ivoire"),
    ("Agence D'Information du Burkina", "Agence d'Information du Burkina"),
    ("l'observateur paalga", "L'Observateur Paalga"),
    ("N'Djamena", "N'Djamena"),
    ("Aujourd'hui au Faso", "Aujourd'hui au Faso"),
    ("bid'a", "Bid'a"),
    # Particles are lowercased after the first word only.
    ("Conseil Supérieur Des Imams", "Conseil Supérieur des Imams"),
    ("le Pays", "Le Pays"),
    ("ibn Abbas", "Ibn Abbas"),
    # Lowercase input is still capitalised word by word.
    ("dialogue interreligieux", "Dialogue Interreligieux"),
    ("communauté musulmane du burkina faso", "Communauté Musulmane du Burkina Faso"),
    # A name entirely in capitals is a headline, not a run of acronyms.
    ("ABDOULAYE TRAORÉ", "Abdoulaye Traoré"),
    ("  ", ""),
    ("", ""),
    # A capitalised piece followed by capitals is one surname, not an acronym.
    ("Université Ouaga 01 Joseph Ki-ZERBO", "Université Ouaga 01 Joseph Ki-Zerbo"),
    ("CNDP-BF", "CNDP-BF"),
])
def test_clean_entity(raw, expected):
    assert ner.clean_entity(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    # Surnames in capitals, the press convention, are not acronyms...
    ("Abdoulaye TRAORÉ", "Abdoulaye Traoré"),
    ("Ousmane SAWADOGO", "Ousmane Sawadogo"),
    # ...but initials and short nicknames are.
    ("IBK", "IBK"),
    ("Salia K. Traoré", "Salia K. Traoré"),
])
def test_clean_entity_person(raw, expected):
    assert ner.clean_entity(raw, person=True) == expected


def test_perform_ner_cleans_persons_as_names():
    class Client:
        def generate_structured(self, _system, _user, schema):
            return schema(persons=["Abdoulaye TRAORÉ"], organizations=["UEMOA"],
                          locations=[], subjects=[])

    result = ner.perform_ner(Client(), "texte")
    assert result.persons == ["Abdoulaye Traoré"]
    assert result.organizations == ["UEMOA"]


def test_deduplication_ignores_case_after_cleaning():
    assert ner.deduplicate_entities(["UEMOA", "uemoa", "Uemoa", "Ki-Zerbo", "ki-zerbo"]) == [
        "UEMOA", "Ki-Zerbo",
    ]
