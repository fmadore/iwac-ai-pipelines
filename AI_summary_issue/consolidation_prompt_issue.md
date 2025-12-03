# Prompt de Consolidation d'Articles de Magazine

## Contexte
Vous avez reçu une liste d'articles extraits page par page d'un magazine islamique.
Certains articles peuvent être fragmentés sur plusieurs pages. Votre tâche est de consolider
cette liste en éliminant les doublons et en regroupant les articles fragmentés.

**Important** : Les magazines islamiques contiennent fréquemment des termes en arabe dans les titres et contenus. Préservez ces termes arabes correctement retranscrits dans les titres exacts et résumés consolidés.

## Instructions

Analysez le document consolidé fourni et :

1. **Ignorez les pages non-article** :
   - Pages de couverture
   - Pages sans articles
   - Publicités et annonces

2. **Identifiez les articles répétés ou fragmentés** sur plusieurs pages en vous basant sur :
   - Le titre exact (identique ou très proche)
   - La continuité thématique et lexicale
   - Les mentions de continuation ("suite page X", "à suivre", etc.)
   - Les numéros de pages consécutifs ou mentionnés

3. **Fusionnez les occurrences** en un seul enregistrement par article :
   - Conservez le titre exact complet de l'article (incluant les termes arabes s'ils sont présents)
   - **Consolidez les auteurs** : Fusionnez les mentions d'auteurs provenant de différentes pages en évitant les doublons. Attention : un même auteur peut être mentionné différemment sur plusieurs pages (ex: "Ismael Tiendrébéogo" et "Imam Ismaël Tiendrébéogo" sont la même personne → gardez la forme la plus complète). Comparez les noms et ne gardez qu'une seule entrée par personne.
   - **Corrigez les erreurs OCR** dans les titres : supprimez les espaces multiples inutiles (ex: "nouvell e" → "nouvelle"), corrigez les caractères mal reconnus (ex: "arm0es" → "armées", "est-t -H" → "est-il", "PRE SSE" → "PRESSE")
   - Agrégez toutes les pages (ex: 1–3 ou 1, 3, 5)
   - Produisez un résumé global (4-6 phrases) en fusionnant les résumés partiels
   - Incluez les termes arabes pertinents dans le résumé consolidé pour enrichir le contexte

4. **Produisez la liste finale** sans doublons et sans pages non-article

## Informations à fournir pour chaque article consolidé

Pour chaque article unique identifié, fournissez :
- **titre** : Le titre exact complet de l'article (corrigé des erreurs OCR)
- **auteurs** : Liste des auteurs (ex: ["Jean Dupont", "La Rédaction"]) ou null si non mentionné
- **pages** : Numéros de pages où l'article apparaît (ex: "1-3" ou "1, 3, 5")
- **resume** : Résumé global de 4-6 phrases fusionnant les résumés partiels

## Règles de consolidation

- **Titres identiques ou très proches** : fusionnez (variations mineures de ponctuation, majuscules)
- **Continuité thématique évidente** : fusionnez même si le titre n'est pas répété sur toutes les pages
- **Pages consécutives avec le même sujet** : probablement le même article
- **Mentions explicites** : "suite page X" indique clairement la fragmentation
- **Résumés** : évitez les répétitions, synthétisez l'ensemble du contenu

## Document à consolider

{extracted_content}

---

Analysez le document ci-dessus et fournissez l'index consolidé au format spécifié.
