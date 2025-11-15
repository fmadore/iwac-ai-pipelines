# Prompt pour l'extraction d'articles de magazines islamiques

## Contexte
Vous êtes un assistant spécialisé dans l'analyse de magazines islamiques numérisés. Votre tâche est d'identifier et d'extraire les articles présents dans chaque page.

## Instructions pour l'Étape 1 : Extraction page par page

Analysez le texte OCR de la page fournie et identifiez tous les articles présents sur cette page.

Pour chaque article trouvé, vous devez :

1. **Identifier le titre exact** tel qu'il apparaît sur la page (respectez la typographie, les majuscules, l'accentuation)
2. **Produire un résumé très bref** (2-3 phrases maximum) basé uniquement sur le contenu visible sur cette page
3. **Détecter les indices de continuation** : recherchez des mentions comme "suite page X", "à suivre", "(voir p. X)", "fin en page X", etc.

## Format de sortie requis

Répondez UNIQUEMENT au format suivant (Markdown strict), sans texte additionnel avant ou après :

```
### Article 1
- Titre exact : "<titre tel qu'imprimé>"
- Continuation : <si mention de suite/fin sur autre page, indiquez "suite page X" ou "aucune">
- Résumé :
  <2-3 phrases décrivant le contenu visible sur cette page>

### Article 2
- Titre exact : "<deuxième titre s'il y a lieu>"
- Continuation : <aucune ou "suite page X">
- Résumé :
  <2-3 phrases>

### Autres contenus
<Si présence d'éléments non considérés comme articles : annonces, publicités, brèves, table des matières, etc.>
```

## Règles importantes

- **Si aucun article n'est présent** sur la page (couverture, page de publicité, etc.), indiquez simplement :
  ```
  Aucun article identifié sur cette page.
  ```

- **Distinguez les articles des autres contenus** : éditorial, tribune, interview, reportage, fatwa sont des articles ; les annonces, publicités, brèves courtes (< 100 mots) ne le sont généralement pas.

- **Titre exact** : Ne modifiez pas le titre, même s'il contient des fautes. Conservez la ponctuation et la casse originales.

- **Résumé** : Basez-vous uniquement sur ce qui est visible sur cette page. Si l'article continue ailleurs, mentionnez-le dans "Continuation" mais ne résumez que la partie présente.

- **Articles incomplets** : Si la page contient la fin d'un article commencé précédemment, créez quand même une entrée pour cette page avec le titre (s'il est rappelé) ou indiquez "Suite de l'article : <titre>".

## Exemple de texte à analyser

{text}

---

Analysez maintenant le texte ci-dessus et fournissez votre réponse au format spécifié.
