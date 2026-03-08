# Prompt pour l'extraction d'articles de magazines islamiques (Claude Agent)

## Contexte

Vous analysez un magazine islamique numérisé (PDF) pour en extraire la table des matières complète. Le processus se fait en **deux étapes** : d'abord une extraction **page par page**, puis une **consolidation** des articles fragmentés sur plusieurs pages.

**IMPORTANT — Langue française** : Tout le contenu produit (titres, résumés, notes) DOIT être en français correct avec TOUS les accents et diacritiques : é, è, ê, ë, à, â, ç, ù, û, ü, î, ï, ô, ö. Ne jamais omettre les accents. Exemples : « éducation » (pas « education »), « réflexion » (pas « reflexion »), « société » (pas « societe »), « défis » (pas « defis »), « pèlerinage » (pas « pelerinage »).

**Important** : Les magazines islamiques contiennent fréquemment des termes en arabe. Ces termes doivent être préservés et inclus dans les titres et résumés lorsqu'ils sont correctement retranscrits par l'OCR.

## Étape 1 : Extraction page par page

Lisez le PDF **une page à la fois** avec le Read tool et le paramètre `pages` (ex : `pages: "1"`, puis `pages: "2"`, etc.). Ne lisez jamais plusieurs pages en un seul appel. Pour **chaque page**, identifiez tous les articles présents.

Pour chaque article trouvé sur une page :

1. **Titre exact** tel qu'il apparaît sur la page (respectez la typographie, les majuscules, l'accentuation)
   - **Rubrique** : Si l'article appartient à une rubrique (ex: "Chapelet", "Tribune", "Dossier", "Plume Libre", "Éditorial"), incluez-la au début du titre entre crochets : `[Chapelet] L'apprentissage de la lumière`
   - **Termes arabes** : Incluez les termes en arabe s'ils sont correctement retranscrits (ex: "الحج", "رمضان", "الإسلام")
   - Ignorez uniquement les textes arabes **mal reconnus** par l'OCR (caractères illisibles comme "fllaoyjjÿ", "ÿjjÿoyjj")
   - **Corrigez les espaces OCR** : Supprimez les espaces multiples inutiles entre les lettres d'un même mot (ex: "nouvell e" → "nouvelle", "Mamou ne" → "Mamoune")
   - **Corrigez les erreurs OCR évidentes** : Remplacez les caractères mal reconnus par les bonnes lettres (ex: "arm0es" → "armées", "est-t -H" → "est-il", "PRE SSE" → "PRESSE")

2. **Auteur(s)** s'ils sont mentionnés sur cette page
   - Cherchez les noms d'auteurs en début ou fin d'article (souvent en italique ou après "Par", "De", etc.)
   - Acceptez les mentions collectives comme "La Rédaction", "Équipe de rédaction", "Le CERFI", etc.
   - Si aucun auteur n'est mentionné, laissez vide

3. **Résumé partiel** — 2-3 phrases basées uniquement sur le contenu visible sur cette page
   - Incluez les termes arabes pertinents s'ils sont correctement retranscrits

4. **Indices de continuation** — Notez les mentions comme "suite page X", "à suivre", "(voir p. X)", "fin en page X"

### Règles pour l'étape 1

- **Page de couverture** : Ignorez (première page du magazine).
- **Pages sans article** (publicité, page blanche, etc.) : Ignorez.
- **Distinguez les articles des autres contenus** : éditorial, tribune, interview, reportage, fatwa sont des articles ; les annonces, publicités, brèves courtes (< 100 mots) ne le sont pas.
- **Titre exact** : Ne modifiez pas le titre, sauf pour corriger les espaces OCR et erreurs OCR évidentes.
- **Résumé partiel** : Basez-vous uniquement sur ce qui est visible sur cette page.
- **Articles incomplets** : Si la page contient la fin d'un article commencé précédemment, notez-le avec le titre (s'il est rappelé) ou "Suite de l'article : <titre>".

## Étape 2 : Consolidation

Après avoir extrait les articles page par page, consolidez la liste :

1. **Fusionnez les articles fragmentés** sur plusieurs pages en vous basant sur :
   - Le titre exact (identique ou très proche)
   - La continuité thématique et lexicale
   - Les mentions de continuation ("suite page X", "à suivre", etc.)
   - Les numéros de pages consécutifs ou mentionnés

2. **Pour chaque article consolidé** :
   - Conservez le titre exact complet (incluant les termes arabes)
   - **Consolidez les auteurs** : fusionnez les mentions d'auteurs provenant de différentes pages en évitant les doublons. Un même auteur peut être mentionné différemment (ex: "Ismael Tiendrébéogo" et "Imam Ismaël Tiendrébéogo" → gardez la forme la plus complète)
   - Agrégez toutes les pages (ex: "1-3" ou "1, 3, 5")
   - Produisez un **résumé global** de 1-2 phrases en fusionnant les résumés partiels

3. **Éliminez les doublons** et les pages non-article

## Format de sortie final

Pour chaque article consolidé, produisez exactement ce format :

```
p. <pages> : <Titre> (<Auteur1>, <Auteur2>)
<Résumé concis en 1-2 phrases.>
```

Si aucun auteur n'est identifié, omettez les parenthèses :

```
p. <pages> : <Titre>
<Résumé concis en 1-2 phrases.>
```

Séparez chaque article par une ligne vide. Exemple complet :

```
p. 1-3 : [Éditorial] L'Islam et la modernité (Imam Ismaël Tiendrébéogo)
Réflexion sur la compatibilité entre les valeurs islamiques et les défis de la société contemporaine.

p. 4-5 : Les enjeux de l'éducation coranique au Burkina Faso
Analyse des défis auxquels font face les écoles coraniques, notamment le manque de financement et la reconnaissance officielle.

p. 6-8 : [Dossier] Le Hajj (الحج) : guide pratique (Cheikh Oumar Konaté, La Rédaction)
Guide détaillé des étapes du pèlerinage à La Mecque, incluant les préparatifs, les rites essentiels et les conseils pratiques pour les pèlerins ouest-africains.
```
