# Convention pour le développement

# TL;DR
### Branches
#### Nomenclature
- [ ] `feature/<issue>-court-descriptif` e.g. `feature/221-creer-api-mockup`
### Commit
#### Format:

```
<type>(<périmètre optionnel>): <description>

[corps optionnel]

[pied optionnel]
```
#### Example:
```
fix(operators): race condition in the concatMap operator

Fixes concurrent signals handling leading to an inconsistent state,
especially with the termination signals of the inner and outer
subscribers.

Resolves: #666
```

## Procédure de nomenclature des branches Git

### Objectif

Le but de ce document est de définir une convention de nommage pour les branches Git afin d'améliorer la lisbilité, la cohérence et la traçabilité du code source au sein de l'équipe.

### Règles de nommage

#### Préfixe

Le nom de chaque branche doit commencer par un préfixe indiquant le type de branche :

- `feature/` : pour les branches de développement de nouvelles fonctionnalités
- `fix/` : pour les branches de correction de bogues critiques
- `chore/` : pour les tâches de maintenance et d'amélioration du workflow, sans affecter le code lui-même.

#### Suffixe

Le nom de la branche doit se terminer par un suffixe décrivant succinctement le sujet de la branche. Le suffixe doit être composé de :
- Un numéro de ticket ou d'incident
- Un court descriptif du sujet

### Exemples

- `feature/221-create-api-mockup`
- `feature/test-new-architecture`
- `fix/389-formulaire-contact-disparait`
- `chore/refactor-tests`
- `chore/migrate-db`
- `chore/refactor-code`


## Procédure de nomenclature des commits
Le message d’un commit doit être clair et concis, il doit indiquer ce qui a été modifié et la raison de cette modification. La convention de nommage la plus utilisée est la “[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)“.

### Format des messages
Le format du message est le suivant :

```
<type>(<périmètre optionnel>): <description>

[corps optionnel]

[pied optionnel]
```

#### Le type

Le type du commit décrit l’origine du changement. Il peut prendre différentes valeurs :

- **feat**: Ajout d’une nouvelle fonctionnalité;
- **fix**: Correction d’un bug;
- **build**: Changement lié au système de build ou qui concerne les dépendances (npm, grunt, gulp, webpack, etc.).
- **ci**: Changement concernant le système d’intégration et de déploiement continu (Jenkins, Travis, Ansible, GitlabCI, etc.)
- **docs**: Ajout ou modification de documentation (README, JSdoc, etc.);
- **perf**: Amélioration des performances;
- **refactor**: Modification n’ajoutant pas de fonctionnalités ni de correction de bug (renommage d’une variable, suppression de code redondant, simplification du code, etc.);
- **style**: Changement lié au style du code (indentation, point virgule, etc.);
- **test**: Ajout ou modification de tests;
- **revert**: Annulation d’un précédent commit;
- **chore**: Toute autre modification (mise à jour de version par exemple).

Pour chacun des types, vous pouvez également utiliser un système d’emoji comme [gitmoji](https://gitmoji.carloscuesta.me/).
#### Le périmètre 

Cet élément facultatif indique simplement le contexte du commit. Il s’agit des composants de notre projet, voici une liste non exhaustive :
- controller;
- route;
- middleware;
- view;
- config;
- service;
- etc.
#### Le sujet

Le sujet décrit succinctement la modification. Certaines règles doivent être respectées :
- Le sujet doit faire moins de 50 caractères;
- Les verbes doivent être à l’impératif (add, remove, update, change, etc.);
- La première lettre ne doit pas être en majuscule;
- Le sujet ne doit pas se terminer par un point.

#### Le corps du message

Le corps du message, qui est optionnel, doit contenir les raisons de la modification ou tout simplement donner plus détails sur le commit. Il peut également indiquer les changements importants (breaking changes). 

#### Le footer

Le footer est également facultatif, celui-ci est surtout utilisé pour faire référence aux tickets (issue) de GitLab par exemple que le commit règle ou aux changements importants (breaking changes). 

#### Quelques exemples

Nous allons voir quelques exemples pour mieux comprendre. Les messages des commits qui vont suivre seront en anglais, mais le principe est exactement le même si vous écrivez vos messages de commit en français.

##### Commit simple (sans corps ni footer)

```
feat(controller): add post's controller
docs(route): add swagger documentation
```

##### Commit avec un corps
```
fix(operators): race condition in the concatMap operator

Fixes concurrent signals handling leading to an inconsistent state,
especially with the termination signals of the inner and outer
subscribers.
```

##### Commit complet (corps et footer)
```
fix(operators): race condition in the concatMap operator

Fixes concurrent signals handling leading to an inconsistent state,
especially with the termination signals of the inner and outer
subscribers.

Resolves: #666
```

```
refactor(repository): remove deprecated method

BREAKING CHANGE: findById and findByPrimary methods have been removed.

Resolves: #78
```

```
feat(database): onUrlChange event (popstate/hashchange/polling)  
  
Added new event to $browser:  
- forward popstate event if available  
- forward hashchange event if popstate not available  
- do polling when neither popstate nor hashchange available  
  
BREAKING CHANGE: $browser.onHashChange, which was removed (use onUrlChange instead)
```