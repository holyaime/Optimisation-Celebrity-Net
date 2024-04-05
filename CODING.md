# <font color="#953734">Convention pour le développement  </font>

# <font color="#2DC26B">TL;DR  </font>
### <font color="#c0504d">Branches</font>
#### <font color="#366092">Nomenclature</font>
- [ ] `feature/<issue>-court-descriptif` e.g. `feature/221-creer-api-mockup`
### <font color="#c0504d">Commit</font>
#### <font color="#366092">Format</font>:

```
<type>(<périmètre optionnel>): <description>

[corps optionnel]

[pied optionnel]
```
#### <font color="#366092">Example</font>:
```
fix(operators): race condition in the concatMap operator

Fixes concurrent signals handling leading to an inconsistent state,
especially with the termination signals of the inner and outer
subscribers.

Resolves: #666
```

## <font color="#e36c09">Procédure de nomenclature des branches Git </font>

### <font color="#fac08f">Objectif  </font>

Le but de ce document est de définir une convention de nommage pour les branches Git afin d'améliorer la lisbilité, la cohérence et la traçabilité du code source au sein de l'équipe.

### <font color="#fac08f">Règles de nommage </font>

#### <font color="#366092">Préfixe</font>

Le nom de chaque branche doit commencer par un préfixe indiquant le type de branche :

- `feature/` : pour les branches de développement de nouvelles fonctionnalités
- `fix/` : pour les branches de correction de bogues critiques
- `chore/` : pour les tâches de maintenance et d'amélioration du workflow, sans affecter le code lui-même.

#### <font color="#366092">Suffixe</font>

Le nom de la branche doit se terminer par un suffixe décrivant succinctement le sujet de la branche. Le suffixe doit être composé de :
- Un numéro de ticket ou d'incident
- Un court descriptif du sujet

### <font color="#fac08f">Exemples</font>

- `feature/221-create-api-mockup`
- `feature/test-new-architecture`
- `fix/389-formulaire-contact-disparait`
- `chore/refactor-tests`
- `chore/migrate-db`
- `chore/refactor-code`


## <font color="#e36c09">Procédure de nomenclature des commits</font>
Le message d’un commit doit être clair et concis, il doit indiquer ce qui a été modifié et la raison de cette modification. La convention de nommage la plus utilisée est la “[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)“.

### <font color="#fac08f">Format des messages </font>
Le format du message est le suivant :

```
<type>(<périmètre optionnel>): <description>

[corps optionnel]

[pied optionnel]
```

#### <font color="#366092">Le type </font>
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
#### <font color="#366092">Le périmètre </font>
Cet élément facultatif indique simplement le contexte du commit. Il s’agit des composants de notre projet, voici une liste non exhaustive :
- **`app`**: modifications liées à la fonctionnalité globale de l'application.
- **`config`**: modifications apportées aux fichiers de configuration.
- **`deps`**: Mises à jour des dépendances ou des bibliothèques externes.
- **`infrastructure`**: modifications apportées à la configuration de l'infrastructure ou du déploiement.
- **`linting`**: peluchage du code ou modifications de formatage.
- **`ui`**: modifications liées à l'interface utilisateur (UI).
- **`utils`**: modifications apportées aux fonctions ou classes utilitaires.
- **`auth`**: modifications liées à l'authentification.
- **`search`**: modifications de la fonctionnalité de recherche.
- **`dashboard`**: modifications liées au tableau de bord.
- **`api`**: modifications liées à l'API (peuvent être ventilées par points de terminaison d'API spécifiques).
- **`database`**: modifications du schéma de base de données ou de migration.
- **`service/users`**: Modifications apportées au service de gestion des utilisateurs.
- **`security`**: changements liés aux failles de sécurité.
- **``compliance``**: modifications pour se conformer à la réglementation.
- **`project/setup`**: Mise en place de l'environnement du projet.

##### PERIMETRE DATA [A MODIFIER]
**Data-Related Scopes:**
- **`data/ingestion`** : Scripts ou code liés à la collecte ou au scraping de données.
- **`data/transformation`**: activités de nettoyage, de prétraitement et de transformation des données.
- **`data/loading`**: sauvegarde des données.
- **`data/activation`**: ajouter les données a des outils metiers
- **`data/exploration`**: Analyse exploratoire des données (EDA) et ingénierie des fonctionnalités.
- **`data/quality`**: les méthodes mises en œuvre pour monitorer la qualite des données.

**Model-Related Scopes:**
- **`model/architecture`**: définition et construction de l'architecture du modèle (par exemple, CNN, RNN, Transformer).
- **`model/training`**: code pour entraîner le modèle, y compris le réglage des hyperparamètres.
- **`model/evaluation`**: code permettant d'évaluer les performances du modèle sur des métriques telles que l'exactitude, la précision, le rappel et le score F1.
- **`model/deployment`**: Techniques et scripts pour déployer le modèle en production.
- **`model/loss_function`**: Implémentation et configuration de fonctions de perte pour la formation.
- **`model/optimizer`**: Sélection et configuration des optimiseurs pour la formation des modèles.
- **`model/data`**: Tout traitement de données durant la modélisation.


#### <font color="#366092">Le sujet  </font>
Le sujet décrit succinctement la modification. Certaines règles doivent être respectées :
- Le sujet doit faire au plus 72 caractères;
- Les verbes doivent être à l’impératif (add, remove, update, change, etc.);
- La première lettre ne doit pas être en majuscule;
- Le sujet ne doit pas se terminer par un point.

#### <font color="#366092">Le corps du message  </font>
Le corps du message, qui est optionnel, doit contenir les raisons de la modification ou tout simplement donner plus détails sur le commit. Il peut également indiquer les changements importants (breaking changes).

#### <font color="#366092">Le footer </font>
Le footer est également facultatif, celui-ci est surtout utilisé pour faire référence aux tickets (issue) de GitLab par exemple que le commit règle ou aux changements importants (breaking changes).

#### <font color="#366092">Quelques exemples  </font>
Nous allons voir quelques exemples pour mieux comprendre. Les messages des commits qui vont suivre seront en anglais, mais le principe est exactement le même si vous écrivez vos messages de commit en français.

##### <font color="#8db3e2">Commit simple (sans corps ni footer)  </font>
```
feat(controller): add post's controller
docs(route): add swagger documentation
```

##### <font color="#8db3e2">Commit avec un corps  </font>
```
fix(operators): race condition in the concatMap operator

Fixes concurrent signals handling leading to an inconsistent state,
especially with the termination signals of the inner and outer
subscribers.
```

##### <font color="#8db3e2">Commit complet (corps et footer)  </font>
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

Added new event to $browser:  - forward popstate event if available  - forward hashchange event if popstate not available  - do polling when neither popstate nor hashchange available

BREAKING CHANGE: $browser.onHashChange, which was removed (use onUrlChange instead)
```

Made with 🧡 by <a href="datakori.tech">Datakori</a>
