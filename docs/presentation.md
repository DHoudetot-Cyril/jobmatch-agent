# JobMatch Agent : le projet expliqué simplement

> Ce document présente le projet sans jargon, pour quelqu'un qui connaît l'informatique mais
> pas forcément l'IA. Pour les détails techniques, voir le [README](../README.md) et le dossier
> [`docs/`](README.md).

## 1. Le problème résolu

On a **une offre d'emploi** et **un CV**. Le système répond à trois questions :

1. **Ce CV correspond-il à cette offre ?** Il donne un score sur 100, avec une justification.
2. **Qu'est-ce qui manque ?** Il liste les compétences absentes, puis un **agent conseiller de carrière** cherche, avec ses propres outils, comment les acquérir : formations vérifiées, compétences proches déjà présentes dans le CV, actions concrètes.
3. **Peut-il écrire une lettre de motivation ?** Il la rédige, la fait relire, puis **demande l'accord d'un humain** avant de la finaliser.

Tout tourne **en local**, sans aucun service cloud. Aucune donnée ne quitte la machine.

## 2. Vue d'ensemble

```
                          ┌──────────────────────────────────────────────┐
   UTILISATEUR            │                  LA MACHINE                  │
   (navigateur ou curl)   │                                              │
        │                 │   ┌──────────────── DOCKER ───────────────┐  │
        │ 1. envoie CV    │   │                                       │  │
        │    + offre      │   │  ┌─────────────┐     ┌─────────────┐  │  │
        └─────────────────┼───┼─▶│   API       │────▶│  Postgres   │  │  │
                          │   │  │  (FastAPI)  │     │ (sauvegarde │  │  │
        ┌─────────────────┼───┼──│  port 8000  │     │  de l'état) │  │  │
        │ 4. résultat     │   │  └──────┬──────┘     └─────────────┘  │  │
        │  + lettre       │   │         │ lance                       │  │
        │  à valider      │   │         ▼                             │  │
        │                 │   │  ┌──────────────────┐                 │  │
        │ 5. "OK" ou      │   │  │  LE GRAPHE       │  questions      │  │
        │  "modifie ça"   │   │  │  D'AGENTS        │  oui/non        │  │
        └─────────────────┼───┼─▶│  (LangGraph)     │───────────┐     │  │
                          │   │  └────────┬─────────┘           ▼     │  │
                          │   │           │              ┌───────────┐│  │
                          │   │           │              │  jevos    ││  │
                          │   │           │              │ (décision ││  │
                          │   │           │              │  rapide,  ││  │
                          │   │           │              │  CPU)     ││  │
                          │   │           │              └───────────┘│  │
                          │   │  OBSERVATION :                        │  │
                          │   │  ┌──────────┐ ┌───────────┐ ┌───────┐ │  │
                          │   │  │ Langfuse │ │Prometheus │→│Grafana│ │  │
                          │   │  │ (traces) │ │(métriques)│ │(écran)│ │  │
                          │   │  └──────────┘ └───────────┘ └───────┘ │  │
                          │   └───────────────────────────────────────┘  │
                          │               │ 2-3. « réfléchis »           │
                          │               ▼                              │
                          │   ┌──────────────────────────────┐           │
                          │   │  Qwen 27B (le "cerveau")     │           │
                          │   │  llama.cpp sur le GPU        │           │
                          │   └──────────────────────────────┘           │
                          └──────────────────────────────────────────────┘
```

On peut ranger ces briques en trois familles :

- **Le moteur** : l'API, le graphe d'agents et Qwen. Ce sont eux qui font le travail.
- **Le contrôleur** : jevos, qui vérifie certaines réponses de Qwen.
- **Le tableau de bord** : Langfuse, Prometheus et Grafana, qui permettent de voir ce qui se passe.

## 3. Le parcours d'une analyse

Exemple tiré du dépôt : un **CV de Data Engineer** face à une **offre de Data Engineer confirmé**
(fichiers dans `data/samples/`).

```
  ┌────────────┐
  │ SUPERVISEUR│  ← le chef d'orchestre : il regarde ce qui est déjà fait
  └─────┬──────┘    et décide quelle équipe travaille ensuite
        │
        ▼
  ╔═══════════════════ ÉQUIPE ANALYSE ═══════════════════╗
  ║  parse_offer            parse_cv       (en parallèle)║
  ║  "Poste: Data Eng.,     "4 ans d'XP,                 ║
  ║   Python requis,         Python, Spark,              ║
  ║   Kafka requis..."       Airflow..."                 ║
  ╚══════════════════════════════════════════════════════╝
        │  (retour au superviseur)
        ▼
  ╔═══════════════════ ÉQUIPE MATCHING ══════════════════╗
  ║  score_match     → Qwen note chaque compétence :     ║
  ║                    Python = maîtrisé, Kafka = absent…║
  ║  verify_skills   → jevos vérifie : "le CV prouve-t-il║
  ║                    Kafka ?" → 3 % → confirmé absent  ║
  ║  analyze_gaps    → conseils : "faire un projet Kafka"║
  ║  ➜ SCORE /100 calculé par le CODE, pas par l'IA      ║
  ╚══════════════════════════════════════════════════════╝
        │
        ▼
  ╔═════════════ ÉQUIPE CONSEIL DE CARRIÈRE ═════════════╗
  ║  career_advisor ⇄ career_tools       (agent autonome)║
  ║  Qwen choisit lui-même ses outils, par exemple :     ║
  ║   « Kafka dans ESCO ? » → absent → il élargit        ║
  ║   « le CV montre-t-il SQL ? » → jevos vérifie        ║
  ║   « formations Kafka ? » → catalogue local vérifié   ║
  ║  career_plan → un plan par lacune (formations,       ║
  ║                actions), VÉRIFIÉ par le code         ║
  ╚══════════════════════════════════════════════════════╝
        │
        ▼
  ╔═══════════════════ ÉQUIPE RÉDACTION ═════════════════╗
  ║  write_letter  → Qwen écrit la lettre                ║
  ║       ▲                 │                            ║
  ║       │ corrige         ▼                            ║
  ║       └──────── review_letter → Qwen relit et cherche║
  ║                        │       les inventions        ║
  ║                 verify_claims → jevos vérifie chaque ║
  ║                        │        affirmation douteuse ║
  ║           (2 allers-retours maximum)                 ║
  ╚══════════════════════════════════════════════════════╝
        │
        ▼
  ┌──────────────┐
  │ human_review │  ⏸  LE GRAPHE SE MET EN PAUSE ET ATTEND L'HUMAIN
  └──────┬───────┘
         ├── "OK"            → terminé ✅
         ├── "plus court"    → retour à l'équipe Rédaction
         └── "non"           → rejeté ❌
```

Une analyse complète prend **environ 8 minutes** sur un seul GPU grand public : le modèle a
27 milliards de paramètres, l'agent de carrière et la boucle de rédaction et de relecture font
chacun plusieurs appels au LLM. Lors du seul essai mesuré avec l'agent de carrière (exemple
`data/samples`, 4 novembre 2026), l'analyse a pris 491 s, dont environ 167 s pour l'agent ;
les essais précédents sans lui prenaient 320 à 360 s.

## 4. Les briques, une par une

### Qwen et llama.cpp, le « cerveau »

- **Qwen3.8-27B** est un LLM (un modèle de langage, comme ChatGPT) **open source**, avec 27 milliards de paramètres.
- **llama.cpp** est le programme qui le fait tourner sur le GPU. Il expose une API **compatible avec celle d'OpenAI**, donc tout outil prévu pour ChatGPT peut l'utiliser sans modification.
- **Ses capacités** : il produit du **JSON structuré** (des données propres, pas du texte libre) et il **lit des images**, ce qui permet de traiter les CV scannés.
- **Pourquoi il n'est pas dans Docker** : faire passer une carte AMD dans un conteneur est compliqué, et ça n'apportait rien ici ([ADR 0001](adr/0001-llama-cpp-on-host.md)).

### LangGraph, l'organisation du travail

C'est la bibliothèque qui organise les agents. Un **agent**, ici, est le plus souvent une étape
qui appelle le LLM avec une consigne précise (le conseiller de carrière, lui, va plus loin : voir
plus bas). On représente le travail comme un **graphe** : des
cases (les agents) reliées par des flèches (l'ordre d'exécution).

- **L'état** : un grand dossier partagé (l'offre, le CV, le score, la lettre…). Chaque agent y lit et y écrit.
- **Le superviseur** : il décide de la suite **avec de simples règles en code**, pas avec l'IA. Comme les étapes sont toujours les mêmes, demander à un LLM de choisir la suivante serait plus lent, plus cher et moins fiable ([ADR 0002](adr/0002-deterministic-supervisor.md)).
- **Les sous-graphes** : chaque équipe est un petit graphe à elle seule. C'est ce qui rend l'ensemble **hiérarchique**.
- **L'interruption (`interrupt`)** : le graphe peut se mettre en **pause** pour attendre un humain. Son état est sauvegardé dans **Postgres** : une analyse en pause reprend même après un redémarrage du serveur. En revanche, une analyse en cours au moment du redémarrage est marquée en échec ([ADR 0004](adr/0004-hitl-interrupt-postgres.md)).

### L'agent conseiller de carrière : un vrai agent autonome

Partout ailleurs, c'est le **code** qui fixe l'ordre des étapes, et le LLM ne fait que remplir
chaque case. Le conseiller de carrière fonctionne autrement : on lui donne la liste des
compétences manquantes et **trois outils**, et c'est **lui qui décide** quel outil appeler, avec
quels mots, dans quel ordre, et quand il a fini. C'est le schéma **ReAct** (« raisonner puis
agir ») : il réfléchit, appelle des outils, lit les résultats, puis recommence.

```
        ┌──────────────────────────────────────────┐
  ┌────▶│ career_advisor : Qwen + 3 outils         │
  │     │ « Quel outil j'appelle maintenant ? »    │
  │     └──────┬────────────────────────┬──────────┘
  │            │ il appelle des outils  │ « j'en sais assez »
  │            ▼                        │ OU limite atteinte
  │     ┌─────────────────────────────┐ │ (6 tours ou 120 s)
  │     │ career_tools (en parallèle) │ │
  │     │ 1. search_skill_taxonomy    │ │
  │     │    → référentiel ESCO       │ │
  │     │ 2. check_cv_evidence        │ │
  │     │    → jevos : dans le CV ?   │ │
  │     │ 3. search_learning_resources│ │
  │     │    → catalogue de formations│ │
  │     └──────┬──────────────────────┘ │
  │ résultats  │                        │
  └────────────┘                        ▼
                    ┌──────────────────────────────────┐
                    │ career_plan : un plan par lacune,│
                    │ puis VÉRIFIÉ par le code         │
                    └──────────────────────────────────┘
```

Les trois outils tournent **en local**, sans Internet :

| Outil | Ce qu'il fait |
|---|---|
| `search_skill_taxonomy` | Cherche la compétence dans **ESCO**, le référentiel européen des compétences, et renvoie les compétences proches. Une copie locale de 272 concepts est livrée avec le projet. |
| `check_cv_evidence` | Demande à **jevos** : « le CV montre-t-il cette compétence ? ». Sert à repérer les compétences **transférables** (par exemple, SQL quand il manque dbt). |
| `search_learning_resources` | Cherche dans un **catalogue de formations** fait à la main : 107 ressources pour 76 compétences, toutes les adresses vérifiées le 4 novembre 2026. |

**Ce que ça donne sur l'exemple** (un seul essai réel, pas une évaluation) : l'agent a fait
3 tours et 19 appels d'outils. Au 1er tour, il a cherché les 4 lacunes dans ESCO ; au 2e, il a
élargi certaines recherches et vérifié dans le CV des compétences proches (PySpark, SQL, Docker,
AWS S3, CI/CD…) ; au 3e, il a cherché des formations pour les 4 lacunes. Résultat : AWS Glue a
été rattaché à « cloud technologies », dbt à « data warehouse », Terraform à « ICT
infrastructure », et Kafka à rien (ESCO ne le connaît pas). Chaque lacune a reçu 1 à 3 formations
du catalogue et 3 actions, en français, et aucune adresse inventée n'a dû être retirée.

#### Workflow ou agent : quelle différence ?

| | **Workflow** (le reste du projet) | **Agent autonome** (le conseiller de carrière) |
|---|---|---|
| Qui décide de l'étape suivante ? | Le code, avec des règles fixes | Le LLM, selon ce que les outils lui renvoient |
| Prévisibilité | Totale : mêmes étapes, même ordre, à chaque fois | Partielle : le nombre et l'ordre des appels changent d'une fois sur l'autre |
| Quand l'utiliser ? | Quand les étapes sont connues à l'avance | Quand la bonne étape dépend de ce qu'on vient de découvrir |

**Pourquoi les deux cohabitent ici.** Analyser, noter, rédiger, faire valider : ces étapes sont
toujours les mêmes, un workflow suffit et reste rapide, fiable et testable
([ADR 0002](adr/0002-deterministic-supervisor.md)). Chercher comment combler une lacune, en
revanche, ne se planifie pas à l'avance : si ESCO ne connaît pas Kafka, il faut élargir la
recherche ou passer à autre chose ; les compétences proches à vérifier dans le CV ne sont connues
qu'après la première recherche. Écrire toutes ces branches à la main serait lourd, alors on laisse
l'agent choisir, **à l'intérieur de limites fixées par le code**
([ADR 0009](adr/0009-workflow-plus-autonomous-agent.md)).

#### Les garde-fous

Un agent autonome peut tourner en rond ou inventer. Le code l'en empêche :

- **Temps et tours limités** : 6 tours d'outils au plus et 120 secondes pour la boucle. Au-delà,
  le code force la réponse finale. Si même celle-ci échoue, un plan de secours est construit à
  partir des résultats des outils.
- **Pas d'adresse inventée** : toute formation (ou tout concept ESCO) qu'aucun outil n'a renvoyé
  pendant l'analyse est retiré du plan, et compté dans Grafana.
- **Compétence transférable = preuve dans le CV** : elle n'est gardée que si jevos a confirmé
  qu'elle apparaît dans le CV.
- **Le CV est fourni par le code** : l'agent ne le recopie jamais lui-même, il ne peut donc ni le
  modifier ni en inventer un morceau.
- **Interrupteur** : `CAREER_ADVISOR_ENABLED=false` retire l'agent, et le projet retrouve
  exactement son fonctionnement précédent.

#### Ses limites

- **ESCO connaît mal les outils récents** : ni Kafka, ni Kubernetes, ni Docker, ni Terraform, ni
  dbt, ni Airflow, ni Spark, ni MLflow, ni Linux. Les rattachements sont donc souvent larges (dbt →
  « data warehouse »), et les compétences transférables sont parfois des intitulés ESCO un peu
  génériques.
- **Il coûte du temps** : environ 167 s mesurées une fois, dont environ 80 s pour la réponse
  finale structurée, avec la « réflexion » de Qwen activée. La désactiver pour cet appel devrait
  aller plus vite, mais ce n'est pas encore possible dans le code (`get_llm()`).
- **Il n'est pas encore évalué** : la suite DeepEval n'a pas encore de mesure de la qualité du plan.

### Pourquoi le score est calculé par le code

Si on demande à un LLM « donne une note sur 100 », il répond au jugé, et sa réponse change d'une
fois sur l'autre. On procède donc autrement :

1. Qwen évalue **chaque compétence** : maîtrisée, partielle ou absente, en citant le passage du CV qui le prouve.
2. Le **code** calcule ensuite : 70 % pour les compétences obligatoires, 15 % pour les souhaitées et 15 % pour l'expérience.

Le résultat peut **s'expliquer** (« des points perdus à cause de Kafka ») et **se tester**
([ADR 0003](adr/0003-score-computed-in-code.md)).

### jevos, le contrôleur rapide (« Système 1 / Système 2 »)

Le nom vient d'une idée de psychologie : le **Système 1** pense vite et de façon instinctive, le
**Système 2** lentement et en réfléchissant.

- **Qwen joue le Système 2** : il est intelligent mais lent, il lui faut plusieurs secondes par réponse.
- **jevos joue le Système 1** : c'est un petit modèle open source sur processeur qui ne sait répondre qu'à des **questions fermées** (oui/non), mais en **environ 50 millisecondes**. Il donne une **probabilité calibrée** : quand il dit « 90 % », il a raison environ 9 fois sur 10.

Le système l'utilise à trois moments :

- **Pour vérifier les compétences** : quand Qwen hésite (« partiel »), jevos tranche. Quand les deux sont nettement en désaccord, la compétence reçoit une note intermédiaire et le désaccord est signalé.
- **Pour vérifier les affirmations de la lettre** : « Ce que dit cette phrase est-il vraiment dans le CV ? »
- **Comme outil de l'agent de carrière** : « le CV montre-t-il cette compétence proche ? »

Avant de l'intégrer, **on l'a testé** sur 194 questions annotées à la main. Ce test a révélé qu'il
faut lui poser les questions **en anglais**, même quand le CV est en français. Sinon, ses
performances chutent de 100 % à 41 % sur la vérification des affirmations
([couche de décision](decision-layer.md), [ADR 0006](adr/0006-decision-layer-jevos.md)).

### L'API (FastAPI), la porte d'entrée

| Action | Ce qu'elle fait |
|---|---|
| `POST /analyses` | On envoie le CV et l'offre. On reçoit **tout de suite** un identifiant, et l'analyse tourne en arrière-plan. |
| `GET /analyses/{id}` | « Où en est-on ? » Réponse : `running`, puis `awaiting_approval`, puis `completed`. |
| `POST /analyses/{id}/decision` | On valide la lettre, on demande une modification ou on la refuse. |

Tout peut se tester dans le navigateur sur **http://127.0.0.1:8000/docs** (Swagger), avec des
boutons et sans écrire une ligne de code. Les formats acceptés sont `.txt`, `.md`, `.pdf` (texte
ou scanné) et les images. Détails dans la [doc de l'API](api.md).

### Langfuse, Prometheus et Grafana : voir ce qui se passe

| Outil | Ce qu'il montre | La question à laquelle il répond |
|---|---|---|
| **Langfuse** (traces) | Le détail d'**une** analyse : chaque appel à Qwen, la consigne, la réponse, les tokens, le temps | « **Pourquoi** cette lettre est-elle mauvaise ? » |
| **Prometheus + Grafana** (métriques) | Des statistiques sur **toutes** les analyses : nombre d'analyses, agent le plus lent, tokens par minute, vitesse du GPU, décisions de jevos | « Le système **va-t-il bien** ? » |

Voir [l'observabilité](observability.md).

### DeepEval, la mesure de la qualité

Comment savoir si les réponses sont bonnes ? On a créé **8 cas fictifs** avec la réponse
attendue. Par exemple : « une infirmière qui postule en data scientist doit avoir un score entre 0 et 25 ».

- **Contrôles automatiques** : le score est-il dans la fourchette attendue ? Les compétences manquantes ont-elles été trouvées ?
- **LLM-juge** : la lettre invente-t-elle des choses ? Est-elle pertinente ?

C'est grâce à cette évaluation qu'on a découvert que **la première version de l'intégration de
jevos faisait baisser les scores**, qu'on en a trouvé la cause, puis qu'on l'a corrigée
([ADR 0008](adr/0008-fusion-v2-after-ab-eval.md)).

### Docker, tout démarrer d'une commande

Les 11 services (API, Postgres, Langfuse et ses 5 composants, Prometheus, Grafana, jevos)
démarrent avec **`make up`**. Chacun est dans son conteneur, avec une version fixée, ce qui rend
le projet **reproductible** sur une autre machine. Seul le LLM tourne en dehors de Docker
([exploitation](operations.md)).

## 5. Comment c'est vérifié

| Niveau | Outil | Ce qu'il vérifie | Quand |
|---|---|---|---|
| Code | **pytest** | Chaque fonction, avec un faux LLM pour aller vite | À chaque modification |
| Style | **ruff** | Code propre et cohérent | À chaque modification |
| Automatique | **GitHub Actions (CI)** | Tests, lint et construction de l'image Docker | À chaque push |
| Qualité IA | **DeepEval** | Les réponses ont-elles du sens ? | À la main (`make eval`, environ 30 min) |
| Modèle de décision | **Banc d'essai** | jevos est-il fiable et bien calibré ? | Une fois, avant de l'intégrer |

## 6. Ce que ce projet démontre

Un simple chatbot montre qu'on sait **appeler un LLM**. Ce projet montre qu'on sait **construire
un système fiable autour d'un LLM** :

- **Une architecture d'agents**, avec des choix justifiés ([ADR](adr/README.md)).
- **Les deux grandes approches, chacune à sa place** : un **workflow** déterministe pour le
  parcours fixe, et un **agent autonome** équipé d'outils là où la tâche est ouverte.
- **Les patrons classiques des systèmes à base de LLM**, chacun utilisé pour une raison précise :

  | Patron | Où dans le projet |
  |---|---|
  | **Enchaînement de prompts** (*prompt chaining*) | analyse → matching → conseil → rédaction, chaque étape nourrit la suivante |
  | **Parallélisation** | `parse_offer` et `parse_cv` en même temps ; les appels d'outils d'un même tour |
  | **Orchestrateur et équipes** | le superviseur et ses quatre sous-graphes |
  | **Évaluateur-optimiseur** | la boucle rédacteur / relecteur de la lettre |
  | **Humain dans la boucle** | la pause `human_review` avant la version finale |
  | **Garde-fous** | score calculé par le code, vérifications de jevos, URL et compétences contrôlées |
  | **Agent ReAct équipé d'outils** | le conseiller de carrière |

- **Une méfiance saine envers l'IA** : le score est calculé par le code, une relecture détecte les inventions, l'agent de carrière ne peut citer que ce que ses outils ont trouvé, et c'est un humain qui valide.
- **De l'observabilité** : traces et métriques, comme en production.
- **Une démarche de mesure** : tester avant d'intégrer, mesurer après, constater qu'une première version dégradait certains résultats, comprendre pourquoi, corriger.

## 7. Pour aller plus loin

| Je veux… | Lire |
|---|---|
| Installer et lancer le projet | [README](../README.md#quick-start), [exploitation](operations.md) |
| Comprendre l'architecture en détail | [architecture](architecture.md) |
| Savoir pourquoi chaque choix a été fait | [ADR](adr/README.md) |
| Comprendre l'agent de carrière | [ADR 0009](adr/0009-workflow-plus-autonomous-agent.md), [architecture, section 6](architecture.md#6-career-advisor-agent), [données ESCO et catalogue](../data/career/README.md) |
| Utiliser l'API | [API](api.md) |
| Comprendre jevos et le banc d'essai | [couche de décision](decision-layer.md) |
| Lire le code | dans cet ordre : `schemas.py` → `graph/build.py` → `graph/teams/matching.py` → `decisions/service.py` → `api/routes.py` |
