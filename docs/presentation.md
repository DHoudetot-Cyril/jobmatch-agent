# JobMatch Agent : le projet expliqué simplement

> Ce document présente le projet sans jargon, pour quelqu'un qui connaît l'informatique mais
> pas forcément l'IA. Pour les détails techniques, voir le [README](../README.md) et le dossier
> [`docs/`](README.md).

## 1. Le problème résolu

On a **une offre d'emploi** et **un CV**. Le système répond à trois questions :

1. **Ce CV correspond-il à cette offre ?** Il donne un score sur 100, avec une justification.
2. **Qu'est-ce qui manque ?** Il liste les compétences absentes et propose des conseils pour les acquérir.
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
  ║  parse_offer            parse_cv        (en parallèle)║
  ║  "Poste: Data Eng.,     "4 ans d'XP,                  ║
  ║   Python requis,         Python, Spark,               ║
  ║   Kafka requis..."       Airflow..."                  ║
  ╚══════════════════════════════════════════════════════╝
        │  (retour au superviseur)
        ▼
  ╔═══════════════════ ÉQUIPE MATCHING ══════════════════╗
  ║  score_match     → Qwen note chaque compétence :      ║
  ║                    Python = maîtrisé, Kafka = absent… ║
  ║  verify_skills   → jevos vérifie : "le CV prouve-t-il ║
  ║                    Kafka ?" → 3 % → confirmé absent   ║
  ║  analyze_gaps    → conseils : "faire un projet Kafka" ║
  ║  ➜ SCORE /100 calculé par le CODE, pas par l'IA       ║
  ╚══════════════════════════════════════════════════════╝
        │
        ▼
  ╔═══════════════════ ÉQUIPE RÉDACTION ═════════════════╗
  ║  write_letter  → Qwen écrit la lettre                 ║
  ║       ▲                 │                             ║
  ║       │ corrige         ▼                             ║
  ║       └──────── review_letter → Qwen relit et cherche ║
  ║                        │       les inventions         ║
  ║                 verify_claims → jevos vérifie chaque  ║
  ║                        │        affirmation douteuse  ║
  ║           (2 allers-retours maximum)                  ║
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

Une analyse complète prend **environ 5 à 6 minutes** sur un seul GPU grand public : le modèle a
27 milliards de paramètres, et la boucle de rédaction et de relecture fait plusieurs appels au LLM.

## 4. Les briques, une par une

### Qwen et llama.cpp, le « cerveau »

- **Qwen3.8-27B** est un LLM (un modèle de langage, comme ChatGPT) **open source**, avec 27 milliards de paramètres.
- **llama.cpp** est le programme qui le fait tourner sur le GPU. Il expose une API **compatible avec celle d'OpenAI**, donc tout outil prévu pour ChatGPT peut l'utiliser sans modification.
- **Ses capacités** : il produit du **JSON structuré** (des données propres, pas du texte libre) et il **lit des images**, ce qui permet de traiter les CV scannés.
- **Pourquoi il n'est pas dans Docker** : faire passer une carte AMD dans un conteneur est compliqué, et ça n'apportait rien ici ([ADR 0001](adr/0001-llama-cpp-on-host.md)).

### LangGraph, l'organisation du travail

C'est la bibliothèque qui organise les agents. Un **agent**, ici, est simplement une étape qui
appelle le LLM avec une consigne précise. On représente le travail comme un **graphe** : des
cases (les agents) reliées par des flèches (l'ordre d'exécution).

- **L'état** : un grand dossier partagé (l'offre, le CV, le score, la lettre…). Chaque agent y lit et y écrit.
- **Le superviseur** : il décide de la suite **avec de simples règles en code**, pas avec l'IA. Comme les étapes sont toujours les mêmes, demander à un LLM de choisir la suivante serait plus lent, plus cher et moins fiable ([ADR 0002](adr/0002-deterministic-supervisor.md)).
- **Les sous-graphes** : chaque équipe est un petit graphe à elle seule. C'est ce qui rend l'ensemble **hiérarchique**.
- **L'interruption (`interrupt`)** : le graphe peut se mettre en **pause** pour attendre un humain. Son état est sauvegardé dans **Postgres** : une analyse en pause reprend même après un redémarrage du serveur. En revanche, une analyse en cours au moment du redémarrage est marquée en échec ([ADR 0004](adr/0004-hitl-interrupt-postgres.md)).

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

Le système l'utilise à deux moments :

- **Pour vérifier les compétences** : quand Qwen hésite (« partiel »), jevos tranche. Quand les deux sont nettement en désaccord, la compétence reçoit une note intermédiaire et le désaccord est signalé.
- **Pour vérifier les affirmations de la lettre** : « Ce que dit cette phrase est-il vraiment dans le CV ? »

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
- **Une méfiance saine envers l'IA** : le score est calculé par le code, une relecture détecte les inventions, et c'est un humain qui valide.
- **De l'observabilité** : traces et métriques, comme en production.
- **Une démarche de mesure** : tester avant d'intégrer, mesurer après, constater qu'une première version dégradait certains résultats, comprendre pourquoi, corriger.

## 7. Pour aller plus loin

| Je veux… | Lire |
|---|---|
| Installer et lancer le projet | [README](../README.md#quick-start), [exploitation](operations.md) |
| Comprendre l'architecture en détail | [architecture](architecture.md) |
| Savoir pourquoi chaque choix a été fait | [ADR](adr/README.md) |
| Utiliser l'API | [API](api.md) |
| Comprendre jevos et le banc d'essai | [couche de décision](decision-layer.md) |
| Lire le code | dans cet ordre : `schemas.py` → `graph/build.py` → `graph/teams/matching.py` → `decisions/service.py` → `api/routes.py` |
