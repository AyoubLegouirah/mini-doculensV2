# Mini DocuLens

Assistant IA pour l'analyse de documents réglementaires pharmaceutiques, développé pour PharmaCo Belgium.

Deux modes disponibles : un **système RAG** pour les questions ponctuelles, et un **agent IA autonome** pour les missions d'analyse complexes avec génération de rapports et graphiques.

## Projets

### Projet 1 — Mini DocuLens (RAG)
Pose une question en langage naturel sur tes documents réglementaires. L'application retrouve instantanément les passages pertinents et répond en citant le document source et le numéro de page.

### Projet 2 — Agent IA (ReAct)
Donne une mission complexe à l'agent. Il raisonne, enchaîne plusieurs recherches, génère un graphique et produit un rapport structuré téléchargeable — sans intervention humaine entre les étapes.

## Fonctionnalités

**Projet 1**
- Upload multi-documents — charge plusieurs PDFs simultanément
- Indexation automatique — découpe, vectorise et indexe les documents localement
- Recherche sémantique — retrouve les passages les plus pertinents par similarité de sens
- Réponses ancrées — Gemini génère une réponse basée uniquement sur les documents fournis
- Citations obligatoires — chaque réponse indique le document source et le numéro de page
- Multilingue — répond en français ou en anglais selon la langue de la question

**Projet 2**
- Pattern ReAct — l'agent raisonne, agit et observe en boucle jusqu'à mission accomplie
- 4 outils disponibles : recherche documentaire, recherche web, génération de rapport, génération de graphique
- Rapports Markdown téléchargeables directement depuis l'interface
- Graphiques automatiques — visualisation des données extraites des documents
- Missions exemples — 3 missions préconfigurées pour démarrer rapidement

## Stack technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Base vectorielle | ChromaDB (local) |
| LLM | Google Gemini 2.0 Flash |
| Extraction PDF | pdfplumber |
| Orchestration RAG | LangChain |
| Agent | LangChain AgentExecutor + ReAct |
| Graphiques | Matplotlib |
| Recherche web | DuckDuckGo Search |

## Installation

```bash
git clone <repo>
cd mini-doculens
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Crée un fichier `.env` à la racine :
```
GEMINI_API_KEY=ta_clé_api
```
## Lancement

**Projet 1 — Interface RAG**
```bash
streamlit run app.py
```

**Projet 2 — Interface Agent**
```bash
streamlit run agent_app.py
```

## Utilisation

**Projet 1**
1. Dans le panneau gauche, upload un ou plusieurs PDFs
2. Clique sur **Indexer les documents** (première fois ~2-3 min pour télécharger le modèle d'embedding)
3. Pose ta question dans la zone principale
4. La réponse s'affiche avec les sources utilisées

**Projet 2**
1. Lance d'abord le Projet 1 pour indexer tes documents
2. Décris ta mission dans le champ principal (ou clique sur un exemple)
3. Clique sur **Lancer l'agent**
4. L'agent raisonne, appelle ses outils et produit le rapport final
## Structure du projet

```
mini-doculens/
├── app.py              # Interface Streamlit — Projet 1
├── agent_app.py        # Interface Streamlit — Projet 2
├── src/
│   ├── ingest.py       # Pipeline d'ingestion (PDF → ChromaDB)
│   ├── rag.py          # Pipeline RAG (question → réponse)
│   ├── prompts.py      # Prompts système et utilisateur
│   ├── tools.py        # Outils de l'agent (@tool)
│   └── agent.py        # AgentExecutor ReAct
├── documents/          # PDFs uploadés
├── chroma_db/          # Base vectorielle (générée automatiquement)
├── models/             # Cache du modèle d'embedding
├── reports/            # Rapports et graphiques générés
└── requirements.txt
```