# Mini DocuLens

Assistant IA pour l'analyse de documents réglementaires pharmaceutiques, développé pour PharmaCo Belgium.

## Utilité

Mini DocuLens permet aux équipes qualité de **poser des questions en langage naturel** sur leurs documents réglementaires (SOPs, guidelines ICH, directives FDA/EMA) et d'obtenir des réponses précises avec les sources exactes.

Fini de parcourir manuellement des dizaines de pages de guidelines — l'application retrouve instantanément l'information pertinente et cite le document source ainsi que le numéro de page.

## Fonctionnalités

- **Upload multi-documents** — charge plusieurs PDFs simultanément
- **Indexation automatique** — découpe, vectorise et indexe les documents localement
- **Recherche sémantique** — retrouve les passages les plus pertinents par similarité de sens
- **Réponses ancrées** — Gemini génère une réponse basée uniquement sur les documents fournis, sans hallucination
- **Citations obligatoires** — chaque réponse indique le document source et le numéro de page
- **Multilingue** — répond en français ou en anglais selon la langue de la question

## Stack technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Base vectorielle | ChromaDB (local) |
| LLM | Google Gemini 2.5 Flash |
| Extraction PDF | pdfplumber |
| Orchestration | LangChain |

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

```bash
streamlit run app.py
```

## Utilisation

1. Dans le panneau gauche, upload un ou plusieurs PDFs
2. Clique sur **Indexer les documents** (première fois ~2-3 min pour télécharger le modèle d'embedding)
3. Pose ta question dans la zone principale
4. La réponse s'affiche avec les sources utilisées

## Structure du projet

```
mini-doculens/
├── app.py              # Interface Streamlit
├── src/
│   ├── ingest.py       # Pipeline d'ingestion (PDF → ChromaDB)
│   ├── rag.py          # Pipeline RAG (question → réponse)
│   └── prompts.py      # Prompts système et utilisateur
├── documents/          # PDFs uploadés
├── chroma_db/          # Base vectorielle (générée automatiquement)
├── models/             # Cache du modèle d'embedding
└── requirements.txt
```
