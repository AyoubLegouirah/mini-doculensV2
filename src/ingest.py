# src/ingest.py

import os
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# --- Configuration ---
DOCUMENTS_DIR = "documents"
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def load_pdfs(documents_dir: str) -> list[dict]:
    """
    Parcourt le dossier documents/ et extrait le texte
    de chaque PDF page par page.
    Retourne une liste de dicts {text, source, page}
    """
    pages = []

    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("⚠️  Aucun PDF trouvé dans le dossier documents/")
        return pages

    for filename in pdf_files:
        filepath = os.path.join(documents_dir, filename)
        print(f"📄 Lecture de : {filename}")

        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()

                if text and text.strip():
                    pages.append({
                        "text": text.strip(),
                        "source": filename,
                        "page": page_num
                    })

    print(f"✅ {len(pages)} pages extraites depuis {len(pdf_files)} PDF(s)")
    return pages


def chunk_pages(pages: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Découpe chaque page en chunks de 400 tokens
    avec 50 tokens de chevauchement.
    Retourne les textes et leurs métadonnées séparément.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks = []
    all_metadatas = []

    for page in pages:
        chunks = splitter.split_text(page["text"])

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": page["source"],
                "page": page["page"]
            })

    print(f"✅ {len(all_chunks)} chunks créés")
    return all_chunks, all_metadatas


def embed_and_index(chunks: list[str], metadatas: list[dict]) -> None:
    """
    Convertit les chunks en embeddings vectoriels
    et les stocke dans ChromaDB localement.
    """
    print(f"🔄 Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
    print("   (Premier lancement = téléchargement ~1.2GB, patience...)")

    embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    cache_folder="./models"
)

    print("🔄 Indexation dans ChromaDB...")

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DB_DIR
    )

    print(f"✅ {len(chunks)} chunks indexés dans ChromaDB")
    print(f"📁 Base vectorielle sauvegardée dans : {CHROMA_DB_DIR}/")


def main():
    print("🚀 Démarrage de l'ingestion des documents...\n")

    # Étape 1 — Lecture des PDFs
    pages = load_pdfs(DOCUMENTS_DIR)
    if not pages:
        return

    # Étape 2 — Chunking
    chunks, metadatas = chunk_pages(pages)

    # Étape 3 — Embedding + Indexation
    embed_and_index(chunks, metadatas)

    print("\n🎉 Ingestion terminée ! La base vectorielle est prête.")


if __name__ == "__main__":
    main()