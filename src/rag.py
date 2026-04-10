# src/rag.py

import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.prompts import RAG_SYSTEM_PROMPT, RAG_HUMAN_PROMPT

# Charge les variables d'environnement depuis .env
load_dotenv()

# --- Configuration ---
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_RESULTS = 5  # Nombre de chunks récupérés par recherche


def load_vectorstore() -> Chroma:
    """
    Charge la base vectorielle ChromaDB existante depuis le disque.
    
    Cette fonction suppose que ingest.py a déjà été exécuté au moins
    une fois. Si le dossier chroma_db/ n'existe pas, une erreur sera
    levée avec un message explicite.
    
    Returns:
        Chroma: Instance de la base vectorielle prête à être interrogée
    
    Raises:
        FileNotFoundError: Si chroma_db/ n'existe pas
    """
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(
            f"Base vectorielle introuvable : {CHROMA_DB_DIR}/\n"
            "Lance d'abord : python -m src.ingest"
        )

    print(f"🔄 Chargement du modèle d'embedding : {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    cache_folder="./models"
)

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    print("✅ Base vectorielle chargée")
    return vectorstore


def retrieve_chunks(vectorstore: Chroma, question: str) -> list[dict]:
    """
    Recherche les chunks les plus pertinents pour une question donnée.
    
    Convertit la question en vecteur avec le même modèle d'embedding
    utilisé lors de l'ingestion, puis effectue une recherche par
    similarité cosinus dans ChromaDB pour trouver les TOP_K_RESULTS
    passages les plus proches sémantiquement.
    
    Args:
        vectorstore: Instance ChromaDB chargée
        question: Question posée par l'utilisateur en langage naturel
    
    Returns:
        Liste de dicts contenant :
        - text: Le contenu textuel du chunk
        - source: Nom du fichier PDF d'origine
        - page: Numéro de page dans le document original
    """
    results = vectorstore.similarity_search(question, k=TOP_K_RESULTS)

    chunks = []
    for doc in results:
        chunks.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "inconnu"),
            "page": doc.metadata.get("page", "?")
        })

    print(f"✅ {len(chunks)} chunks récupérés depuis ChromaDB")
    return chunks


def build_context(chunks: list[dict]) -> str:
    """
    Formate les chunks récupérés en un bloc de contexte lisible
    pour le LLM.
    
    Chaque chunk est présenté avec sa source et son numéro de page
    pour permettre au LLM de citer correctement ses références dans
    la réponse finale.
    
    Args:
        chunks: Liste de chunks avec leurs métadonnées
    
    Returns:
        String formatée prête à être injectée dans le prompt RAG
    
    Example:
        [Document 1 - fichier.pdf, Page 3]
        Contenu du chunk ici...
        
        [Document 2 - autre.pdf, Page 7]
        Contenu du chunk ici...
    """
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Document {i} - {chunk['source']}, Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(question: str, context: str) -> str:
    """
    Envoie la question + le contexte à Gemini et retourne la réponse.
    
    Configure Gemini avec la clé API depuis .env, applique le system
    prompt pharmaceutique défini dans prompts.py, et génère une réponse
    ancrée uniquement dans le contexte fourni.
    
    Args:
        question: Question originale de l'utilisateur
        context: Bloc de contexte formaté par build_context()
    
    Returns:
        Réponse textuelle générée par Gemini avec citations des sources
    
    Raises:
        ValueError: Si GEMINI_API_KEY n'est pas définie dans .env
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY introuvable.\n"
            "Vérifie que ton fichier .env contient : GEMINI_API_KEY=ta_clé"
        )

    # Configure le client Gemini avec la clé API
    genai.configure(api_key=api_key)

    # Instancie le modèle
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=RAG_SYSTEM_PROMPT
    )

    # Formate le prompt avec le contexte et la question
    prompt = RAG_HUMAN_PROMPT.format(
        context=context,
        question=question
    )

    print("🔄 Génération de la réponse via Gemini...")
    response = model.generate_content(prompt)

    return response.text


def ask(question: str) -> dict:
    """
    Fonction principale du pipeline RAG — point d'entrée unique.
    
    Orchestre les 3 phases du pipeline dans l'ordre :
    1. Retrieval  — récupère les chunks pertinents depuis ChromaDB
    2. Augmentation — formate le contexte pour le LLM
    3. Generation — envoie à Gemini et récupère la réponse
    
    C'est cette fonction qui sera appelée depuis app.py (Streamlit)
    et depuis src/ingest.py pour les tests en ligne de commande.
    
    Args:
        question: Question posée par l'utilisateur en langage naturel
    
    Returns:
        Dict contenant :
        - answer: La réponse générée par Gemini
        - chunks: Les chunks sources utilisés pour générer la réponse
        - question: La question originale (pour affichage dans l'UI)
    
    Example:
        result = ask("Quels sont les délais de validation ICH Q10 ?")
        print(result["answer"])
        print(result["chunks"])  # Pour afficher les sources
    """
    print(f"\n❓ Question : {question}\n")

    # Phase 1 — Chargement de la base vectorielle
    vectorstore = load_vectorstore()

    # Phase 2 — Retrieval des chunks pertinents
    chunks = retrieve_chunks(vectorstore, question)

    # Phase 3 — Construction du contexte
    context = build_context(chunks)

    # Phase 4 — Génération de la réponse
    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "chunks": chunks,
        "question": question
    }


# Point d'entrée pour tester en ligne de commande
if __name__ == "__main__":
    question_test = "What are the key principles of pharmaceutical quality systems?"
    result = ask(question_test)

    print("\n" + "="*60)
    print("RÉPONSE :")
    print("="*60)
    print(result["answer"])

    print("\n" + "="*60)
    print("SOURCES UTILISÉES :")
    print("="*60)
    for chunk in result["chunks"]:
        print(f"  📄 {chunk['source']} — Page {chunk['page']}")