# app.py

import os
import shutil
import streamlit as st
from src.ingest import load_pdfs, chunk_pages, embed_and_index
from src.rag import ask

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Mini DocuLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def save_uploaded_files(uploaded_files: list) -> int:
    """
    Sauvegarde les fichiers uploadés dans le dossier documents/.
    
    Vide d'abord le dossier pour éviter les doublons, puis écrit
    chaque fichier uploadé via Streamlit sur le disque local.
    
    Args:
        uploaded_files: Liste de fichiers UploadedFile depuis st.file_uploader
    
    Returns:
        Nombre de fichiers sauvegardés avec succès
    """
    # Vide le dossier documents/ avant chaque nouvelle indexation
    if os.path.exists("documents"):
        shutil.rmtree("documents")
    os.makedirs("documents")

    count = 0
    for uploaded_file in uploaded_files:
        filepath = os.path.join("documents", uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        count += 1

    return count


def run_ingestion() -> bool:
    """
    Orchestre le pipeline d'ingestion complet depuis l'interface.
    
    Appelle successivement les 3 fonctions d'ingest.py en affichant
    une barre de progression Streamlit à chaque étape.
    Supprime l'ancienne base ChromaDB avant de reconstruire.
    
    Returns:
        True si l'ingestion s'est terminée sans erreur, False sinon
    """
    try:
        # Supprime l'ancienne base vectorielle pour repartir propre
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        with st.spinner("📄 Lecture des PDFs..."):
            pages = load_pdfs("documents")
            if not pages:
                st.error("Aucun texte extrait. Vérifie que tes PDFs contiennent du texte.")
                return False

        with st.spinner("✂️ Découpage en chunks..."):
            chunks, metadatas = chunk_pages(pages)

        with st.spinner("🧠 Calcul des embeddings et indexation... (peut prendre 2-3 min)"):
            embed_and_index(chunks, metadatas)

        return True

    except Exception as e:
        st.error(f"Erreur durant l'ingestion : {str(e)}")
        return False


def display_sources(chunks: list[dict]) -> None:
    """
    Affiche les chunks sources dans des encadrés expandables.
    
    Chaque source est présentée avec le nom du document et le numéro
    de page en titre, et le contenu du chunk en texte expandable pour
    ne pas surcharger l'interface.
    
    Args:
        chunks: Liste de chunks avec métadonnées source et page
    """
    st.markdown("### 📚 Sources utilisées")

    for i, chunk in enumerate(chunks, start=1):
        with st.expander(f"Source {i} — {chunk['source']}, Page {chunk['page']}"):
            st.markdown(f"```\n{chunk['text']}\n```")


# ---------------------------------------------------------------------------
# INTERFACE PRINCIPALE
# ---------------------------------------------------------------------------

# Titre principal
st.title("🔬 Mini DocuLens")
st.markdown("*Assistant IA pour documents réglementaires pharmaceutiques — PharmaCo Belgium*")
st.divider()

# ---------------------------------------------------------------------------
# SIDEBAR — Upload et indexation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Documents")
    st.markdown("Upload tes PDFs puis clique sur **Indexer**.")

    # Widget d'upload — accepte plusieurs PDFs simultanément
    uploaded_files = st.file_uploader(
        label="Sélectionne tes PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Guidelines FDA, EMA, ICH ou tout autre document réglementaire"
    )

    # Bouton d'indexation — visible uniquement si des fichiers sont uploadés
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} fichier(s) sélectionné(s)")

        if st.button("🚀 Indexer les documents", use_container_width=True, type="primary"):
            # Sauvegarde les fichiers sur le disque
            count = save_uploaded_files(uploaded_files)
            st.info(f"💾 {count} fichier(s) sauvegardé(s)")

            # Lance le pipeline d'ingestion
            success = run_ingestion()

            if success:
                st.success("🎉 Base vectorielle prête !")
                # Marque la session comme indexée pour débloquer la zone de question
                st.session_state["indexed"] = True
            else:
                st.session_state["indexed"] = False

    st.divider()

    # Indicateur d'état de la base vectorielle
    if os.path.exists("chroma_db") and os.listdir("chroma_db"):
        st.success("🟢 Base vectorielle active")
    else:
        st.warning("🔴 Aucune base vectorielle — indexe d'abord tes documents")

# ---------------------------------------------------------------------------
# ZONE PRINCIPALE — Question et réponse
# ---------------------------------------------------------------------------

# Vérifie si une base vectorielle existe (indexation faite au moins une fois)
db_ready = os.path.exists("chroma_db") and os.listdir("chroma_db")

if not db_ready:
    # Message d'accueil si aucune base n'existe encore
    st.info("👈 Commence par uploader tes PDFs dans le panneau de gauche et clique sur **Indexer les documents**.")

else:
    # Zone de question — active uniquement si la base est prête
    st.markdown("### ❓ Pose ta question")

    question = st.text_input(
        label="Question",
        placeholder="Ex: What are the key principles of ICH Q10 quality system?",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Rechercher", type="primary", use_container_width=True)

    # Déclenche la recherche RAG
    if search_button and question.strip():
        with st.spinner("🔄 Recherche en cours..."):
            try:
                result = ask(question)

                # Affiche la réponse principale
                st.markdown("### 💬 Réponse")
                st.markdown(result["answer"])

                st.divider()

                # Affiche les sources utilisées
                display_sources(result["chunks"])

            except Exception as e:
                st.error(f"Erreur : {str(e)}")

    elif search_button and not question.strip():
        st.warning("⚠️ Saisis une question avant de rechercher.")