# app.py
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FICHE D'IDENTITÉ DU FICHIER                                         ║
# ║                                                                      ║
# ║  Rôle    : Interface utilisateur Streamlit du pipeline RAG.         ║
# ║            Upload de PDFs → indexation → question → réponse.        ║
# ║                                                                      ║
# ║  En Java : @Controller Spring MVC + Thymeleaf, mais sans HTML/CSS.  ║
# ║            Streamlit génère l'interface à partir du code Python pur. ║
# ║                                                                      ║
# ║  En Angular : Composant avec template HTML intégré —                ║
# ║               mais déclaré en Python dans un seul fichier.          ║
# ║                                                                      ║
# ║  Lancement : streamlit run app.py                                   ║
# ║              (ouvre automatiquement http://localhost:8501)           ║
# ║                                                                      ║
# ║  Flux    :                                                           ║
# ║    Sidebar : upload PDFs → save_uploaded_files() → run_ingestion()  ║
# ║    Main    : question saisie → ask() → réponse + sources affichées  ║
# ║                                                                      ║
# ║  Dépendances :                                                       ║
# ║    ← src/ingest.py : load_pdfs, chunk_pages, embed_and_index        ║
# ║    ← src/rag.py    : ask()                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════
# BLOC 1 : IMPORTS
# ═══════════════════════════════════════════════════════════════════════

import os
import shutil
# ↑ MICRO-COURS : le module shutil (shell utilities)
#   "shutil" = raccourci pour "shell utilities" — opérations de haut niveau
#   sur les fichiers et dossiers (copier, déplacer, supprimer).
#
#   En Java : équivalent de Apache Commons IO FileUtils ou Files NIO.
#   En Python :
#     shutil.rmtree("dossier")          → supprimer un dossier et son contenu
#     shutil.copy("source", "dest")     → copier un fichier
#     shutil.move("source", "dest")     → déplacer un fichier
#     shutil.copytree("src", "dst")     → copier un dossier entier
#
#   Différence avec os :
#     os.remove("fichier")   → supprime UN fichier
#     os.rmdir("dossier")    → supprime un dossier VIDE seulement
#     shutil.rmtree("dossier") → supprime un dossier et TOUT son contenu

import streamlit as st
# ↑ MICRO-COURS : Streamlit — qu'est-ce que c'est ?
#
#   Streamlit est une bibliothèque Python qui transforme un script Python
#   en application web interactive SANS écrire de HTML/CSS/JavaScript.
#
#   Analogie Java/Spring :
#     Tu connais Spring MVC + Thymeleaf :
#       @Controller → gère les routes
#       Thymeleaf   → génère le HTML
#       HTML/CSS    → mise en page
#     Total : 3 fichiers minimum pour une page simple.
#
#   Avec Streamlit :
#     Un seul fichier Python → application web complète.
#     st.title("Bonjour")      → génère un <h1>
#     st.text_input("Nom")     → génère un <input>
#     st.button("Clique")      → génère un <button>
#     Streamlit gère le HTML, CSS, JS, WebSocket, rechargement... tout.
#
#   En Angular : comme un composant dont le template HTML serait
#   écrit en Python au lieu de HTML. Chaque st.xxx() est un widget.
#
#   Convention : toujours importé comme "st" (alias universellement adopté).

from src.ingest import load_pdfs, chunk_pages, embed_and_index
# ↑ Import des 3 fonctions du pipeline d'ingestion (src/ingest.py)
from src.rag import ask
# ↑ Import du point d'entrée unique du pipeline RAG (src/rag.py)


# ═══════════════════════════════════════════════════════════════════════
# BLOC 2 : CONFIGURATION DE LA PAGE STREAMLIT
#
# MICRO-COURS : Le modèle d'exécution Streamlit — CRUCIAL à comprendre
#
# C'est la différence fondamentale entre Streamlit et Spring/Angular.
#
# En Spring/Angular, ton code s'exécute UNE FOIS au démarrage,
# puis des méthodes (@GetMapping, ngOnClick) sont appelées sur
# des ÉVÉNEMENTS spécifiques.
#
# En Streamlit : le script Python ENTIER est ré-exécuté du début
# à la fin à CHAQUE interaction de l'utilisateur :
#   - Clic sur un bouton → tout le fichier re-tourne
#   - Saisie dans un input → tout le fichier re-tourne
#   - Upload d'un fichier → tout le fichier re-tourne
#
# Conséquence : l'ordre du code dans le fichier = l'ordre d'affichage.
# Ce qui est en haut s'affiche en premier, ce qui est en bas ensuite.
#
# Pour PERSISTER des données entre deux rechargements → st.session_state
# (expliqué plus bas)
#
# st.set_page_config() DOIT être le premier appel Streamlit du fichier.
# Si tu appelles st.title() avant, tu auras une erreur.
# ═══════════════════════════════════════════════════════════════════════

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Mini DocuLens",     # Titre de l'onglet navigateur (comme <title> HTML)
    page_icon="🔬",                 # Favicon de l'onglet (peut être un emoji ou une URL d'image)
    layout="wide",                  # "wide" = pleine largeur | "centered" = centré (défaut)
    initial_sidebar_state="expanded" # Sidebar ouverte au chargement initial
)
# ↑ En Java/Spring : équivalent d'une configuration dans index.html :
#   <head>
#     <title>Mini DocuLens</title>
#     <link rel="icon" href="🔬">
#   </head>


# ═══════════════════════════════════════════════════════════════════════
# BLOC 3 : FONCTION save_uploaded_files — Sauvegarde des PDFs uploadés
#
# Rôle : Reçoit les fichiers uploadés via Streamlit, vide le dossier
#        documents/ et sauvegarde les nouveaux fichiers sur le disque.
# ═══════════════════════════════════════════════════════════════════════

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
    # ─────────────────────────────────────────────────────────────────
    # Supprime le dossier documents/ s'il existe, puis le recrée vide.
    # Cela évite de mélanger d'anciens et de nouveaux PDFs.
    #
    # shutil.rmtree("documents") : supprime le dossier ET tout son contenu
    # os.makedirs("documents")   : recrée le dossier vide
    #
    # En Java (Apache Commons IO) :
    #   FileUtils.deleteDirectory(new File("documents"));
    #   new File("documents").mkdirs();
    # ─────────────────────────────────────────────────────────────────
    # Vide le dossier documents/ avant chaque nouvelle indexation
    if os.path.exists("documents"):
        shutil.rmtree("documents")   # Supprime documents/ et tous les PDFs dedans
    os.makedirs("documents")         # Recrée un dossier documents/ vide

    count = 0  # Compteur de fichiers sauvegardés avec succès
    # ↑ En Java : int count = 0;

    for uploaded_file in uploaded_files:
        filepath = os.path.join("documents", uploaded_file.name)
        # ↑ uploaded_file.name : le nom original du fichier (ex: "sop_nettoyage.pdf")
        #   os.path.join() construit "documents/sop_nettoyage.pdf"

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : Mode d'écriture binaire "wb"
        #
        # On a vu "w" (write texte) dans tools.py pour les rapports Markdown.
        # Ici on utilise "wb" (write binary) pour les fichiers PDF.
        #
        # "w"  = write text   : écrit des chaînes de caractères (UTF-8 par défaut)
        # "wb" = write binary : écrit des octets bruts (bytes)
        #
        # Les PDFs sont des fichiers BINAIRES (pas du texte lisible).
        # Si tu utilisais "w" pour un PDF, Python essaierait d'encoder
        # les octets en texte → fichier corrompu.
        #
        # Règle : toujours "wb" pour les images, PDFs, archives ZIP, etc.
        #         toujours "w" + encoding="utf-8" pour les fichiers texte.
        #
        # En Java :
        #   try (FileOutputStream fos = new FileOutputStream(filepath)) {
        #       fos.write(uploadedFile.getBytes());
        #   }
        # ─────────────────────────────────────────────────────────────
        with open(filepath, "wb") as f:       # "wb" = write binary (pour les PDFs)
            f.write(uploaded_file.getbuffer()) # getbuffer() = lire les octets du fichier uploadé
        # ↑ uploaded_file.getbuffer() : méthode Streamlit qui retourne
        #   le contenu du fichier uploadé sous forme de bytes (tableau d'octets).
        #   En Java Spring : MultipartFile.getBytes()

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : Incrément en Python
        #
        # En Java : count++ ou ++count ou count += 1
        # En Python : count += 1  UNIQUEMENT
        #
        # Python N'A PAS l'opérateur ++ (ni --).
        # count++ en Python → SyntaxError
        # ─────────────────────────────────────────────────────────────
        count += 1  # En Java : count++ — Python n'a pas d'opérateur ++

    return count  # Nombre de fichiers sauvegardés


# ═══════════════════════════════════════════════════════════════════════
# BLOC 4 : FONCTION run_ingestion — Pipeline d'ingestion depuis l'UI
#
# Rôle : Orchestre les 3 étapes d'ingestion (load → chunk → embed)
#        avec des indicateurs visuels Streamlit (spinners).
# ═══════════════════════════════════════════════════════════════════════

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
            shutil.rmtree("chroma_db")  # Supprime chroma_db/ et son contenu

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : st.spinner() — afficher un indicateur de chargement
        #
        # En Angular : <mat-progress-spinner> ou <mat-spinner>
        # En Java/Vaadin : ProgressBar ou Spinner component
        #
        # Syntaxe Python :
        #   with st.spinner("Message de chargement..."):
        #       code_long_à_exécuter()
        #   # ← spinner disparaît automatiquement ici
        #
        # C'est un context manager (comme with open() et with pdfplumber).
        # Streamlit affiche un spinner animé ET le message pendant que
        # le code dans le bloc s'exécute.
        #
        # RAPPEL modèle d'exécution Streamlit :
        #   Pendant l'exécution du bloc "with st.spinner()", le navigateur
        #   est "gelé" (Streamlit attend que le Python finisse).
        #   Pour des opérations très longues, il faudrait de l'asynchrone
        #   (st.empty() + threading), mais ici le spinner suffit.
        # ─────────────────────────────────────────────────────────────
        with st.spinner("📄 Lecture des PDFs..."):          # Affiche spinner pendant la lecture
            pages = load_pdfs("documents")
            if not pages:
                st.error("Aucun texte extrait. Vérifie que tes PDFs contiennent du texte.")
                # ↑ st.error() : affiche un message d'erreur en rouge dans l'UI
                #   En Angular : <mat-error> ou snackbar de type "error"
                return False  # Sortie anticipée si aucun texte

        with st.spinner("✂️ Découpage en chunks..."):       # Affiche spinner pendant le chunking
            chunks, metadatas = chunk_pages(pages)

        with st.spinner("🧠 Calcul des embeddings et indexation... (peut prendre 2-3 min)"):
            embed_and_index(chunks, metadatas)              # Étape la plus longue (~2-3 min au 1er lancement)

        return True  # Ingestion terminée avec succès

    except Exception as e:
        st.error(f"Erreur durant l'ingestion : {str(e)}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# BLOC 5 : FONCTION display_sources — Affichage des sources en accordéon
#
# Rôle : Affiche les chunks sources dans des panneaux dépliables
#        pour ne pas surcharger l'interface.
# ═══════════════════════════════════════════════════════════════════════

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
    # ↑ st.markdown() : affiche du texte formaté en Markdown
    #   "###" = H3 en Markdown = <h3> en HTML
    #   En Angular : <h3>Sources utilisées</h3>

    for i, chunk in enumerate(chunks, start=1):  # enumerate avec index déjà vu dans ingest.py et rag.py
        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : st.expander() — section dépliable (accordéon)
        #
        # En Angular : <mat-expansion-panel> d'Angular Material
        #              <details>/<summary> en HTML natif
        #
        # with st.expander("Titre") :
        #   Tout le code dans ce bloc est CACHÉ par défaut.
        #   L'utilisateur peut cliquer sur le titre pour déplier/replier.
        #   Utile pour les contenus longs qu'on ne veut pas afficher systématiquement.
        # ─────────────────────────────────────────────────────────────
        with st.expander(f"Source {i} — {chunk['source']}, Page {chunk['page']}"):
            # ↑ Le titre de l'accordéon — f-string avec accès dict (guillemets simples)
            st.markdown(f"```\n{chunk['text']}\n```")
            # ↑ ``` en Markdown = bloc de code avec police monospace
            #   Affiche le texte du chunk dans un encadré de code


# ═══════════════════════════════════════════════════════════════════════
# BLOC 6 : INTERFACE PRINCIPALE — Code exécuté au niveau du module
#
# MICRO-COURS : Code au niveau du module dans Streamlit
#
# En Spring MVC, tu aurais :
#   @GetMapping("/") → méthode appelée sur requête HTTP GET
#   @PostMapping("/") → méthode appelée sur requête HTTP POST
#
# En Streamlit, il n'y a PAS de routes ou de méthodes d'événements.
# Tout le code en dehors des fonctions s'exécute à chaque rechargement.
# C'est le "corps" de ta page — comme le contenu d'un ngOnInit()
# qui s'exécute à chaque interaction.
#
# L'ordre du code = l'ordre d'affichage dans la page.
# ═══════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# INTERFACE PRINCIPALE
# ---------------------------------------------------------------------------

# Titre principal de la page
st.title("🔬 Mini DocuLens")
# ↑ st.title() : affiche un titre H1 — équivalent de <h1> HTML
#   ou de st.markdown("# Mini DocuLens")

st.markdown("*Assistant IA pour documents réglementaires pharmaceutiques — PharmaCo Belgium*")
# ↑ Les * en Markdown = italique — équivalent de <em> HTML

st.divider()
# ↑ Affiche une ligne de séparation horizontale — équivalent de <hr> HTML


# ---------------------------------------------------------------------------
# SIDEBAR — Upload et indexation
# ---------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────
# MICRO-COURS : with st.sidebar — la barre latérale
#
# En Angular Material : <mat-sidenav> ou <mat-drawer>
# En Bootstrap/HTML : <aside class="sidebar">
#
# Tout le code dans le bloc "with st.sidebar:" apparaît
# dans la barre latérale gauche de l'application.
# En dehors du bloc → affiché dans la zone principale (centre).
#
# C'est encore une utilisation du context manager "with".
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:           # Tout ce bloc s'affiche dans la sidebar gauche
    st.header("📁 Documents")
    # ↑ st.header() : titre H2 — équivalent de <h2>
    st.markdown("Upload tes PDFs puis clique sur **Indexer**.")
    # ↑ **texte** en Markdown = gras — équivalent de <strong>

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : st.file_uploader() — widget d'upload de fichiers
    #
    # En Angular : <input type="file" (change)="onFileChange($event)">
    #              avec un FileReader et FormData
    # En Spring  : @RequestParam("file") MultipartFile file
    #
    # Streamlit gère TOUT automatiquement :
    #   - L'interface drag-and-drop
    #   - La validation du type de fichier (type=["pdf"])
    #   - La gestion de plusieurs fichiers (accept_multiple_files=True)
    #   - Le stockage temporaire en mémoire
    #
    # Retourne une liste d'objets UploadedFile (ou [] si rien uploadé).
    # ─────────────────────────────────────────────────────────────────
    # Widget d'upload — accepte plusieurs PDFs simultanément
    uploaded_files = st.file_uploader(
        label="Sélectionne tes PDFs",
        type=["pdf"],                    # Filtre : seulement les fichiers .pdf
        accept_multiple_files=True,      # Permet de sélectionner plusieurs fichiers à la fois
        help="Guidelines FDA, EMA, ICH ou tout autre document réglementaire"
        # ↑ "help" : texte d'aide affiché au survol (tooltip)
    )
    # ↑ RAPPEL modèle Streamlit :
    #   À chaque rechargement, st.file_uploader() est ré-exécuté.
    #   Si l'utilisateur a uploadé des fichiers, uploaded_files contient la liste.
    #   Si non, uploaded_files = [] (liste vide, donc falsy)

    # ─────────────────────────────────────────────────────────────────
    # Affichage conditionnel : visible seulement si des fichiers sont uploadés
    # "if uploaded_files:" → True si la liste est non vide (truthy)
    # ─────────────────────────────────────────────────────────────────
    # Bouton d'indexation — visible uniquement si des fichiers sont uploadés
    if uploaded_files:                                    # True si liste non vide
        st.success(f"✅ {len(uploaded_files)} fichier(s) sélectionné(s)")
        # ↑ st.success() : message vert de succès — équivalent Angular toastr "success"

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : st.button() — bouton cliquable
        #
        # En Angular : <button (click)="onIndexer()">Indexer</button>
        # En HTML/JS  : <button onclick="indexer()">Indexer</button>
        #
        # st.button() retourne True UNIQUEMENT sur le rechargement
        # déclenché par le clic — False tout le reste du temps.
        #
        # PIÈGE du modèle Streamlit :
        #   if st.button("Clique"):   ← True seulement au moment du clic
        #       faire_quelque_chose() ← exécuté UNE SEULE FOIS (au clic)
        #
        # Paramètres :
        #   use_container_width=True : le bouton prend toute la largeur disponible
        #   type="primary"           : style bleu (bouton principal) vs gris (secondaire)
        # ─────────────────────────────────────────────────────────────
        if st.button("🚀 Indexer les documents", use_container_width=True, type="primary"):
            # Ce bloc s'exécute UNE SEULE FOIS, lors du clic sur le bouton

            # Sauvegarde les fichiers sur le disque
            count = save_uploaded_files(uploaded_files)
            st.info(f"💾 {count} fichier(s) sauvegardé(s)")
            # ↑ st.info() : message bleu d'information — équivalent toastr "info" Angular

            # Lance le pipeline d'ingestion complet
            success = run_ingestion()

            if success:
                st.success("🎉 Base vectorielle prête !")

                # ─────────────────────────────────────────────────────
                # MICRO-COURS : st.session_state — persistance entre rechargements
                #
                # PROBLÈME : Streamlit re-exécute tout le script à chaque clic.
                # Si on veut qu'une valeur PERSISTE d'un rechargement à l'autre
                # (ex: "l'indexation est terminée"), il faut la stocker dans
                # st.session_state.
                #
                # En Java/Spring : HttpSession → session.setAttribute("indexed", true)
                # En Angular     : BehaviorSubject<boolean> dans un service
                # En Streamlit   : st.session_state["indexed"] = True
                #
                # st.session_state est un dict spécial qui SURVIT aux rechargements.
                # Variables normales Python → perdues à chaque rechargement.
                # Variables dans st.session_state → conservées pendant toute la session.
                #
                # Exemple simple :
                #   st.session_state["compteur"] = 0     # initialisation
                #   st.session_state["compteur"] += 1    # incrémentation persistante
                # ─────────────────────────────────────────────────────
                # Marque la session comme indexée pour débloquer la zone de question
                st.session_state["indexed"] = True   # Persiste entre les rechargements
            else:
                st.session_state["indexed"] = False

    st.divider()

    # ─────────────────────────────────────────────────────────────────
    # Indicateur d'état de la base vectorielle
    #
    # os.path.exists("chroma_db") : vérifie que le dossier existe
    # os.listdir("chroma_db")     : vérifie qu'il n'est pas vide
    #   os.listdir() retourne une liste des fichiers dans le dossier.
    #   Une liste non vide est truthy → "and" garantit que c'est non vide.
    #
    # En Java :
    #   Files.exists(Path.of("chroma_db")) &&
    #   new File("chroma_db").list().length > 0
    # ─────────────────────────────────────────────────────────────────
    # Indicateur d'état de la base vectorielle
    if os.path.exists("chroma_db") and os.listdir("chroma_db"):
        st.success("🟢 Base vectorielle active")
    else:
        st.warning("🔴 Aucune base vectorielle — indexe d'abord tes documents")
        # ↑ st.warning() : message orange d'avertissement — équivalent toastr "warning" Angular


# ---------------------------------------------------------------------------
# ZONE PRINCIPALE — Question et réponse
# ---------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────
# Vérifie si une base vectorielle exploitable existe sur le disque
# (que l'indexation ait été faite dans cette session OU dans une précédente)
# ─────────────────────────────────────────────────────────────────────
# Vérifie si une base vectorielle existe (indexation faite au moins une fois)
db_ready = os.path.exists("chroma_db") and os.listdir("chroma_db")
# ↑ Variable locale Python — perdue à chaque rechargement,
#   mais recalculée à chaque rechargement depuis le disque (os.path.exists).
#   Pas besoin de session_state ici car c'est lu depuis le système de fichiers.

if not db_ready:
    # Message d'accueil si aucune base n'existe encore
    st.info("👈 Commence par uploader tes PDFs dans le panneau de gauche et clique sur **Indexer les documents**.")

else:
    # ─────────────────────────────────────────────────────────────────
    # Zone de question — active uniquement si la base est prête
    # ─────────────────────────────────────────────────────────────────
    st.markdown("### ❓ Pose ta question")

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : st.text_input() — champ de saisie de texte
    #
    # En Angular : <mat-form-field><input matInput [(ngModel)]="question"></mat-form-field>
    # En HTML    : <input type="text" placeholder="...">
    #
    # Retourne la valeur actuelle du champ (string vide "" si rien saisi).
    # Paramètres :
    #   label              : label du champ (affiché au-dessus par défaut)
    #   placeholder        : texte gris dans le champ vide
    #   label_visibility   : "collapsed" = cache le label (juste le placeholder)
    # ─────────────────────────────────────────────────────────────────
    question = st.text_input(
        label="Question",
        placeholder="Ex: What are the key principles of ICH Q10 quality system?",
        label_visibility="collapsed"  # Cache le label "Question" — le placeholder suffit
    )

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : st.columns() — mise en page en colonnes
    #
    # En Angular Material : <mat-grid-list cols="6">
    # En CSS              : display: grid; grid-template-columns: 1fr 5fr;
    #
    # st.columns([1, 5]) : crée 2 colonnes avec ratio 1:5
    #   Colonne 1 : 1/6 de la largeur totale (pour le bouton)
    #   Colonne 2 : 5/6 de la largeur totale (espace vide ici)
    #
    # Retourne des objets "colonnes" — on les déstructure comme un tuple.
    # Chaque colonne est un context manager : "with col1:" pour y placer des widgets.
    # ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 5])  # Déstructuration de tuple (pattern déjà vu)
    with col1:
        search_button = st.button("🔍 Rechercher", type="primary", use_container_width=True)
        # ↑ Le bouton est placé dans col1 (1/6 de largeur)
        # col2 reste vide — sert juste à pousser le bouton vers la gauche

    # ─────────────────────────────────────────────────────────────────
    # Logique de déclenchement de la recherche RAG
    #
    # Deux conditions pour lancer la recherche :
    #   1. search_button : le bouton a été cliqué (True au clic)
    #   2. question.strip() : la question n'est pas vide ni que des espaces
    #
    # L'opérateur "and" court-circuite :
    #   Si search_button est False → question.strip() n'est pas évalué.
    # ─────────────────────────────────────────────────────────────────
    # Déclenche la recherche RAG
    if search_button and question.strip():
        with st.spinner("🔄 Recherche en cours..."):
            try:
                result = ask(question)  # Appelle le pipeline RAG complet (src/rag.py)

                # Affiche la réponse principale
                st.markdown("### 💬 Réponse")
                st.markdown(result["answer"])  # Gemini peut répondre en Markdown → st.markdown() le rend

                st.divider()

                # Affiche les chunks sources en accordéon
                display_sources(result["chunks"])

            except Exception as e:
                st.error(f"Erreur : {str(e)}")

    elif search_button and not question.strip():
        # L'utilisateur a cliqué "Rechercher" sans rien saisir
        st.warning("⚠️ Saisis une question avant de rechercher.")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONCLUSION DU FICHIER                                               ║
# ║                                                                      ║
# ║  🎯 CONCEPTS PYTHON CLÉS VUS ICI                                     ║
# ║                                                                      ║
# ║  1. Modèle d'exécution Streamlit : tout le script re-tourne          ║
# ║     à chaque interaction — pas de routes ni de méthodes d'événements ║
# ║     → l'ordre du code = l'ordre d'affichage                         ║
# ║                                                                      ║
# ║  2. st.session_state["clé"] = valeur                                 ║
# ║     → la seule façon de persister des données entre rechargements    ║
# ║     → équivalent HttpSession Java / BehaviorSubject Angular          ║
# ║                                                                      ║
# ║  3. count += 1 → Python n'a PAS ++ ni --                            ║
# ║     → toujours += 1 ou -= 1                                         ║
# ║                                                                      ║
# ║  4. "wb" pour les fichiers binaires (PDFs, images)                  ║
# ║     "w" + encoding="utf-8" pour les fichiers texte                  ║
# ║                                                                      ║
# ║  5. shutil.rmtree("dossier") → supprime dossier + contenu           ║
# ║     os.makedirs("dossier", exist_ok=True) → recrée                  ║
# ║                                                                      ║
# ║  ⚠️  PIÈGES À ÉVITER                                                 ║
# ║                                                                      ║
# ║  - st.set_page_config() DOIT être le PREMIER appel Streamlit        ║
# ║  - st.button() retourne True SEULEMENT au rechargement du clic      ║
# ║    → le code dans "if st.button():" s'exécute une seule fois        ║
# ║  - Variables Python normales = perdues à chaque rechargement         ║
# ║    → utiliser st.session_state pour la persistance                  ║
# ║  - shutil.rmtree() est DESTRUCTIF — pas de corbeille, pas de retour ║
# ║    → toujours vérifier os.path.exists() avant                       ║
# ║                                                                      ║
# ║  🔗 CONNEXION AVEC L'ARCHITECTURE                                    ║
# ║                                                                      ║
# ║  Ce fichier est l'UI du pipeline RAG "simple" :                    ║
# ║    ← src/ingest.py : load_pdfs, chunk_pages, embed_and_index        ║
# ║    ← src/rag.py    : ask()                                          ║
# ║  app.py     → pipeline RAG  : 1 question → 1 réponse               ║
# ║  agent_app.py → agent ReAct : 1 mission → N étapes → 1 livrable    ║
# ╚══════════════════════════════════════════════════════════════════════╝
