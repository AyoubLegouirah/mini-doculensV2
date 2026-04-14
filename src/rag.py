# src/rag.py
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FICHE D'IDENTITÉ DU FICHIER                                         ║
# ║                                                                      ║
# ║  Rôle    : Cœur du pipeline RAG (Retrieval-Augmented Generation).   ║
# ║            Orchestration des 3 phases :                             ║
# ║              R → Retrieval   : cherche les chunks pertinents        ║
# ║              A → Augmentation: formate le contexte pour le LLM      ║
# ║              G → Generation  : envoie à Gemini, retourne la réponse ║
# ║                                                                      ║
# ║  En Java : @Service principal — équivalent d'un                     ║
# ║            QuestionAnsweringService qui coordonne :                 ║
# ║              VectorStoreRepository (ChromaDB)                       ║
# ║              + LLMClient (Gemini API)                               ║
# ║              + PromptBuilder (src/prompts.py)                       ║
# ║                                                                      ║
# ║  Flux    :                                                           ║
# ║    question (str)                                                   ║
# ║      → load_vectorstore()  → base ChromaDB chargée depuis disque   ║
# ║      → retrieve_chunks()   → 5 passages les plus pertinents        ║
# ║      → build_context()     → texte formaté avec sources/pages      ║
# ║      → generate_answer()   → réponse Gemini citant ses sources     ║
# ║      → dict {answer, chunks, question}                             ║
# ║                                                                      ║
# ║  Dépendances :                                                       ║
# ║    ← src/prompts.py  (RAG_SYSTEM_PROMPT, RAG_HUMAN_PROMPT)         ║
# ║    ← chroma_db/      (créé par src/ingest.py)                      ║
# ║    → app.py          (appelle ask() depuis l'UI Streamlit)          ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════
# BLOC 1 : IMPORTS
#
# Remarque par rapport à ingest.py :
#   On retrouve les mêmes bibliothèques pour HuggingFaceEmbeddings
#   et Chroma — mais cette fois on les utilise pour LIRE la base,
#   pas pour l'écrire.
#
# Nouvelle import : google.generativeai
#   C'est le SDK Python officiel de Google pour l'API Gemini.
#   "import X as Y" crée un alias : au lieu d'écrire
#   "google.generativeai.configure()", on écrit "genai.configure()".
#
# MICRO-COURS : import X as Y — l'alias d'import
#
#   En Java tu importes le nom complet, pas d'alias natif :
#     import com.google.generativeai.GenerativeModel; // nom long
#
#   En Python, "as" crée un raccourci :
#     import google.generativeai as genai
#     genai.configure(...)  ← raccourci lisible
#
#   Convention : numpy → np, pandas → pd, generativeai → genai
#   Ces alias sont standardisés dans la communauté Python.
# ═══════════════════════════════════════════════════════════════════════

import os                                          # Opérations système : variables d'env, chemins
from dotenv import load_dotenv                     # Charge le fichier .env
import google.generativeai as genai                # SDK Gemini (alias "genai" = convention)
from langchain_huggingface import HuggingFaceEmbeddings    # Modèle d'embedding (même que ingest.py)
from langchain_community.vectorstores import Chroma        # Base vectorielle ChromaDB
from src.prompts import RAG_SYSTEM_PROMPT, RAG_HUMAN_PROMPT
# ↑ MICRO-COURS : Import depuis un module du projet (src/prompts.py)
#
#   En Java : import com.pharmaco.prompts.Prompts;
#   En Python : from src.prompts import RAG_SYSTEM_PROMPT, RAG_HUMAN_PROMPT
#
#   "src.prompts" = le fichier src/prompts.py
#   Les points "." remplacent les "/" dans les chemins de packages.
#   On importe directement les constantes — pas besoin de préfixe ensuite.
#   Après cet import : RAG_SYSTEM_PROMPT est utilisable directement (sans "Prompts.")


# ═══════════════════════════════════════════════════════════════════════
# BLOC 2 : CHARGEMENT DE L'ENVIRONNEMENT ET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

load_dotenv()  # Charge .env → rend GEMINI_API_KEY accessible via os.getenv()

CHROMA_DB_DIR   = "chroma_db"                                                    # Dossier créé par ingest.py
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # DOIT être le même modèle qu'à l'ingestion
TOP_K_RESULTS   = 5  # Nombre de chunks retournés par la recherche vectorielle
# ↑ POURQUOI TOP_K = 5 ?
#   Trop peu (1-2) : risque de manquer des infos importantes.
#   Trop (20+) : le contexte envoyé au LLM devient immense et coûteux.
#   5 est un bon compromis : assez de contexte, token budget raisonnable.


# ═══════════════════════════════════════════════════════════════════════
# BLOC 3 : FONCTION load_vectorstore — Charger ChromaDB depuis le disque
#
# Rôle : Vérifie que la base existe, charge le modèle d'embedding,
#        ouvre la base ChromaDB en lecture.
#
# IMPORTANT : Cette fonction ne CRÉE pas la base — elle la LIT.
#   Ingest.py crée la base (Chroma.from_texts(..., persist_directory=...))
#   Rag.py relit la base (Chroma(persist_directory=..., embedding_function=...))
# ═══════════════════════════════════════════════════════════════════════

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
    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : Lever une exception en Python (raise)
    #
    # En Java :
    #   if (!Files.exists(Path.of(CHROMA_DB_DIR))) {
    #       throw new FileNotFoundException("Base introuvable...");
    #   }
    #
    # En Python :
    #   if not os.path.exists(CHROMA_DB_DIR):
    #       raise FileNotFoundError("Base introuvable...")
    #
    # Points clés :
    #   - "raise" = "throw" en Java
    #   - FileNotFoundError est une exception built-in Python
    #   - On passe le message directement au constructeur
    #   - La f-string permet un message d'erreur dynamique et lisible
    #
    # Autres exceptions Python courantes :
    #   ValueError       → IllegalArgumentException Java
    #   TypeError        → ClassCastException Java
    #   KeyError         → NoSuchElementException Java (pour les dicts)
    #   AttributeError   → NullPointerException Java (accès attribut sur None)
    #   RuntimeError     → RuntimeException Java
    # ─────────────────────────────────────────────────────────────────
    if not os.path.exists(CHROMA_DB_DIR):  # En Java : if (!Files.exists(Path.of(CHROMA_DB_DIR)))
        raise FileNotFoundError(           # En Java : throw new FileNotFoundException(...)
            f"Base vectorielle introuvable : {CHROMA_DB_DIR}/\n"
            "Lance d'abord : python -m src.ingest"
        )

    print(f"🔄 Chargement du modèle d'embedding : {EMBEDDING_MODEL}")

    # ─────────────────────────────────────────────────────────────────
    # POURQUOI recharger HuggingFaceEmbeddings ici (déjà dans ingest.py) ?
    #
    # ChromaDB a besoin du modèle d'embedding pour :
    #   1. Convertir la QUESTION de l'utilisateur en vecteur
    #   2. Comparer ce vecteur avec ceux stockés (similarité cosinus)
    #
    # C'est le même modèle qu'à l'ingestion — OBLIGATOIRE.
    # Utiliser un modèle différent donnerait des résultats absurdes
    # (les espaces vectoriels seraient incompatibles).
    # ─────────────────────────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="./models"           # Réutilise le cache téléchargé par ingest.py
    )

    # ─────────────────────────────────────────────────────────────────
    # Chroma(...) en mode LECTURE — différent de Chroma.from_texts() en ingest.py
    #
    # ingest.py  : Chroma.from_texts(texts=..., persist_directory=...) → CRÉE + ÉCRIT
    # rag.py     : Chroma(persist_directory=..., embedding_function=...) → LIT SEULEMENT
    #
    # Analogie Java :
    #   ingest.py → EntityManager.persist(entity)  → INSERT en DB
    #   rag.py    → EntityManager.find(id)          → SELECT depuis la DB
    # ─────────────────────────────────────────────────────────────────
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,   # Chemin de la base sur disque
        embedding_function=embeddings      # Modèle pour vectoriser les questions
    )

    print("✅ Base vectorielle chargée")
    return vectorstore  # Retourne l'objet Chroma prêt à être interrogé


# ═══════════════════════════════════════════════════════════════════════
# BLOC 4 : FONCTION retrieve_chunks — Phase R du RAG
#
# Rôle : Convertit la question en vecteur, cherche les passages
#        les plus proches dans ChromaDB (recherche sémantique).
#
# C'est la phase "Retrieval" du RAG : trouver les bons documents.
# ═══════════════════════════════════════════════════════════════════════

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
    # ─────────────────────────────────────────────────────────────────
    # similarity_search(question, k=5) :
    #   1. Transforme "question" en vecteur via HuggingFaceEmbeddings
    #   2. Calcule la distance cosinus entre ce vecteur et TOUS les chunks
    #   3. Retourne les k=5 chunks avec la distance la plus faible
    #      (= les plus similaires sémantiquement à la question)
    #
    # Résultat : liste d'objets Document LangChain
    #   doc.page_content → le texte du chunk (str)
    #   doc.metadata     → dict {"source": "file.pdf", "page": 3}
    # ─────────────────────────────────────────────────────────────────
    results = vectorstore.similarity_search(question, k=TOP_K_RESULTS)
    # ↑ "k=TOP_K_RESULTS" = argument nommé — k est le nombre de résultats à retourner

    chunks = []
    for doc in results:
        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : dict.get("key", valeur_par_défaut)
        #
        # En Java : map.getOrDefault("source", "inconnu")
        # En Python : dict.get("source", "inconnu")
        #
        # Différence avec dict["source"] :
        #   dict["source"]              → KeyError si "source" absent (comme map.get() qui retourne null)
        #   dict.get("source", "inconnu") → retourne "inconnu" si absent (sécurisé)
        #
        # PIÈGE Java → Python :
        #   En Java, map.get("key") retourne null si absent → tu testes != null
        #   En Python, dict["key"] LÈVE une exception (KeyError) si absent
        #   → Toujours préférer dict.get("key", default) pour un accès sécurisé
        # ─────────────────────────────────────────────────────────────
        chunks.append({
            "text":   doc.page_content,                      # Texte du chunk (attribut LangChain Document)
            "source": doc.metadata.get("source", "inconnu"), # Nom du PDF (ou "inconnu" si métadonnée manquante)
            "page":   doc.metadata.get("page", "?")          # Numéro de page (ou "?" si absent)
        })

    print(f"✅ {len(chunks)} chunks récupérés depuis ChromaDB")
    return chunks


# ═══════════════════════════════════════════════════════════════════════
# BLOC 5 : FONCTION build_context — Phase A du RAG
#
# Rôle : Transforme la liste de chunks en un bloc de texte formaté
#        que le LLM peut lire et citer.
#
# C'est la phase "Augmentation" du RAG : préparer le contexte.
#
# Exemple de sortie :
#   [Document 1 - sop_nettoyage.pdf, Page 3]
#   Le nettoyage des équipements doit être effectué...
#
#   [Document 2 - ich_q10.pdf, Page 7]
#   Les systèmes de qualité pharmaceutique exigent...
# ═══════════════════════════════════════════════════════════════════════

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
    context_parts = []  # Liste des parties de contexte formatées (une par chunk)

    # ─────────────────────────────────────────────────────────────────
    # enumerate(chunks, start=1) → déjà vu dans ingest.py
    # Ici on l'utilise pour numéroter les documents : Document 1, 2, 3...
    # ─────────────────────────────────────────────────────────────────
    for i, chunk in enumerate(chunks, start=1):
        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : Accès à un dict dans une f-string
        #
        # En Java :
        #   String.format("[Document %d - %s, Page %s]\n%s",
        #       i, chunk.get("source"), chunk.get("page"), chunk.get("text"))
        #
        # En Python :
        #   f"[Document {i} - {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        #
        # ATTENTION : dans une f-string avec guillemets doubles "...",
        # les clés dict DOIVENT utiliser des guillemets SIMPLES '...'
        # pour éviter les conflits : chunk['source'] et non chunk["source"]
        #
        # MICRO-COURS : Concaténation implicite de strings littérales
        #
        # Python permet de coller deux strings entre parenthèses :
        #   context_parts.append(
        #       f"[Document {i}...]\n"   ← string 1
        #       f"{chunk['text']}"        ← string 2 (collée automatiquement)
        #   )
        # C'est équivalent à :
        #   context_parts.append(f"[Document {i}...]\n" + f"{chunk['text']}")
        # Utile pour la lisibilité sur plusieurs lignes.
        # ─────────────────────────────────────────────────────────────
        context_parts.append(
            f"[Document {i} - {chunk['source']}, Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : str.join(iterable) — joindre une liste en string
    #
    # En Java :
    #   String.join("\n\n", contextParts)
    #   // ou : contextParts.stream().collect(Collectors.joining("\n\n"))
    #
    # En Python :
    #   "\n\n".join(context_parts)
    #
    # ATTENTION : la syntaxe Python est "inversée" par rapport à Java.
    #   Java   : String.join(séparateur, liste)  ← séparateur en premier
    #   Python : séparateur.join(liste)          ← le séparateur est l'objet
    #
    # Exemple simple :
    #   mots = ["bonjour", "monde", "python"]
    #   " | ".join(mots)   → "bonjour | monde | python"
    #   ", ".join(mots)    → "bonjour, monde, python"
    #   "".join(mots)      → "bonjourmondepython"
    #
    # "\n\n" = double saut de ligne → sépare visuellement les documents
    # ─────────────────────────────────────────────────────────────────
    return "\n\n".join(context_parts)
    # ↑ Retourne UN SEUL string avec tous les chunks séparés par des lignes vides


# ═══════════════════════════════════════════════════════════════════════
# BLOC 6 : FONCTION generate_answer — Phase G du RAG
#
# Rôle : Envoie la question + le contexte à Google Gemini et retourne
#        la réponse générée.
#
# C'est la phase "Generation" du RAG : le LLM synthétise une réponse
# à partir des passages récupérés.
#
# FLUX DÉTAILLÉ :
#   1. Lit GEMINI_API_KEY depuis les variables d'environnement
#   2. Configure le client genai avec cette clé
#   3. Crée un modèle Gemini avec RAG_SYSTEM_PROMPT (le "contrat de travail")
#   4. Formate RAG_HUMAN_PROMPT avec {context} et {question}
#   5. Envoie le prompt à Gemini et retourne la réponse texte
# ═══════════════════════════════════════════════════════════════════════

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
    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : os.getenv() — Lire une variable d'environnement
    #
    # En Java :
    #   String apiKey = System.getenv("GEMINI_API_KEY");
    #   if (apiKey == null) throw new IllegalArgumentException("...");
    #
    # En Python :
    #   api_key = os.getenv("GEMINI_API_KEY")
    #   if not api_key:
    #       raise ValueError("...")
    #
    # os.getenv("VAR") retourne None si la variable n'existe pas.
    # "if not api_key:" est vrai si api_key est None OU une string vide "".
    # C'est plus robuste que "if api_key is None:" car ça couvre les deux cas.
    # ─────────────────────────────────────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY")  # En Java : System.getenv("GEMINI_API_KEY")
    if not api_key:                         # Vrai si None ou chaîne vide ""
        raise ValueError(                  # En Java : throw new IllegalArgumentException(...)
            "GEMINI_API_KEY introuvable.\n"
            "Vérifie que ton fichier .env contient : GEMINI_API_KEY=ta_clé"
        )

    # ─────────────────────────────────────────────────────────────────
    # genai.configure(api_key=...) :
    #   Configure le client Gemini au niveau global du module.
    #   Après cet appel, tous les modèles Gemini créés utiliseront cette clé.
    #
    # Analogie Java :
    #   Comme configurer un RestTemplate ou WebClient globalement
    #   avec un intercepteur d'authentification.
    # ─────────────────────────────────────────────────────────────────
    genai.configure(api_key=api_key)  # Injecte la clé API dans le client Gemini

    # ─────────────────────────────────────────────────────────────────
    # genai.GenerativeModel(...) :
    #   Crée une instance du modèle Gemini avec sa configuration.
    #
    # model_name="gemini-2.5-flash" : version du modèle (rapide et économique)
    # system_instruction=RAG_SYSTEM_PROMPT : le "contrat de travail" permanent
    #   défini dans src/prompts.py — le modèle le reçoit avant chaque question
    #
    # Analogie Java :
    #   new GeminiClient.Builder()
    #       .modelName("gemini-2.5-flash")
    #       .systemInstruction(RAG_SYSTEM_PROMPT)
    #       .build();
    # ─────────────────────────────────────────────────────────────────
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",        # Modèle Gemini rapide et économique
        system_instruction=RAG_SYSTEM_PROMPT  # Import de src/prompts.py — règles pharma
    )

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : str.format() — Remplir les placeholders d'un template
    #
    # RAG_HUMAN_PROMPT contient : "...{context}...\nQuestion : {question}..."
    # .format(context=..., question=...) remplace les {} par les vraies valeurs.
    #
    # En Java :
    #   String prompt = RAG_HUMAN_PROMPT
    #       .replace("{context}", context)
    #       .replace("{question}", question);
    #   // ou plus proprement :
    #   MessageFormat.format(RAG_HUMAN_PROMPT, context, question);
    #
    # En Python :
    #   prompt = RAG_HUMAN_PROMPT.format(context=context, question=question)
    #
    # DIFFÉRENCE AVEC LES F-STRINGS :
    #   f"...{context}..."           → évaluation IMMÉDIATE au moment de l'écriture
    #   "...{context}...".format()   → évaluation DIFFÉRÉE, appliquée à la demande
    #
    # Ici on utilise .format() car RAG_HUMAN_PROMPT est défini ailleurs
    # (dans prompts.py) — on ne peut pas faire une f-string sur une variable externe.
    # ─────────────────────────────────────────────────────────────────
    prompt = RAG_HUMAN_PROMPT.format(
        context=context,    # Remplace {context} par les chunks formatés
        question=question   # Remplace {question} par la vraie question
    )

    print("🔄 Génération de la réponse via Gemini...")
    response = model.generate_content(prompt)  # Envoi à l'API Gemini et attente de la réponse
    # ↑ generate_content() est un appel SYNCHRONE (bloquant) — il attend la réponse
    #   Analogie Java : restTemplate.postForObject(url, request, Response.class)

    return response.text  # .text = le contenu textuel de la réponse Gemini


# ═══════════════════════════════════════════════════════════════════════
# BLOC 7 : FONCTION ask — Façade / Point d'entrée unique du pipeline RAG
#
# Rôle : Orchestre les 4 fonctions précédentes dans l'ordre.
#        C'est la SEULE fonction publique que les autres fichiers utilisent.
#
# MICRO-COURS : Pattern Facade en Python
#
#   En Java, tu reconnaîtrais le Pattern Facade :
#     Une classe expose une interface simplifiée qui cache la complexité
#     de plusieurs sous-systèmes.
#
#   Ici, ask() expose UNE seule méthode simple :
#     result = ask("ta question")
#
#   En interne, elle orchestre :
#     load_vectorstore() → retrieve_chunks() → build_context() → generate_answer()
#
#   app.py (l'UI Streamlit) n'a pas besoin de connaître ces détails.
#   Il appelle juste ask() et reçoit {answer, chunks, question}.
# ═══════════════════════════════════════════════════════════════════════

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

    # Phase 1 — Chargement de la base vectorielle (lecture de chroma_db/)
    vectorstore = load_vectorstore()

    # Phase 2 — Retrieval des chunks pertinents (recherche sémantique)
    chunks = retrieve_chunks(vectorstore, question)

    # Phase 3 — Construction du contexte (formatage pour le LLM)
    context = build_context(chunks)

    # Phase 4 — Génération de la réponse (appel Gemini API)
    answer = generate_answer(question, context)

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : Retourner un dict comme résultat structuré
    #
    # En Java, tu créerais un objet de résultat :
    #   record RagResult(String answer, List<Map<String,Object>> chunks, String question) {}
    #   return new RagResult(answer, chunks, question);
    #
    # En Python, un dict suffit pour des résultats structurés simples :
    #   return {"answer": answer, "chunks": chunks, "question": question}
    #
    # L'appelant accède aux champs par clé : result["answer"]
    # C'est idiomatique Python — pas besoin de créer une classe pour tout.
    # ─────────────────────────────────────────────────────────────────
    return {
        "answer":   answer,    # Réponse textuelle de Gemini
        "chunks":   chunks,    # Liste des 5 chunks utilisés (pour afficher les sources)
        "question": question   # Question originale (pour affichage dans l'UI)
    }


# ═══════════════════════════════════════════════════════════════════════
# BLOC 8 : POINT D'ENTRÉE POUR LES TESTS EN LIGNE DE COMMANDE
#
# Permet de tester le pipeline RAG directement : python src/rag.py
# Sans passer par l'UI Streamlit.
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    question_test = "What are the key principles of pharmaceutical quality systems?"
    result = ask(question_test)

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : Répétition de chaîne avec * (opérateur de répétition)
    #
    # En Java : "=".repeat(60)    → "============...=" (60 fois)
    # En Python : "=" * 60        → "============...=" (60 fois)
    #
    # L'opérateur * appliqué à une string répète celle-ci N fois.
    # Aussi utilisable avec des listes : [0] * 5 → [0, 0, 0, 0, 0]
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)  # "="*60 = 60 symboles "=" collés → ligne de séparation visuelle
    print("RÉPONSE :")
    print("="*60)
    print(result["answer"])   # Accès à la valeur du dict par clé

    print("\n" + "="*60)
    print("SOURCES UTILISÉES :")
    print("="*60)
    for chunk in result["chunks"]:
        print(f"  📄 {chunk['source']} — Page {chunk['page']}")
        # ↑ chunk['source'] avec guillemets simples à l'intérieur de la f-string


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONCLUSION DU FICHIER                                               ║
# ║                                                                      ║
# ║  🎯 CONCEPTS PYTHON CLÉS VUS ICI                                     ║
# ║                                                                      ║
# ║  1. import X as Y → alias d'import (genai, np, pd = conventions)   ║
# ║                                                                      ║
# ║  2. raise Exception("msg") → throw new Exception Java               ║
# ║     Exceptions courantes : ValueError, FileNotFoundError, KeyError  ║
# ║                                                                      ║
# ║  3. dict.get("key", default) → getOrDefault() Java                  ║
# ║     Toujours préférer à dict["key"] pour les accès potentiellement  ║
# ║     absents — évite les KeyError                                    ║
# ║                                                                      ║
# ║  4. "sep".join(liste) → String.join(sep, liste) Java                ║
# ║     ATTENTION : syntaxe inversée vs Java !                          ║
# ║                                                                      ║
# ║  5. str.format(key=val) → remplacement différé des {placeholders}   ║
# ║     ≠ f-string (immédiat) — utilisé pour les templates externes     ║
# ║                                                                      ║
# ║  ⚠️  PIÈGES À ÉVITER                                                 ║
# ║                                                                      ║
# ║  - dict["key"] lève KeyError si absent → utiliser dict.get()       ║
# ║  - "sep".join(liste) et non join(sep, liste) comme en Java          ║
# ║  - Dans une f-string "...", les clés dict avec guillemets simples   ║
# ║    : chunk['source'] et non chunk["source"] (conflit de quotes)     ║
# ║  - os.getenv() retourne None (pas exception) si variable absente    ║
# ║                                                                      ║
# ║  🔗 CONNEXION AVEC L'ARCHITECTURE                                    ║
# ║                                                                      ║
# ║  Ce fichier est le pivot de l'architecture :                        ║
# ║    ← src/prompts.py  : fournit les templates de prompts             ║
# ║    ← chroma_db/      : base vectorielle créée par src/ingest.py     ║
# ║    → app.py          : UI Streamlit appelle ask() pour chaque       ║
# ║                        question de l'utilisateur                    ║
# ║    → agent_app.py    : l'agent ReAct utilise aussi ce pipeline      ║
# ╚══════════════════════════════════════════════════════════════════════╝
