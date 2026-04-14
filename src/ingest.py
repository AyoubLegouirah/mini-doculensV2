# src/ingest.py
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FICHE D'IDENTITÉ DU FICHIER                                         ║
# ║                                                                      ║
# ║  Rôle    : Pipeline d'ingestion des documents PDF.                  ║
# ║            Ce script s'exécute UNE SEULE FOIS avant de lancer       ║
# ║            l'application. Il lit les PDFs, les découpe en morceaux  ║
# ║            (chunks), les transforme en vecteurs numériques          ║
# ║            (embeddings) et les stocke dans ChromaDB.                ║
# ║                                                                      ║
# ║  En Java : Équivalent d'un @Service Spring Batch qui prépare une    ║
# ║            base de données au démarrage. Ou un script Flyway/       ║
# ║            Liquibase, mais pour une base vectorielle.               ║
# ║                                                                      ║
# ║  Flux    : documents/*.pdf                                          ║
# ║              → load_pdfs()     → liste de pages (texte + métadata) ║
# ║              → chunk_pages()   → morceaux de 400 caractères        ║
# ║              → embed_and_index() → vecteurs stockés dans chroma_db/║
# ║                                                                      ║
# ║  Usage   : python src/ingest.py   (lancé en ligne de commande)     ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════
# BLOC 1 : IMPORTS — Charger les bibliothèques nécessaires
#
# MICRO-COURS : import vs from X import Y
#
# En Java, tu écrirais toujours :
#   import com.example.service.MyService;
# Python a deux formes :
#
#   Forme 1 — import complet :
#     import os
#     os.listdir(".")     ← tu dois préfixer avec "os."
#
#   Forme 2 — import partiel (only what you need) :
#     from os import listdir
#     listdir(".")         ← pas besoin de préfixe
#
# Analogie Java : comme si tu pouvais écrire :
#   import static com.example.Utils.listdir;
#   listdir(".");   ← sans préfixe
#
# Convention Python : on préfère "import os" et "os.listdir()"
# pour la lisibilité — on sait d'où vient la fonction.
# Mais "from X import Y" est courant pour des noms longs (LangChain).
# ═══════════════════════════════════════════════════════════════════════

import os                                          # Module standard : opérations système (fichiers, chemins, env vars)
from dotenv import load_dotenv                     # Charge les variables depuis le fichier .env
import pdfplumber                                  # Bibliothèque pour lire et extraire le texte des PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Découpe le texte en chunks intelligemment
from langchain_huggingface import HuggingFaceEmbeddings             # Transforme du texte en vecteurs numériques
from langchain_community.vectorstores import Chroma                 # Base de données vectorielle locale (ChromaDB)


# ═══════════════════════════════════════════════════════════════════════
# BLOC 2 : CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
#
# load_dotenv() lit le fichier .env à la racine du projet et injecte
# son contenu dans les variables d'environnement du processus.
#
# Exemple de fichier .env :
#   GOOGLE_API_KEY=AIzaSyXXXXXX
#
# MICRO-COURS : Variables d'environnement Python vs Spring
#
# En Spring Boot, tu utilises :
#   @Value("${google.api.key}")
#   private String apiKey;
#
# En Python, après load_dotenv() :
#   import os
#   api_key = os.getenv("GOOGLE_API_KEY")   ← équivalent de @Value
#
# POURQUOI load_dotenv() EST APPELÉ AU NIVEAU DU MODULE ?
#   En Java, @Value est traité au démarrage du contexte Spring.
#   Ici, load_dotenv() s'exécute dès que ce fichier est importé
#   ou lancé — avant toute fonction. C'est le comportement voulu.
# ═══════════════════════════════════════════════════════════════════════

load_dotenv()  # Charge .env → les clés API sont disponibles via os.getenv()


# ═══════════════════════════════════════════════════════════════════════
# BLOC 3 : CONSTANTES DE CONFIGURATION
#
# En Java tu écrirais une classe de configuration :
#   @Configuration
#   public class IngestConfig {
#       public static final String DOCUMENTS_DIR = "documents";
#       public static final int CHUNK_SIZE = 400;
#       ...
#   }
#
# En Python, les constantes sont simplement déclarées en MAJUSCULES
# au niveau du module (fichier). Pas besoin de classe conteneur.
#
# Ces valeurs paramètrent le pipeline d'ingestion :
#   DOCUMENTS_DIR   : dossier où se trouvent les PDFs à ingérer
#   CHROMA_DB_DIR   : dossier où ChromaDB va stocker sa base vectorielle
#   EMBEDDING_MODEL : nom du modèle HuggingFace pour créer les vecteurs
#   CHUNK_SIZE      : taille max d'un chunk (en nombre de caractères)
#   CHUNK_OVERLAP   : chevauchement entre deux chunks consécutifs
#                     (important pour ne pas perdre le contexte en coupure)
# ═══════════════════════════════════════════════════════════════════════

DOCUMENTS_DIR   = "documents"                                                       # Dossier source des PDFs
CHROMA_DB_DIR   = "chroma_db"                                                       # Dossier destination pour ChromaDB
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"    # Modèle multilingue (FR/EN/...)
CHUNK_SIZE      = 400    # Un chunk = 400 caractères max (compromis : assez grand pour avoir du sens, assez petit pour être précis)
CHUNK_OVERLAP   = 50     # Les 50 derniers caractères d'un chunk sont répétés au début du suivant → évite de couper une phrase


# ═══════════════════════════════════════════════════════════════════════
# BLOC 4 : FONCTION load_pdfs — Étape 1 du pipeline
#
# Rôle : Parcourt le dossier documents/ et extrait le texte
#        de chaque PDF page par page.
#
# Équivalent Java :
#   public List<Map<String, Object>> loadPdfs(String documentsDir) { ... }
# ═══════════════════════════════════════════════════════════════════════

def load_pdfs(documents_dir: str) -> list[dict]:
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ MICRO-COURS : Signature de fonction Python avec type hints      │
    # │                                                                 │
    # │ En Java :                                                       │
    # │   public List<Map<String,Object>> loadPdfs(String dir) { }     │
    # │                                                                 │
    # │ En Python :                                                     │
    # │   def load_pdfs(documents_dir: str) -> list[dict]:             │
    # │                                                                 │
    # │ Points clés :                                                   │
    # │  - "def" remplace le type de retour en préfixe                 │
    # │  - ": str" après le paramètre = type hint (OPTIONNEL)          │
    # │  - "-> list[dict]" = type de retour (OPTIONNEL)                │
    # │  - Les types hints sont informatifs, pas contraignants          │
    # │    (Python ne les vérifie pas à l'exécution — contrairement    │
    # │     à Java où le compilateur les impose)                        │
    # │                                                                 │
    # │ "list[dict]" = List<Map<String,Object>> en Java                │
    # │   list  → ArrayList                                            │
    # │   dict  → HashMap<String, Object>                              │
    # └─────────────────────────────────────────────────────────────────┘
    """
    Parcourt le dossier documents/ et extrait le texte
    de chaque PDF page par page.
    Retourne une liste de dicts {text, source, page}
    """
    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : Les listes Python (list)
    #
    # En Java tu initialises une liste ainsi :
    #   List<Map<String,Object>> pages = new ArrayList<>();
    #
    # En Python :
    #   pages = []    ← liste vide, pas de type déclaré
    #
    # Python utilise [] pour les listes (comme les tableaux Java).
    # Une liste Python peut contenir des éléments de types DIFFÉRENTS
    # (contrairement à List<T> Java qui est typée).
    # Exemple : [1, "bonjour", True, {"key": "val"}]
    # ─────────────────────────────────────────────────────────────────
    pages = []  # En Java : List<Map<String,Object>> pages = new ArrayList<>();

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : List Comprehension — la syntaxe la plus "pythonique"
    #
    # C'est l'une des features les plus emblématiques de Python.
    # Syntaxe : [expression for element in collection if condition]
    #
    # En Java (Stream API) :
    #   List<String> pdfFiles = Arrays.stream(new File(documentsDir).list())
    #       .filter(f -> f.endsWith(".pdf"))
    #       .collect(Collectors.toList());
    #
    # En Python (list comprehension) :
    #   pdf_files = [f for f in os.listdir(documents_dir) if f.endswith(".pdf")]
    #
    # Lecture à voix haute : "crée une liste de f, pour chaque f dans
    # os.listdir(documents_dir), si f se termine par '.pdf'"
    #
    # Exemple simple pour bien comprendre :
    #   nombres = [1, 2, 3, 4, 5, 6]
    #   pairs   = [n for n in nombres if n % 2 == 0]
    #   # → [2, 4, 6]
    #
    # PIÈGE Java → Python : pas d'accolades {} ni de parenthèses ()
    # pour les blocs — c'est l'indentation qui structure le code.
    # ─────────────────────────────────────────────────────────────────
    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith(".pdf")]
    # ↑ os.listdir() retourne tous les fichiers du dossier (comme File.list() en Java)
    # ↑ .endswith(".pdf") filtre pour ne garder que les PDFs (comme String.endsWith() Java)

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : Vérifier si une liste est vide en Python
    #
    # En Java : if (pdfFiles.isEmpty()) { ... }
    # En Python : if not pdf_files: ← "not" inverse la valeur booléenne
    #
    # En Python, une liste vide [] est "falsy" (équivalent à False).
    # Donc "if not pdf_files:" est vrai si la liste est vide.
    #
    # Équivalences Python :
    #   []    → False (falsy)
    #   [1]   → True  (truthy)
    #   ""    → False (falsy)
    #   "abc" → True  (truthy)
    #   0     → False (falsy)
    #   None  → False (falsy)
    # ─────────────────────────────────────────────────────────────────
    if not pdf_files:  # En Java : if (pdfFiles.isEmpty())
        print("⚠️  Aucun PDF trouvé dans le dossier documents/")
        return pages   # Retourne une liste vide [] — en Java : return new ArrayList<>();

    # ─────────────────────────────────────────────────────────────────
    # Boucle sur chaque fichier PDF trouvé
    # En Java : for (String filename : pdfFiles) { ... }
    # En Python : for filename in pdf_files:
    # ─────────────────────────────────────────────────────────────────
    for filename in pdf_files:
        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : os.path.join — construire un chemin de fichier
        #
        # En Java : Path.of(documentsDir, filename).toString()
        # En Python : os.path.join(documents_dir, filename)
        #
        # Exemple :
        #   os.path.join("documents", "sop_001.pdf")
        #   → "documents/sop_001.pdf"  (Unix/Mac)
        #   → "documents\sop_001.pdf" (Windows)
        #
        # TOUJOURS utiliser os.path.join plutôt que "dir + '/' + file"
        # car os.path.join gère automatiquement les séparateurs selon l'OS.
        # ─────────────────────────────────────────────────────────────
        filepath = os.path.join(documents_dir, filename)  # Construit "documents/sop_001.pdf"

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : f-strings — interpolation de chaînes
        #
        # En Java :
        #   System.out.println("📄 Lecture de : " + filename);
        #   // ou avec String.format :
        #   System.out.printf("📄 Lecture de : %s%n", filename);
        #
        # En Python (f-string) :
        #   print(f"📄 Lecture de : {filename}")
        #
        # Le préfixe "f" devant les guillemets active l'interpolation.
        # Tout ce qui est entre {} est évalué et converti en string.
        #
        # Exemples :
        #   nom = "Alice"
        #   age = 30
        #   print(f"Bonjour {nom}, tu as {age} ans")
        #   print(f"Dans 10 ans tu auras {age + 10} ans")  ← expression possible
        # ─────────────────────────────────────────────────────────────
        print(f"📄 Lecture de : {filename}")  # f-string : {} évalue la variable directement

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : with ... as ... (Gestionnaire de contexte)
        #
        # En Java, tu utilises try-with-resources pour fermer
        # automatiquement une ressource (fichier, connexion...) :
        #   try (InputStream is = new FileInputStream(path)) {
        #       // utiliser is
        #   }  // ← is.close() appelé automatiquement ici
        #
        # En Python, le "with" fait exactement la même chose :
        #   with pdfplumber.open(filepath) as pdf:
        #       # utiliser pdf
        #   # ← pdf.close() appelé automatiquement ici
        #
        # La variable "pdf" n'est accessible QUE dans le bloc with (indenté).
        # C'est l'équivalent Python de try-with-resources.
        # ─────────────────────────────────────────────────────────────
        with pdfplumber.open(filepath) as pdf:  # Ouvre le PDF, le ferme automatiquement à la fin du bloc
            # ─────────────────────────────────────────────────────────
            # MICRO-COURS : enumerate() — boucle avec index
            #
            # Problème classique : tu veux boucler sur une collection
            # ET avoir l'index de chaque élément.
            #
            # En Java :
            #   for (int i = 1; i <= pdf.getPages().size(); i++) {
            #       Page page = pdf.getPages().get(i - 1);
            #   }
            #
            # En Python SANS enumerate (moins pythonique) :
            #   i = 1
            #   for page in pdf.pages:
            #       print(i)
            #       i += 1
            #
            # En Python AVEC enumerate (pythonique) :
            #   for page_num, page in enumerate(pdf.pages, start=1):
            #       print(page_num)   ← commence à 1 (start=1)
            #
            # enumerate() retourne des paires (index, élément).
            # "start=1" indique que l'index commence à 1 (et non 0).
            # "page_num, page" = déstructuration de la paire (voir ci-dessous).
            # ─────────────────────────────────────────────────────────
            for page_num, page in enumerate(pdf.pages, start=1):
                # page_num : numéro de la page (commence à 1 grâce à start=1)
                # page     : objet pdfplumber représentant la page

                # ─────────────────────────────────────────────────────
                # MICRO-COURS : Déstructuration (unpacking) de tuple
                #
                # enumerate() retourne des tuples : (1, page1), (2, page2), ...
                # L'écriture "page_num, page in enumerate(...)" déstructure
                # automatiquement chaque tuple en deux variables.
                #
                # C'est comme en TypeScript/Angular :
                #   const [pageNum, page] = [1, pageObject];
                #
                # En Java il n'y a pas d'équivalent natif (avant Java 21).
                # Tu aurais dû écrire :
                #   Pair<Integer, Page> pair = ...;
                #   int pageNum = pair.getFirst();
                #   Page page   = pair.getSecond();
                # ─────────────────────────────────────────────────────
                text = page.extract_text()  # pdfplumber extrait le texte brut de la page (retourne None si page vide)

                # ─────────────────────────────────────────────────────
                # MICRO-COURS : "if text and text.strip()"
                #
                # Deux conditions enchaînées avec "and" :
                #
                # 1. "if text" : vérifie que text n'est pas None.
                #    extract_text() retourne None pour les pages sans texte (images).
                #    En Java : if (text != null)
                #
                # 2. "text.strip()" : supprime les espaces/retours à la ligne
                #    en début et fin (comme .trim() en Java).
                #    Une string de pure espaces "" est "falsy" après strip().
                #    En Java : !text.trim().isEmpty()
                #
                # L'opérateur "and" en Python est COURT-CIRCUIT :
                #   si "text" est None (falsy), Python n'évalue PAS "text.strip()"
                #   → pas de NullPointerException !
                # ─────────────────────────────────────────────────────
                if text and text.strip():  # En Java : if (text != null && !text.trim().isEmpty())
                    # ─────────────────────────────────────────────────
                    # MICRO-COURS : Dictionnaire Python {"key": value}
                    #
                    # En Java tu écrirais :
                    #   Map<String, Object> pageData = new HashMap<>();
                    #   pageData.put("text", text.trim());
                    #   pageData.put("source", filename);
                    #   pageData.put("page", pageNum);
                    #
                    # En Python, syntaxe littérale compacte :
                    #   {"text": text.strip(), "source": filename, "page": page_num}
                    #
                    # Les clés (text, source, page) sont des strings.
                    # Les valeurs peuvent être de n'importe quel type.
                    # Accès : dict["text"] ou dict.get("text")
                    # ─────────────────────────────────────────────────
                    pages.append({              # .append() = .add() en Java (ArrayList)
                        "text":   text.strip(), # Texte de la page nettoyé (strip() = trim() Java)
                        "source": filename,     # Nom du fichier PDF source
                        "page":   page_num      # Numéro de page (pour les citations)
                    })

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : len() — taille d'une collection
    #
    # En Java : pages.size()    ou  pdfFiles.length (pour les tableaux)
    # En Python : len(pages)    ou  len(pdf_files)
    #
    # len() fonctionne sur tout : list, dict, str, tuple, etc.
    # ─────────────────────────────────────────────────────────────────
    print(f"✅ {len(pages)} pages extraites depuis {len(pdf_files)} PDF(s)")
    return pages  # Retourne la liste complète des pages avec leur texte et métadonnées


# ═══════════════════════════════════════════════════════════════════════
# BLOC 5 : FONCTION chunk_pages — Étape 2 du pipeline
#
# Rôle : Découpe chaque page en "chunks" (morceaux de texte)
#        pour l'indexation vectorielle.
#
# POURQUOI DÉCOUPER EN CHUNKS ?
#   Un LLM ne peut pas traiter un document entier à chaque requête.
#   On découpe en petits morceaux de ~400 caractères pour :
#     1. Pouvoir trouver précisément le passage pertinent
#     2. Rester dans la limite de tokens du LLM
#
# POURQUOI UN CHEVAUCHEMENT (OVERLAP) ?
#   Si on coupe "...fin de la phrase A. Début de la phrAs..." en deux,
#   le contexte est perdu. Le chevauchement de 50 chars répète la fin
#   du chunk précédent au début du suivant pour préserver le sens.
# ═══════════════════════════════════════════════════════════════════════

def chunk_pages(pages: list[dict]) -> tuple[list[str], list[dict]]:
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ MICRO-COURS : tuple comme type de retour                        │
    # │                                                                 │
    # │ En Java, une méthode ne peut retourner qu'une seule valeur.     │
    # │ Pour en retourner plusieurs, tu crées un objet ou un record :   │
    # │   record ChunkResult(List<String> chunks, List<Map> metas) {}   │
    # │   return new ChunkResult(chunks, metas);                        │
    # │                                                                 │
    # │ En Python, tu retournes directement un tuple (paire, triplet):  │
    # │   return all_chunks, all_metadatas                              │
    # │   ← Python crée automatiquement un tuple (all_chunks, all_metas)│
    # │                                                                 │
    # │ L'appelant peut déstructurer :                                  │
    # │   chunks, metadatas = chunk_pages(pages)                        │
    # │   ← chaque variable reçoit sa partie du tuple                  │
    # │                                                                 │
    # │ "tuple[list[str], list[dict]]" dans la signature indique       │
    # │ qu'on retourne un tuple de 2 éléments : une liste de strings   │
    # │ et une liste de dicts.                                          │
    # └─────────────────────────────────────────────────────────────────┘
    """
    Découpe chaque page en chunks de 400 tokens
    avec 50 tokens de chevauchement.
    Retourne les textes et leurs métadonnées séparément.
    """
    # ─────────────────────────────────────────────────────────────────
    # RecursiveCharacterTextSplitter : le "splitter intelligent" de LangChain
    #
    # Il essaie de couper aux endroits naturels dans cet ordre :
    #   1. Paragraphes ("\n\n")
    #   2. Lignes ("\n")
    #   3. Phrases (". ")
    #   4. Mots (" ")
    #   5. Caractères (en dernier recours)
    #
    # Paramètres :
    #   chunk_size    : taille max d'un chunk (400 caractères)
    #   chunk_overlap : chevauchement entre chunks (50 caractères)
    #   length_function : comment mesurer la taille — ici "len" = nombre de caractères
    #                     (len est une fonction built-in Python, comme length() en Java)
    # ─────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,         # 400 caractères max par chunk
        chunk_overlap=CHUNK_OVERLAP,   # 50 caractères répétés entre chunks consécutifs
        length_function=len,           # len = fonction Python built-in qui compte les caractères
    )
    # ↑ MICRO-COURS : Arguments nommés (keyword arguments)
    #   En Java : new TextSplitter(400, 50, String::length)
    #   En Python : TextSplitter(chunk_size=400, chunk_overlap=50, length_function=len)
    #   Les "=" dans les arguments sont des NOMS, pas des assignations.
    #   Avantage : l'ordre n'a plus d'importance, et le code est auto-documenté.

    all_chunks    = []  # Contiendra les textes des chunks (strings)
    all_metadatas = []  # Contiendra les métadonnées de chaque chunk (source + page)

    # ─────────────────────────────────────────────────────────────────
    # Boucle sur chaque page extraite par load_pdfs()
    # Pour chaque page, on découpe le texte en chunks
    # et on duplique les métadonnées pour chaque chunk.
    # ─────────────────────────────────────────────────────────────────
    for page in pages:
        # page["text"] accède à la valeur associée à la clé "text" dans le dict
        # En Java : page.get("text")  ou  page.getText() si c'est un objet
        chunks = splitter.split_text(page["text"])
        # ↑ split_text() retourne une liste de strings (les morceaux)
        # Exemple : "Texte de 1200 chars" → ["chars 1-400", "chars 351-750", "chars 701-1100"]

        for chunk in chunks:
            all_chunks.append(chunk)      # Ajoute le texte du chunk à la liste principale

            # ─────────────────────────────────────────────────────────
            # POURQUOI dupliquer les métadonnées pour chaque chunk ?
            #
            # Un document de 3 pages donne par ex. 12 chunks.
            # ChromaDB a besoin de savoir, pour CHAQUE chunk :
            #   - De quel fichier il vient (source)
            #   - De quelle page (page)
            # Cela permet de citer la source précise dans la réponse.
            #
            # all_chunks[i] et all_metadatas[i] sont TOUJOURS alignés :
            # le chunk à l'index i a les métadonnées à l'index i.
            # ─────────────────────────────────────────────────────────
            all_metadatas.append({
                "source": page["source"],  # Ex: "sop_nettoyage.pdf"
                "page":   page["page"]     # Ex: 3
            })

    print(f"✅ {len(all_chunks)} chunks créés")
    return all_chunks, all_metadatas
    # ↑ MICRO-COURS : Retour multiple avec tuple
    #   Python emballe automatiquement "all_chunks, all_metadatas" dans un tuple.
    #   L'appelant écrit : chunks, metadatas = chunk_pages(pages)
    #   Python déstructure le tuple et assigne chaque partie à la bonne variable.


# ═══════════════════════════════════════════════════════════════════════
# BLOC 6 : FONCTION embed_and_index — Étape 3 du pipeline
#
# Rôle : Transforme les chunks textuels en vecteurs numériques
#        (embeddings) et les stocke dans ChromaDB pour la recherche
#        sémantique ultérieure.
#
# QU'EST-CE QU'UN EMBEDDING ?
#   C'est une représentation mathématique du "sens" d'un texte.
#   Un modèle IA convertit "le chien court" en un vecteur de 384 nombres.
#   Deux phrases au sens similaire auront des vecteurs proches dans l'espace.
#   C'est ce qui permet la recherche sémantique (par sens, pas par mot-clé).
#
# Équivalent Java : il n'y a pas vraiment d'équivalent Spring Boot standard.
#   C'est comme un @Service qui appelle un service ML externe pour
#   transformer des strings en float[] et les stocker dans une DB spécialisée.
# ═══════════════════════════════════════════════════════════════════════

def embed_and_index(chunks: list[str], metadatas: list[dict]) -> None:
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ MICRO-COURS : "-> None" comme type de retour                    │
    # │                                                                 │
    # │ En Java, une méthode qui ne retourne rien est déclarée "void" : │
    # │   public void embedAndIndex(List<String> chunks, ...) { }      │
    # │                                                                 │
    # │ En Python, l'équivalent est "-> None" dans la signature.        │
    # │ C'est optionnel — une fonction sans "return" retourne None      │
    # │ automatiquement.                                                │
    # └─────────────────────────────────────────────────────────────────┘
    """
    Convertit les chunks en embeddings vectoriels
    et les stocke dans ChromaDB localement.
    """
    print(f"🔄 Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
    print("   (Premier lancement = téléchargement ~1.2GB, patience...)")

    # ─────────────────────────────────────────────────────────────────
    # HuggingFaceEmbeddings : le modèle qui transforme les textes en vecteurs
    #
    # model_name         : nom du modèle sur HuggingFace Hub (téléchargé automatiquement)
    # model_kwargs       : paramètres passés au modèle PyTorch
    #                      {"device": "cpu"} = utilise le CPU (pas de GPU requis)
    # encode_kwargs      : paramètres pour l'encodage
    #                      {"normalize_embeddings": True} = normalise les vecteurs
    #                      pour que la similarité cosinus fonctionne correctement
    # cache_folder       : où stocker le modèle téléchargé (évite de re-télécharger)
    #
    # MICRO-COURS : dict comme argument de fonction {"key": value}
    #
    # En Java : new HuggingFaceEmbeddings(modelName, Map.of("device", "cpu"), ...)
    # En Python : HuggingFaceEmbeddings(model_kwargs={"device": "cpu"})
    #
    # Les {"device": "cpu"} sont des dictionnaires Python passés directement
    # comme valeurs d'arguments nommés. Très courant dans les APIs Python.
    # ─────────────────────────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},                  # Pas de GPU requis
        encode_kwargs={"normalize_embeddings": True},    # Normalisation pour la similarité cosinus
        cache_folder="./models"                          # Dossier de cache local du modèle
    )

    print("🔄 Indexation dans ChromaDB...")

    # ─────────────────────────────────────────────────────────────────
    # Chroma.from_texts() : crée la base vectorielle en une seule commande
    #
    # Cette méthode fait trois choses en une :
    #   1. Appelle embeddings.embed_documents(chunks) → liste de vecteurs float[]
    #   2. Stocke chaque vecteur avec ses métadonnées dans ChromaDB
    #   3. Persiste la base sur disque dans persist_directory
    #
    # Paramètres :
    #   texts            : liste des textes des chunks
    #   embedding        : le modèle d'embedding à utiliser
    #   metadatas        : liste de dicts {source, page} alignée avec texts
    #   persist_directory: dossier où sauvegarder la base ChromaDB sur disque
    #
    # Après l'exécution, le dossier chroma_db/ contiendra une base SQLite
    # avec tous les vecteurs et leurs métadonnées, persistée sur disque.
    # ─────────────────────────────────────────────────────────────────
    vectorstore = Chroma.from_texts(
        texts=chunks,                    # Les textes à vectoriser et indexer
        embedding=embeddings,            # Le modèle qui convertit texte → vecteur
        metadatas=metadatas,             # Les métadonnées {source, page} de chaque chunk
        persist_directory=CHROMA_DB_DIR  # Sauvegarde sur disque dans "chroma_db/"
    )

    print(f"✅ {len(chunks)} chunks indexés dans ChromaDB")
    print(f"📁 Base vectorielle sauvegardée dans : {CHROMA_DB_DIR}/")


# ═══════════════════════════════════════════════════════════════════════
# BLOC 7 : FONCTION main — Orchestration du pipeline complet
#
# Rôle : Appelle les trois fonctions dans l'ordre pour exécuter
#        le pipeline d'ingestion de bout en bout.
#
# Équivalent Java :
#   public static void main(String[] args) {
#       List<Map<String,Object>> pages = loadPdfs(DOCUMENTS_DIR);
#       var result = chunkPages(pages);
#       embedAndIndex(result.chunks(), result.metas());
#   }
# ═══════════════════════════════════════════════════════════════════════

def main():
    """
    Point d'entrée du pipeline d'ingestion.
    Exécute les 3 étapes dans l'ordre : lecture → chunking → indexation.
    """
    print("🚀 Démarrage de l'ingestion des documents...\n")

    # Étape 1 — Lecture des PDFs
    pages = load_pdfs(DOCUMENTS_DIR)
    if not pages:  # Si aucune page n'a été extraite, on arrête ici
        return     # "return" sans valeur dans une fonction void = sortie anticipée (comme "return;" en Java)

    # Étape 2 — Chunking
    # MICRO-COURS : Déstructuration du tuple retourné par chunk_pages()
    # chunk_pages() retourne (all_chunks, all_metadatas) — un tuple de 2 éléments.
    # Python les assigne directement à "chunks" et "metadatas".
    # En Java tu aurais besoin d'un objet intermédiaire ChunkResult.
    chunks, metadatas = chunk_pages(pages)

    # Étape 3 — Embedding + Indexation
    embed_and_index(chunks, metadatas)

    print("\n🎉 Ingestion terminée ! La base vectorielle est prête.")


# ═══════════════════════════════════════════════════════════════════════
# BLOC 8 : POINT D'ENTRÉE DU SCRIPT — if __name__ == "__main__"
#
# MICRO-COURS : C'est l'une des patterns les plus importantes de Python.
#
# En Java, le point d'entrée est TOUJOURS :
#   public static void main(String[] args) { ... }
#
# En Python, chaque fichier est un "module" qui peut être :
#   A) Exécuté directement : python src/ingest.py
#   B) Importé par un autre fichier : import ingest
#
# Le problème : si tu fais "import ingest" dans un autre fichier,
# tu ne veux PAS que main() s'exécute automatiquement !
#
# La solution : "if __name__ == '__main__':"
#   - Quand exécuté directement : __name__ vaut "__main__" → le if est VRAI
#   - Quand importé            : __name__ vaut "ingest"   → le if est FAUX
#
# Exemple simple :
#   # fichier: calcul.py
#   def addition(a, b):
#       return a + b
#
#   if __name__ == "__main__":
#       print(addition(2, 3))   ← s'exécute SEULEMENT avec : python calcul.py
#                               ← NE s'exécute PAS si : import calcul
#
# PIÈGE pour dev Java : oublier ce guard et voir son code s'exécuter
# à l'import, causant des side-effects inattendus.
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()  # Appelle main() uniquement si ce fichier est exécuté directement


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONCLUSION DU FICHIER                                               ║
# ║                                                                      ║
# ║  🎯 CONCEPTS PYTHON CLÉS VUS ICI                                     ║
# ║                                                                      ║
# ║  1. import / from X import Y                                        ║
# ║     → comme "import" Java mais deux formes, conventions différentes ║
# ║                                                                      ║
# ║  2. List comprehension [x for x in col if cond]                     ║
# ║     → équivalent concis de stream().filter().collect() Java         ║
# ║                                                                      ║
# ║  3. with ... as ... (context manager)                               ║
# ║     → équivalent de try-with-resources Java                         ║
# ║                                                                      ║
# ║  4. enumerate(col, start=1)                                         ║
# ║     → boucle avec index automatique, sans i++ manuel                ║
# ║                                                                      ║
# ║  5. Retour de tuple : return a, b → chunks, metas = func()          ║
# ║     → retour multiple sans classe intermédiaire                     ║
# ║                                                                      ║
# ║  6. if __name__ == "__main__"                                        ║
# ║     → guard essentiel pour distinguer exécution vs import           ║
# ║                                                                      ║
# ║  ⚠️  PIÈGES À ÉVITER                                                 ║
# ║                                                                      ║
# ║  - "for item in liste" itère les VALEURS (pas les index comme Java) ║
# ║    → utiliser enumerate() si tu as besoin de l'index               ║
# ║  - Oublier "if __name__ == '__main__':" → code exécuté à l'import  ║
# ║  - f"texte {var}" ≠ "{var}" template LangChain (évaluation immédiate║
# ║    vs différée)                                                     ║
# ║  - len() et pas .size() ni .length                                  ║
# ║                                                                      ║
# ║  🔗 CONNEXION AVEC L'ARCHITECTURE                                    ║
# ║                                                                      ║
# ║  Ce fichier s'exécute UNE SEULE FOIS en ligne de commande :         ║
# ║    python src/ingest.py                                             ║
# ║  Il prépare la base vectorielle chroma_db/ qui sera lue par        ║
# ║  src/rag.py à chaque question de l'utilisateur.                    ║
# ║  src/prompts.py n'est PAS utilisé ici (les prompts sont pour le    ║
# ║  LLM, pas pour l'ingestion).                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
