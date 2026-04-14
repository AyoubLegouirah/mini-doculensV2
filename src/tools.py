# src/tools.py
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FICHE D'IDENTITÉ DU FICHIER                                         ║
# ║                                                                      ║
# ║  Rôle    : Définit les "outils" que l'agent ReAct peut appeler.     ║
# ║            Chaque fonction @tool est une capacité de l'agent :      ║
# ║              - search_documents : cherche dans les PDFs indexés     ║
# ║              - generate_report  : crée un rapport Markdown          ║
# ║              - search_web       : cherche sur internet              ║
# ║              - generate_chart   : génère un graphique PNG           ║
# ║                                                                      ║
# ║  En Java : @Component avec méthodes @Tool — comme des actions       ║
# ║            d'un chatbot Spring AI ou @ManagedOperation JMX          ║
# ║                                                                      ║
# ║  Flux    :                                                           ║
# ║    src/agent.py reçoit AGENT_TOOLS                                  ║
# ║      → AgentExecutor choisit quel outil appeler                     ║
# ║      → passe les arguments (strings)                                ║
# ║      → reçoit une string de résultat                                ║
# ║      → continue son raisonnement (pattern ReAct)                   ║
# ║                                                                      ║
# ║  Dépendances :                                                       ║
# ║    ← src/rag.py   : importe ask() pour search_documents             ║
# ║    → src/agent.py : exporte AGENT_TOOLS                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════
# BLOC 1 : IMPORTS
# ═══════════════════════════════════════════════════════════════════════

import os                       # Opérations système (makedirs, chemins...)
import json                     # Sérialisation/désérialisation JSON (importé, utilisé implicitement)
from datetime import datetime   # Classe datetime pour horodatage des fichiers
# ↑ MICRO-COURS : "from datetime import datetime"
#   Le module s'appelle "datetime" ET la classe principale s'appelle aussi "datetime".
#   Donc : from datetime import datetime
#          ↑ module         ↑ classe
#   Après cet import : datetime.now() fonctionne directement (sans préfixe).
#   En Java : import java.time.LocalDateTime; → LocalDateTime.now()

from langchain.tools import tool  # Le décorateur @tool de LangChain
from src.rag import ask           # Pipeline RAG complet (défini dans src/rag.py)


# ═══════════════════════════════════════════════════════════════════════
# BLOC 2 : OUTIL search_documents — Recherche dans les PDFs indexés
#
# MICRO-COURS : Le décorateur @tool de LangChain
#
# Un décorateur Python est une fonction qui ENVELOPPE une autre fonction
# pour lui ajouter des comportements sans modifier son code.
#
# Syntaxe : @nom_du_decorateur
#           def ma_fonction(...):
#
# En Java, l'équivalent le plus proche est une annotation Spring :
#   @Service, @Transactional, @Cacheable, @Tool
#   Ces annotations ajoutent des comportements sans modifier la méthode.
#
# Que fait @tool de LangChain ?
#   1. Lit la docstring de la fonction → devient la DESCRIPTION de l'outil
#      (le LLM lit cette description pour décider QUAND utiliser cet outil)
#   2. Lit le nom de la fonction → devient le NOM de l'outil
#   3. Lit les type hints des paramètres → définit le SCHÉMA d'entrée
#   4. Enveloppe la fonction dans un objet Tool LangChain compatible avec AgentExecutor
#
# IMPORTANT : La docstring EST le prompt envoyé au LLM pour qu'il comprenne
# QUAND et COMMENT utiliser cet outil. Elle doit être claire et descriptive.
#
# Exemple simple de décorateur :
#   def mesurer_temps(fonction):
#       def enveloppe(*args, **kwargs):
#           debut = time.time()
#           resultat = fonction(*args, **kwargs)
#           print(f"Durée : {time.time() - debut:.2f}s")
#           return resultat
#       return enveloppe
#
#   @mesurer_temps
#   def calcul_lent():
#       time.sleep(1)
#
# @mesurer_temps est exactement équivalent à : calcul_lent = mesurer_temps(calcul_lent)
# Le "@" est du sucre syntaxique (syntactic sugar) — comme @Override en Java
# sauf qu'en Python les décorateurs sont de vraies fonctions.
# ═══════════════════════════════════════════════════════════════════════

@tool  # ← Transforme search_documents en un outil LangChain utilisable par l'agent
def search_documents(query: str) -> str:
    """
    Recherche des informations précises dans les documents PDF indexés (SOPs,
    guidelines réglementaires FDA, EMA, ICH). Utilise cet outil quand tu as
    besoin de retrouver une information spécifique dans la base documentaire
    interne de PharmaCo Belgium.

    Args:
        query: La question ou le sujet à rechercher dans les documents

    Returns:
        La réponse générée par le LLM avec les sources citées (document + page)
    """
    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : try / except en Python
    #
    # En Java :
    #   try {
    #       result = ask(query);
    #   } catch (FileNotFoundException e) {
    #       return "Erreur : base introuvable.";
    #   } catch (Exception e) {
    #       return "Erreur : " + e.getMessage();
    #   }
    #
    # En Python :
    #   try:
    #       result = ask(query)
    #   except FileNotFoundError:
    #       return "Erreur : base introuvable."
    #   except Exception as e:
    #       return f"Erreur : {str(e)}"
    #
    # Points clés :
    #   - Pas d'accolades — l'indentation délimite les blocs
    #   - "except TypeException:" sans variable → on n'a pas besoin de l'objet
    #   - "except Exception as e:" → capture TOUTES les exceptions avec alias "e"
    #   - str(e) = e.getMessage() en Java (convertit l'exception en string)
    #   - L'ordre compte : exceptions spécifiques d'abord, génériques ensuite
    #     (comme en Java : catch spécifique avant catch(Exception e))
    # ─────────────────────────────────────────────────────────────────
    try:
        result = ask(query)  # Appelle le pipeline RAG complet (src/rag.py)

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : List comprehension dans "\n".join()
        #
        # On combine deux concepts vus précédemment :
        #   1. List comprehension : [expression for x in collection]
        #   2. "\n".join(liste)   : jointure avec séparateur
        #
        # En Java :
        #   List<String> lines = result.get("chunks").stream()
        #       .map(c -> "  - " + c.get("source") + ", page " + c.get("page"))
        #       .collect(Collectors.toList());
        #   String sourcesText = String.join("\n", lines);
        #
        # En Python (tout en une ligne) :
        #   sources_text = "\n".join([
        #       f"  - {chunk['source']}, page {chunk['page']}"
        #       for chunk in result["chunks"]
        #   ])
        #
        # La list comprehension est évaluée d'abord → liste de strings formatées
        # Puis "\n".join() les colle avec des retours à la ligne.
        # ─────────────────────────────────────────────────────────────
        sources_text = "\n".join([
            f"  - {chunk['source']}, page {chunk['page']}"
            for chunk in result["chunks"]
        ])

        return f"{result['answer']}\n\nSources utilisées :\n{sources_text}"
        # ↑ Retourne la réponse Gemini + la liste formatée des sources

    except FileNotFoundError:
        # FileNotFoundError est levée par load_vectorstore() dans rag.py
        # si chroma_db/ n'existe pas (ingest.py n'a pas encore été lancé)
        return "Erreur : aucune base documentaire trouvée. Lance d'abord l'ingestion des PDFs."

    except Exception as e:  # "as e" → assigne l'exception à la variable "e" (comme catch(Exception e) Java)
        return f"Erreur lors de la recherche : {str(e)}"
        # ↑ str(e) convertit l'exception en message lisible — équivalent de e.getMessage() Java
        #   MAIS str(e) peut retourner une string vide si l'exception n'a pas de message.
        #   Plus robuste : repr(e) donne toujours quelque chose.


# ═══════════════════════════════════════════════════════════════════════
# BLOC 3 : OUTIL generate_report — Génération de rapport Markdown
#
# Rôle : Prend un contenu textuel et le sauvegarde en fichier .md
#        dans le dossier reports/ avec un header standardisé.
# ═══════════════════════════════════════════════════════════════════════

@tool
def generate_report(content: str, report_title: str = "Rapport d'analyse") -> str:
    # ┌─────────────────────────────────────────────────────────────────┐
    # │ MICRO-COURS : Paramètres avec valeur par défaut                 │
    # │                                                                 │
    # │ En Java, tu simulais ça avec la surcharge de méthodes :        │
    # │   public String generateReport(String content) {               │
    # │       return generateReport(content, "Rapport d'analyse");     │
    # │   }                                                             │
    # │   public String generateReport(String content, String title) { │
    # │       ...                                                       │
    # │   }                                                             │
    # │                                                                 │
    # │ En Python, un seul paramètre avec valeur par défaut suffit :   │
    # │   def generate_report(content, report_title="Rapport d'analyse")│
    # │                                                                 │
    # │ L'appelant peut omettre report_title :                         │
    # │   generate_report("mon contenu")                               │
    # │   → report_title vaut automatiquement "Rapport d'analyse"      │
    # │                                                                 │
    # │ Ou le fournir explicitement :                                   │
    # │   generate_report("mon contenu", "Analyse ICH Q10")            │
    # │   generate_report("mon contenu", report_title="Analyse ICH Q10")│
    # │                                                                 │
    # │ RÈGLE IMPORTANTE : Les paramètres avec défaut doivent TOUJOURS │
    # │ être APRÈS les paramètres sans défaut dans la signature.        │
    # │   ✅ def f(a, b="défaut")                                       │
    # │   ❌ def f(a="défaut", b)  ← SyntaxError                       │
    # └─────────────────────────────────────────────────────────────────┘
    """
    Génère un rapport structuré en Markdown et le sauvegarde dans le dossier
    reports/. Utilise cet outil quand tu as collecté suffisamment d'informations
    et que tu es prêt à produire le livrable final pour l'utilisateur.

    Args:
        content: Le contenu complet du rapport (analyses, comparaisons, conclusions)
        report_title: Le titre du rapport (ex: "Analyse écarts SOPs vs EMA")

    Returns:
        Confirmation avec le chemin du fichier généré
    """
    try:
        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : isinstance() — vérification de type à l'exécution
        #
        # En Java : if (content instanceof Map) { ... }
        # En Python : if isinstance(content, dict): ...
        #
        # isinstance(objet, Type) retourne True si "objet" est une instance
        # de "Type" (ou d'une sous-classe de "Type").
        #
        # Pourquoi ce fix est nécessaire ici ?
        #   LangChain peut parfois passer les arguments comme un dict
        #   au lieu de strings séparées (comportement interne de l'agent).
        #   On normalise l'entrée avant de continuer.
        #
        # Autres usages courants :
        #   isinstance(x, str)   → vérifie que x est une string
        #   isinstance(x, int)   → vérifie que x est un entier
        #   isinstance(x, list)  → vérifie que x est une liste
        #   isinstance(x, (str, int))  → vérifie str OU int (tuple de types)
        # ─────────────────────────────────────────────────────────────
        # Fix : si LangChain passe un dict, on extrait les valeurs
        if isinstance(content, dict):  # En Java : if (content instanceof Map)
            report_title = content.get("report_title", report_title)
            content = content.get("content", str(content))

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : os.makedirs() — créer des dossiers récursivement
        #
        # En Java : Files.createDirectories(Path.of("reports"))
        # En Python : os.makedirs("reports", exist_ok=True)
        #
        # "exist_ok=True" : ne lève pas d'erreur si le dossier existe déjà
        # Sans exist_ok=True : FileExistsError si le dossier existe → crash
        # Toujours mettre exist_ok=True sauf si tu veux détecter le doublon.
        #
        # os.makedirs() (avec un 's') crée TOUS les dossiers intermédiaires :
        #   os.makedirs("a/b/c") → crée a/, puis a/b/, puis a/b/c/
        # os.mkdir() (sans 's') ne crée qu'UN seul niveau → erreur si parent absent.
        # ─────────────────────────────────────────────────────────────
        # Crée le dossier reports/ s'il n'existe pas
        os.makedirs("reports", exist_ok=True)  # En Java : Files.createDirectories(Path.of("reports"))

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : datetime.now().strftime() — formatage de date
        #
        # En Java :
        #   LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        #
        # En Python :
        #   datetime.now().strftime("%Y%m%d_%H%M%S")
        #
        # strftime = "string from time" — convertit une date en string formatée
        # Codes de format (similaires à Java) :
        #   %Y → année 4 chiffres (2024)    | yyyy Java
        #   %m → mois 2 chiffres (01-12)    | MM Java
        #   %d → jour 2 chiffres (01-31)    | dd Java
        #   %H → heure 24h (00-23)          | HH Java
        #   %M → minutes (00-59)            | mm Java
        #   %S → secondes (00-59)           | ss Java
        #
        # Ici : "20241215_143022" → nom de fichier unique et triable
        # ─────────────────────────────────────────────────────────────
        # Génère un nom de fichier unique avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Ex: "20241215_143022"

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : Chaînage de méthodes sur les strings
        #
        # En Java :
        #   String safeTitle = reportTitle.toLowerCase()
        #       .replace(" ", "_")
        #       .replace("/", "-");
        #
        # En Python, identique — les méthodes sont chaînables :
        #   safe_title = report_title.lower().replace(" ", "_").replace("/", "-")
        #
        # Méthodes string Python courantes vs Java :
        #   Python            | Java
        #   .lower()          | .toLowerCase()
        #   .upper()          | .toUpperCase()
        #   .strip()          | .trim()
        #   .replace(a, b)    | .replace(a, b)
        #   .split(",")       | .split(",")
        #   .startswith("x")  | .startsWith("x")
        #   .endswith("x")    | .endsWith("x")
        #   .contains("x")    | → "x" in string (opérateur "in")
        # ─────────────────────────────────────────────────────────────
        safe_title = report_title.lower().replace(" ", "_").replace("/", "-")
        # Ex: "Analyse ICH / EMA" → "analyse_ich_-_ema"

        filename = f"reports/{safe_title}_{timestamp}.md"
        # Ex: "reports/analyse_ich_-_ema_20241215_143022.md"

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : Triple guillemets """...""" dans une f-string
        #
        # On peut combiner f"..." et """...""" : f"""..."""
        # Cela donne une f-string multi-lignes avec interpolation.
        #
        # Le contenu du rapport est construit en une seule chaîne :
        #   - Header avec titre (# titre = H1 en Markdown)
        #   - Métadonnées (date, système)
        #   - Séparateur Markdown (---)
        #   - Contenu fourni par l'agent
        #   - Footer automatique
        # ─────────────────────────────────────────────────────────────
        # Construit le rapport complet avec header
        report_content = f"""# {report_title}

**Généré le** : {datetime.now().strftime("%d/%m/%Y à %H:%M")}
**Système** : Mini DocuLens — Agent IA PharmaCo Belgium

---

{content}

---
*Rapport généré automatiquement par l'Agent IA Mini DocuLens*
"""

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : Écriture de fichier avec "with open()"
        #
        # En Java (try-with-resources) :
        #   try (FileWriter fw = new FileWriter(filename, StandardCharsets.UTF_8);
        #        BufferedWriter bw = new BufferedWriter(fw)) {
        #       bw.write(reportContent);
        #   }
        #
        # En Python :
        #   with open(filename, "w", encoding="utf-8") as f:
        #       f.write(report_content)
        #
        # Arguments de open() :
        #   filename        : chemin du fichier à créer/écraser
        #   "w"             : mode write (écrase si existe, crée si absent)
        #                     Autres modes : "r" (read), "a" (append), "rb" (read binary)
        #   encoding="utf-8": encodage du fichier (TOUJOURS spécifier pour éviter
        #                     les problèmes de caractères accentués sur Windows)
        #
        # "as f" : assigne l'objet fichier à la variable "f"
        # f.write(str) : écrit la string dans le fichier
        # Le fichier est fermé automatiquement à la fin du bloc "with"
        # ─────────────────────────────────────────────────────────────
        # Écrit le fichier
        with open(filename, "w", encoding="utf-8") as f:  # Ouvre en mode écriture, ferme automatiquement
            f.write(report_content)                        # Écrit tout le contenu d'un coup

        return f"Rapport généré avec succès : {filename}"

    except Exception as e:
        return f"Erreur lors de la génération du rapport : {str(e)}"


# ═══════════════════════════════════════════════════════════════════════
# BLOC 4 : OUTIL search_web — Recherche internet via DuckDuckGo
#
# Rôle : Recherche des informations actuelles sur internet.
#        Utile pour les guidelines réglementaires récentes ou
#        les informations absentes des PDFs internes.
# ═══════════════════════════════════════════════════════════════════════

@tool
def search_web(query: str) -> str:
    """
    Recherche des informations actuelles sur internet — normes réglementaires
    EMA, FDA, ICH, actualités pharmaceutiques. Utilise cet outil quand tu as
    besoin d'informations externes qui ne sont pas dans la base documentaire
    interne, comme les versions actuelles des guidelines ou les exigences
    réglementaires récentes.

    Args:
        query: Le sujet ou la question à rechercher sur internet

    Returns:
        Un résumé des informations trouvées en ligne
    """
    try:
        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : Import à l'intérieur d'une fonction (lazy import)
        #
        # En général, les imports sont tous en haut du fichier.
        # Mais ici, l'import est DANS la fonction — pourquoi ?
        #
        # Raison 1 — Optionalité :
        #   duckduckgo-search est une dépendance OPTIONNELLE.
        #   Si elle n'est pas installée, on ne veut PAS que l'IMPORT
        #   en haut du fichier fasse planter toute l'application au démarrage.
        #   En l'important ici, l'erreur n'arrive que si cet outil est utilisé.
        #
        # Raison 2 — Meilleur message d'erreur :
        #   On peut attraper ImportError spécifiquement et donner
        #   des instructions d'installation claires.
        #
        # En Java, l'équivalent serait du chargement dynamique de classe :
        #   Class.forName("com.duckduckgo.Search")
        #   → lève ClassNotFoundException si non disponible
        #
        # Convention : les imports dans les fonctions sont acceptés mais
        # doivent être justifiés (optionalité, évitement de circularité...).
        # ─────────────────────────────────────────────────────────────
        # Import ici pour ne pas bloquer si non installé
        from langchain_community.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()  # Crée une instance du moteur de recherche
        result = search.run(query)      # Effectue la recherche et retourne un string de résultats
        return result

    except ImportError:
        # ImportError : levée quand un module Python n'est pas installé
        # En Java : ClassNotFoundException
        # Ici on retourne un message d'aide plutôt que de faire crasher l'agent
        return (
            "Outil search_web non disponible : installe 'duckduckgo-search' "
            "avec : pip install duckduckgo-search"
        )
    except Exception as e:
        return f"Erreur lors de la recherche web : {str(e)}"


# ═══════════════════════════════════════════════════════════════════════
# BLOC 5 : OUTIL generate_chart — Génération de graphique PNG
#
# Rôle : Parse une string de données "catégorie:valeur, ..."
#        et génère un graphique en barres avec matplotlib.
# ═══════════════════════════════════════════════════════════════════════

@tool
def generate_chart(data: str, chart_title: str = "Graphique d'analyse") -> str:
    """
    Génère un graphique en barres à partir de données textuelles et le sauvegarde
    en image PNG dans le dossier reports/. Utilise cet outil quand la mission
    demande une visualisation graphique, un diagramme, ou une représentation
    visuelle de données. Les données doivent être sous forme de paires
    'catégorie:valeur' séparées par des virgules.

    Args:
        data: Données à visualiser, format 'catégorie1:valeur1, catégorie2:valeur2'
        chart_title: Le titre du graphique

    Returns:
        Confirmation avec le chemin de l'image générée
    """
    try:
        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : Import conditionnel de matplotlib
        #
        # matplotlib est importé ici (et non en haut du fichier) pour
        # les mêmes raisons que DuckDuckGo : dépendance optionnelle.
        #
        # matplotlib.use("Agg") :
        #   Configure le "backend" de rendu graphique.
        #   "Agg" = Anti-Grain Geometry → rendu en mémoire SANS affichage à l'écran.
        #   DOIT être appelé AVANT "import matplotlib.pyplot" sinon erreur.
        #   Obligatoire en environnement serveur (Streamlit, API) où il n'y a
        #   pas d'écran disponible.
        #
        # Analogie Java : comme configurer un renderer graphique headless
        #   System.setProperty("java.awt.headless", "true");
        # ─────────────────────────────────────────────────────────────
        import matplotlib
        matplotlib.use("Agg")        # Mode sans interface graphique — AVANT l'import pyplot
        import matplotlib.pyplot as plt  # Interface de haut niveau pour créer des graphiques

        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/chart_{timestamp}.png"

        # ─────────────────────────────────────────────────────────────
        # BLOC DE PARSING — Conversion de la string en listes de données
        #
        # Entrée attendue : "EMA:12, FDA:8, ICH:5, Interne:15"
        # Sortie souhaitée :
        #   categories = ["EMA", "FDA", "ICH", "Interne"]
        #   values     = [12.0, 8.0, 5.0, 15.0]
        #
        # MICRO-COURS : str.split() — découper une string
        #
        # En Java :
        #   String[] items = data.split(",");
        #   for (String item : items) { ... }
        #
        # En Python :
        #   for item in data.split(","):   ← pas besoin de stocker dans un tableau
        #
        # str.split(séparateur) retourne une liste de strings.
        # str.split(",")  → ["EMA:12", " FDA:8", " ICH:5", " Interne:15"]
        # str.split(",", 1) → maxsplit=1 : split une seule fois max
        # ─────────────────────────────────────────────────────────────
        # Parse les données format "catégorie:valeur, catégorie:valeur"
        categories = []  # Noms des catégories (axe X du graphique)
        values     = []  # Valeurs numériques (hauteur des barres)

        for item in data.split(","):  # Découpe par virgule → liste d'items "catégorie:valeur"
            item = item.strip()       # Supprime les espaces autour de chaque item

            if ":" in item:           # En Java : item.contains(":")
                # ─────────────────────────────────────────────────────
                # MICRO-COURS : str.split(":", 1) — split avec limite
                #
                # "EMA:12:extra".split(":") → ["EMA", "12", "extra"]
                # "EMA:12:extra".split(":", 1) → ["EMA", "12:extra"]
                #                                          ↑ arrêt après 1 split
                #
                # Le "1" est le nombre MAX de splits à effectuer.
                # Cela protège si la valeur contient aussi un ":"
                # ─────────────────────────────────────────────────────
                parts = item.split(":", 1)         # Découpe "EMA:12" → ["EMA", "12"]
                categories.append(parts[0].strip()) # "EMA" → ajout à la liste des catégories

                # ─────────────────────────────────────────────────────
                # MICRO-COURS : float() — conversion string vers décimal
                #
                # En Java : Double.parseDouble(parts[1].trim())
                # En Python : float(parts[1].strip())
                #
                # float() lève ValueError si la string n'est pas un nombre.
                # Ici on attrape l'exception et on met 1.0 par défaut.
                # C'est le pattern try/except pour la conversion de types.
                #
                # Autres conversions Python :
                #   int("42")    → 42          | Integer.parseInt() Java
                #   float("3.14")→ 3.14        | Double.parseDouble() Java
                #   str(42)      → "42"        | String.valueOf() Java
                #   bool(0)      → False       | Boolean.valueOf() Java
                # ─────────────────────────────────────────────────────
                try:
                    values.append(float(parts[1].strip()))  # "12" → 12.0
                except ValueError:
                    values.append(1.0)  # Valeur par défaut si non parsable (ex: "N/A")

        # ─────────────────────────────────────────────────────────────
        # Si aucune donnée n'a pu être parsée, données d'exemple par défaut
        # ─────────────────────────────────────────────────────────────
        # Si pas de données parsables, crée un graphique exemple
        if not categories:  # Liste vide → falsy en Python (comme vu dans ingest.py)
            categories = ["Donnée 1", "Donnée 2", "Donnée 3"]
            values     = [3, 5, 2]

        # ─────────────────────────────────────────────────────────────
        # BLOC MATPLOTLIB — Création du graphique
        #
        # matplotlib est la bibliothèque standard de visualisation Python.
        # Analogie Java : comme JFreeChart ou JavaFX Charts.
        #
        # plt.subplots() : crée une figure (la "toile") et un axe (le "dessin")
        #   fig = la fenêtre / le container
        #   ax  = le graphique lui-même (axes, barres, labels...)
        #   figsize=(10, 6) = dimensions en pouces (largeur, hauteur)
        # ─────────────────────────────────────────────────────────────
        # Génère le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        # ↑ MICRO-COURS : déstructuration de tuple
        #   subplots() retourne un tuple (figure, axes)
        #   fig, ax = ... assigne chaque élément à sa variable
        #   (pattern déjà vu avec chunk_pages() dans ingest.py)

        bars = ax.bar(categories, values, color="#4A90D9", edgecolor="white")
        # ↑ ax.bar() crée les barres verticales
        #   Retourne une collection d'objets Rectangle (les barres)

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : zip() — itération parallèle sur deux listes
        #
        # Problème : on veut itérer sur "bars" ET "values" simultanément
        #
        # En Java :
        #   for (int i = 0; i < bars.size(); i++) {
        #       Bar bar = bars.get(i);
        #       double value = values.get(i);
        #   }
        #
        # En Python SANS zip (maladroit) :
        #   for i in range(len(bars)):
        #       bar = bars[i]
        #       value = values[i]
        #
        # En Python AVEC zip (pythonique) :
        #   for bar, value in zip(bars, values):
        #
        # zip(a, b) crée des paires : zip([1,2,3], ["a","b","c"])
        #   → [(1,"a"), (2,"b"), (3,"c")]
        # Chaque paire est déstructurée dans "bar, value" automatiquement.
        #
        # Exemple simple :
        #   noms   = ["Alice", "Bob", "Charlie"]
        #   scores = [85, 92, 78]
        #   for nom, score in zip(noms, scores):
        #       print(f"{nom} : {score}/100")
        # ─────────────────────────────────────────────────────────────
        # Ajoute les valeurs sur les barres
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # Position X : centre de la barre
                bar.get_height() + 0.1,              # Position Y : juste au-dessus de la barre
                str(int(value)),                     # Texte : valeur convertie en entier puis en string
                ha="center",                         # Alignement horizontal : centré
                va="bottom",                         # Alignement vertical : en bas du texte
                fontsize=11
            )

        # Configuration des labels et du style du graphique
        ax.set_title(chart_title, fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Nombre de références", fontsize=11)
        ax.set_xlabel("Catégories", fontsize=11)
        plt.xticks(rotation=30, ha="right")  # Rotation des labels X pour lisibilité
        plt.tight_layout()                   # Ajuste automatiquement les marges

        # ─────────────────────────────────────────────────────────────
        # Sauvegarde et fermeture du graphique
        #
        # plt.savefig() : enregistre l'image sur disque
        #   dpi=150         : résolution (150 points par pouce — bonne qualité)
        #   bbox_inches="tight" : rogne les marges blanches en excès
        #
        # plt.close() : IMPORTANT — libère la mémoire du graphique
        #   Sans plt.close(), matplotlib accumule les figures en mémoire
        #   et peut provoquer des fuites mémoire si generate_chart() est
        #   appelé plusieurs fois.
        # ─────────────────────────────────────────────────────────────
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()  # Libère la mémoire — toujours appeler après savefig()

        return f"Graphique généré avec succès : {filename}"

    except Exception as e:
        return f"Erreur lors de la génération du graphique : {str(e)}"


# ═══════════════════════════════════════════════════════════════════════
# BLOC 6 : EXPORT DE LA LISTE DES OUTILS POUR L'AGENT
#
# AGENT_TOOLS est la liste des 4 outils passée à AgentExecutor dans agent.py.
# C'est le "menu" que le LLM peut consulter pour décider quoi appeler.
#
# Analogie Java :
#   public static final List<Tool> AGENT_TOOLS =
#       List.of(searchDocuments, generateReport, searchWeb, generateChart);
#
# LangChain lit les 4 éléments de cette liste et extrait pour chaque outil :
#   - Son nom (nom de la fonction Python)
#   - Sa description (sa docstring)
#   - Son schéma d'entrée (les paramètres et leurs types)
#
# Le LLM reçoit ces informations et décide QUEL outil appeler et QUELS
# arguments lui passer en fonction du contexte de la conversation.
# ═══════════════════════════════════════════════════════════════════════

# Liste exportée pour agent.py
AGENT_TOOLS = [search_documents, generate_report, search_web, generate_chart]
# ↑ On passe les FONCTIONS elles-mêmes (pas leurs résultats) dans la liste.
#   En Java : List.of(this::searchDocuments, this::generateReport, ...)
#   (méthodes références — même concept)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONCLUSION DU FICHIER                                               ║
# ║                                                                      ║
# ║  🎯 CONCEPTS PYTHON CLÉS VUS ICI                                     ║
# ║                                                                      ║
# ║  1. @decorateur → @Annotation Java (mais ce sont de vraies fonctions)║
# ║     @tool lit la docstring, le nom et les types → crée un outil LLM ║
# ║                                                                      ║
# ║  2. Paramètre par défaut : def f(p="défaut")                        ║
# ║     → évite la surcharge de méthodes Java                           ║
# ║                                                                      ║
# ║  3. try/except X / except Y / except Exception as e                 ║
# ║     → try/catch Java, du plus spécifique au plus général            ║
# ║     → str(e) = e.getMessage() Java                                  ║
# ║                                                                      ║
# ║  4. isinstance(x, dict) → x instanceof Map Java                     ║
# ║     → vérification de type à l'exécution                           ║
# ║                                                                      ║
# ║  5. zip(liste1, liste2) → itération parallèle sans index manuel     ║
# ║     → évite le for (int i=0; i<n; i++) Java                        ║
# ║                                                                      ║
# ║  ⚠️  PIÈGES À ÉVITER                                                 ║
# ║                                                                      ║
# ║  - matplotlib.use("Agg") AVANT import matplotlib.pyplot             ║
# ║    → ordre obligatoire, sinon erreur de backend                     ║
# ║  - Toujours plt.close() après plt.savefig()                         ║
# ║    → sinon fuite mémoire                                            ║
# ║  - except plus spécifique (FileNotFoundError) AVANT except Exception║
# ║    → sinon le catch général intercepte tout                         ║
# ║  - "x in str" pour contains en Python, pas .contains() comme Java  ║
# ║                                                                      ║
# ║  🔗 CONNEXION AVEC L'ARCHITECTURE                                    ║
# ║                                                                      ║
# ║  Ce fichier est le "boîte à outils" de l'agent :                   ║
# ║    ← src/rag.py   : search_documents appelle ask()                  ║
# ║    → src/agent.py : importe AGENT_TOOLS pour créer l'AgentExecutor ║
# ║  Les docstrings sont CRITIQUES — le LLM les lit pour décider        ║
# ║  quel outil appeler. Une mauvaise docstring = mauvaises décisions.  ║
# ╚══════════════════════════════════════════════════════════════════════╝
