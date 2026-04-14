# src/prompts.py
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FICHE D'IDENTITÉ DU FICHIER                                         ║
# ║                                                                      ║
# ║  Rôle       : Centralise les templates de prompts envoyés au LLM    ║
# ║               (Google Gemini). C'est le "script" que le modèle IA   ║
# ║               reçoit pour savoir comment se comporter.              ║
# ║                                                                      ║
# ║  En Java    : Équivalent d'une classe de constantes statiques :      ║
# ║               public class Prompts {                                 ║
# ║                   public static final String RAG_SYSTEM = "...";    ║
# ║               }                                                      ║
# ║                                                                      ║
# ║  En Angular : Équivalent d'un fichier constants.ts :                 ║
# ║               export const RAG_SYSTEM_PROMPT = `...`;               ║
# ║                                                                      ║
# ║  Flux       : Ce fichier ne reçoit rien et ne calcule rien.         ║
# ║               Il est IMPORTÉ par src/rag.py qui utilise ces         ║
# ║               constantes pour construire la chaîne LangChain.       ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════
# MICRO-COURS : Les constantes en Python
#
# En Java, tu déclares une constante ainsi :
#   public static final String MA_CONSTANTE = "valeur";
#
# En Python, il n'existe PAS de mot-clé "final" ou "const".
# La convention universelle est d'écrire le nom EN MAJUSCULES_AVEC_UNDERSCORE.
# C'est un contrat implicite entre développeurs : "ne modifie pas cette valeur".
#
# Exemple simple :
#   MAX_RETRIES = 3          # tout le monde comprend que c'est une constante
#   api_url = "http://..."   # minuscule = variable normale, modifiable
#
# PIÈGE pour dev Java : Python ne t'empêche PAS de modifier une "constante".
# C'est une convention, pas une contrainte du langage.
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# MICRO-COURS : Les chaînes multi-lignes avec triple guillemets """..."""
#
# En Java, pour une longue chaîne sur plusieurs lignes, tu écrirais :
#   String prompt = "Ligne 1\n" +
#                   "Ligne 2\n" +
#                   "Ligne 3";
#
# Ou en Java 15+ avec les Text Blocks :
#   String prompt = """
#       Ligne 1
#       Ligne 2
#       """;
#
# En Python, les triple guillemets """ permettent d'écrire une chaîne
# sur autant de lignes que tu veux, avec des retours à la ligne réels :
#   texte = """
#   Bonjour,
#   Comment allez-vous ?
#   """
#
# Tout ce qui est entre les """ est inclus dans la chaîne, y compris
# les espaces et les sauts de ligne.
#
# PIÈGE : Les espaces au début de chaque ligne SONT inclus dans la chaîne.
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# BLOC 1 : SYSTEM PROMPT — Instructions permanentes pour le LLM
#
# Ce qu'est un "system prompt" :
#   Dans une conversation avec un LLM (Large Language Model), il y a deux
#   types de messages :
#     - Le "system prompt" : instructions invisibles pour l'utilisateur,
#       envoyées en premier, qui définissent le RÔLE et le COMPORTEMENT du
#       modèle. C'est comme un contrat de travail donné à l'IA.
#     - Le "human prompt" : les questions posées par l'utilisateur.
#
# Analogie Java/Spring :
#   Imagine un @Service configuré avec des règles métier au démarrage
#   de l'application (dans @PostConstruct). Ces règles s'appliquent
#   à chaque appel, l'utilisateur ne les voit pas.
#
# Ce prompt dit au modèle :
#   1. Qui il est (expert pharma chez PharmaCo Belgium)
#   2. Ce qu'il peut faire (répondre UNIQUEMENT sur les docs fournis)
#   3. Comment répondre (format avec citation de source)
# ═══════════════════════════════════════════════════════════════════════

RAG_SYSTEM_PROMPT = """Tu es un assistant expert en réglementation pharmaceutique.
Tu travailles pour PharmaCo Belgium et tu aides les équipes qualité à retrouver
des informations précises dans les SOPs et guidelines réglementaires.

RÈGLES STRICTES :
1. Tu réponds UNIQUEMENT en te basant sur les documents fournis dans le contexte.
2. Tu cites TOUJOURS la source exacte : nom du document et numéro de page.
3. Si l'information n'est pas dans le contexte, tu réponds :
   "Je n'ai pas trouvé cette information dans les documents disponibles."
4. Tu ne inventes jamais d'information.
5. Tu réponds en français sauf si la question est posée en anglais.

FORMAT DE RÉPONSE :
- Réponse claire et concise
- Citation obligatoire : [Source: nom_du_fichier, Page X]
"""
# ↑ POURQUOI UNE CONSTANTE GLOBALE (pas dans une classe) ?
#   En Java tu aurais fait : public class Prompts { public static final String ... }
#   En Python, un fichier (module) EST déjà un espace de noms.
#   Pas besoin d'une classe conteneur — les variables au niveau du fichier
#   sont accessibles via : from src.prompts import RAG_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════
# BLOC 2 : HUMAN PROMPT TEMPLATE — La question avec son contexte
#
# Ce prompt est un TEMPLATE : il contient des espaces réservés {context}
# et {question} qui seront remplacés dynamiquement par LangChain.
#
# MICRO-COURS : Les placeholders {variable} dans les templates LangChain
#
# Ce n'est PAS une f-string Python standard (qui commence par f"...").
# C'est un PromptTemplate LangChain — les accolades {variable} sont
# des marqueurs que LangChain remplace au moment de l'appel.
#
# Comparaison :
#
#   # Python pur (f-string) — évaluation immédiate :
#   context = "doc1"
#   question = "quelle dose ?"
#   texte = f"Contexte : {context}\nQuestion : {question}"
#   # → remplacé AU MOMENT de l'écriture de la ligne
#
#   # LangChain PromptTemplate — évaluation différée :
#   template = "Contexte : {context}\nQuestion : {question}"
#   # → les {} sont remplacés PLUS TARD quand LangChain appelle le LLM
#   # → c'est comme un PreparedStatement JDBC en Java
#
# Analogie Java :
#   PreparedStatement stmt = conn.prepareStatement(
#       "SELECT * FROM docs WHERE context = ? AND question = ?"
#   );
#   stmt.setString(1, context);   ← remplacement différé
#   stmt.setString(2, question);
#
# Ce template sera utilisé dans src/rag.py avec ChatPromptTemplate.from_messages()
# ═══════════════════════════════════════════════════════════════════════

RAG_HUMAN_PROMPT = """Contexte extrait des documents :
{context}

Question : {question}

Réponds en citant précisément les sources."""
# ↑ {context}  : sera rempli par les chunks de documents retrouvés par ChromaDB
#   {question} : sera rempli par la question posée par l'utilisateur dans Streamlit


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONCLUSION DU FICHIER                                               ║
# ║                                                                      ║
# ║  🎯 CE QUE TU DOIS RETENIR                                           ║
# ║                                                                      ║
# ║  1. CONSTANTES Python : MAJUSCULES_UNDERSCORE (convention, pas loi) ║
# ║     → En Java : public static final String                          ║
# ║                                                                      ║
# ║  2. TRIPLE GUILLEMETS """...""" : chaînes multi-lignes lisibles      ║
# ║     → En Java 15+ : Text Blocks avec """                            ║
# ║                                                                      ║
# ║  3. MODULE = espace de noms : pas besoin de classe conteneur        ║
# ║     → En Java tu créerais une classe Prompts avec static final      ║
# ║     → En Python le fichier lui-même joue ce rôle                   ║
# ║                                                                      ║
# ║  4. PLACEHOLDERS {variable} : templates LangChain (évaluation       ║
# ║     différée, comme PreparedStatement en Java)                      ║
# ║                                                                      ║
# ║  ⚠️  PIÈGES À ÉVITER                                                 ║
# ║                                                                      ║
# ║  - Ne confonds pas f"{var}" (f-string Python, immédiat) et          ║
# ║    "{var}" dans un template LangChain (différé)                     ║
# ║  - Python ne t'empêche pas de modifier une "constante" en MAJUSCULE ║
# ║    → c'est une convention sociale, pas une contrainte du langage    ║
# ║  - Les espaces en début de ligne dans """...""" SONT dans la chaîne ║
# ║                                                                      ║
# ║  🔗 CONNEXION AVEC L'ARCHITECTURE                                    ║
# ║                                                                      ║
# ║  Ce fichier est importé par src/rag.py :                            ║
# ║    from src.prompts import RAG_SYSTEM_PROMPT, RAG_HUMAN_PROMPT      ║
# ║                                                                      ║
# ║  RAG_SYSTEM_PROMPT → configure le comportement global du LLM        ║
# ║  RAG_HUMAN_PROMPT  → template pour chaque question utilisateur      ║
# ║  Les deux forment ensemble le "ChatPromptTemplate" LangChain        ║
# ╚══════════════════════════════════════════════════════════════════════╝
