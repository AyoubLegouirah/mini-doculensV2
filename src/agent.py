# src/agent.py
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FICHE D'IDENTITÉ DU FICHIER                                         ║
# ║                                                                      ║
# ║  Rôle    : Crée et orchestre l'agent ReAct.                         ║
# ║            Un agent ReAct est un LLM qui tourne en BOUCLE :         ║
# ║              1. Thought  : "Je dois chercher X, je vais utiliser Y" ║
# ║              2. Action   : appelle un outil (search_documents, etc.)║
# ║              3. Observation : lit le résultat de l'outil            ║
# ║              4. Répète jusqu'à avoir la réponse finale              ║
# ║                                                                      ║
# ║  En Java : @Service orchestrateur — comme un moteur BPM (Camunda)  ║
# ║            ou un WorkflowEngine qui enchaîne des étapes             ║
# ║            conditionnellement selon les résultats précédents.       ║
# ║                                                                      ║
# ║  Flux    :                                                           ║
# ║    mission (str)                                                    ║
# ║      → create_agent()       : LLM + prompt + outils assemblés      ║
# ║      → AgentExecutor.invoke : boucle ReAct (N itérations max)      ║
# ║        ├─ Gemini raisonne   : Thought                              ║
# ║        ├─ Gemini choisit    : Action + Action Input                ║
# ║        ├─ Outil exécuté     : résultat retourné                    ║
# ║        └─ Gemini observe    : Observation → retour au début        ║
# ║      → dict {answer, mission, success}                             ║
# ║                                                                      ║
# ║  Dépendances :                                                       ║
# ║    ← src/tools.py   : importe AGENT_TOOLS (les 4 outils)           ║
# ║    → agent_app.py   : appelle run_agent() depuis l'UI Streamlit    ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════
# BLOC 1 : IMPORTS
# ═══════════════════════════════════════════════════════════════════════

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
# ↑ Wrapper LangChain pour Google Gemini en mode "Chat" (échanges multi-tours)
#   Différent de genai.GenerativeModel (SDK Google brut) utilisé dans rag.py :
#   ChatGoogleGenerativeAI est intégré nativement dans l'écosystème LangChain
#   → compatible avec AgentExecutor, prompts, mémoire, etc.

from langchain.agents import AgentExecutor, create_react_agent
# ↑ AgentExecutor   : la boucle de contrôle (gère Thought/Action/Observation)
#   create_react_agent : factory qui assemble LLM + prompt + outils en un agent

from langchain.prompts import PromptTemplate
# ↑ Template de prompt avec variables nommées et validation de schéma

from src.tools import AGENT_TOOLS
# ↑ La liste [search_documents, generate_report, search_web, generate_chart]
#   définie dans src/tools.py


# ═══════════════════════════════════════════════════════════════════════
# BLOC 2 : CHARGEMENT DES VARIABLES D'ENVIRONNEMENT ET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

load_dotenv()

AGENT_MODEL    = "gemini-2.5-flash"  # Modèle Gemini utilisé par l'agent
MAX_ITERATIONS = 10                  # Circuit-breaker : l'agent s'arrête après 10 étapes max
# ↑ POURQUOI une limite d'itérations ?
#   Sans limite, un agent peut entrer dans une boucle infinie
#   (chercher → pas trouvé → re-chercher → ...).
#   MAX_ITERATIONS = 10 est un filet de sécurité.
#   Analogie Java : timeout sur un @Async ou un ExecutorService.


# ═══════════════════════════════════════════════════════════════════════
# BLOC 3 : LE PROMPT REACT — Le "manuel de raisonnement" de l'agent
#
# MICRO-COURS : Qu'est-ce que le pattern ReAct ?
#
# ReAct = Reasoning + Acting
# C'est un pattern de prompting qui force le LLM à alterner :
#   - Thought     : réfléchir avant d'agir (raisonnement visible)
#   - Action      : choisir un outil à utiliser
#   - Action Input: les arguments à passer à l'outil
#   - Observation : lire et analyser le résultat de l'outil
#   - (recommencer jusqu'à pouvoir répondre)
#   - Final Answer: la réponse définitive
#
# ANALOGIE JAVA :
#   Imagine un service Java avec une boucle do-while :
#
#   do {
#       String thought = llm.think(currentContext);       // Thought
#       String action  = llm.chooseAction(thought);       // Action
#       String input   = llm.prepareInput(thought);       // Action Input
#       String obs     = toolRegistry.execute(action, input); // Observation
#       currentContext += thought + action + input + obs;
#   } while (!llm.hasAnswer(currentContext));             // Final Answer
#
# POURQUOI CE PROMPT EST DIFFÉRENT DU RAG_SYSTEM_PROMPT ?
#   RAG_SYSTEM_PROMPT (src/prompts.py) : dit "répondre à des questions"
#   AGENT_PROMPT_TEMPLATE (ici) : dit "accomplis une mission en plusieurs étapes"
#   L'agent a besoin d'un PROTOCOLE de raisonnement strict pour que
#   LangChain puisse parser ses réponses et déclencher les outils.
# ═══════════════════════════════════════════════════════════════════════

# --- System prompt de l'agent ---
# C'est ici que tu définis la "personnalité" et les règles de l'agent.
# Différent du RAG prompt : ici l'agent doit raisonner sur QUOI faire,
# pas juste répondre à une question.
AGENT_PROMPT_TEMPLATE = """Tu es un agent IA expert en réglementation pharmaceutique.
Tu travailles pour PharmaCo Belgium et tu aides les équipes qualité à analyser
leurs documents réglementaires et produire des rapports structurés.

Tu as accès aux outils suivants :
{tools}

RÈGLES DE RAISONNEMENT :
1. Réfléchis toujours avant d'agir — explique ce que tu vas faire et pourquoi.
2. Utilise search_documents pour toute information dans la base documentaire interne.
3. Utilise search_web uniquement si l'information n'est pas dans les documents internes.
4. Utilise generate_report uniquement quand tu as collecté TOUTES les informations nécessaires.
5. Ne génère jamais un rapport incomplet — fais d'abord toutes tes recherches.
6. Si tu ne trouves pas une information, dis-le clairement dans le rapport.

FORMAT DE RAISONNEMENT OBLIGATOIRE :
Thought: [Ce que tu penses faire et pourquoi]
Action: [Le nom exact de l'outil à utiliser]
Action Input: [Ce que tu envoies à l'outil]
Observation: [Le résultat retourné par l'outil]
... (répète autant de fois que nécessaire)
Thought: [J'ai toutes les informations, je peux générer le rapport]
Action: generate_report
Action Input: [Le contenu complet du rapport]
Final Answer: [Confirmation et chemin du rapport généré]

Noms des outils disponibles : {tool_names}

Mission à accomplir :
{input}

Historique du raisonnement :
{agent_scratchpad}"""
# ─────────────────────────────────────────────────────────────────────
# MICRO-COURS : Les 4 placeholders spéciaux du prompt ReAct LangChain
#
# {tools}           : LangChain injecte automatiquement la liste des outils
#                     avec leurs descriptions (lues depuis les docstrings).
#                     Le LLM lit ces descriptions pour savoir QUOI utiliser.
#
# {tool_names}      : LangChain injecte les noms exacts des outils séparés
#                     par des virgules. Le LLM doit utiliser ces noms EXACTS
#                     dans "Action: [nom_exact]" pour que le parsing fonctionne.
#
# {input}           : La mission fournie par l'utilisateur
#                     → rempli par agent_executor.invoke({"input": mission})
#
# {agent_scratchpad}: L'historique du raisonnement en cours
#                     (tous les Thought/Action/Observation précédents)
#                     → rempli AUTOMATIQUEMENT par AgentExecutor à chaque itération
#                     → c'est la "mémoire de travail" de l'agent pour la mission en cours
#
# Ces 4 placeholders sont OBLIGATOIRES pour le pattern ReAct LangChain.
# Manquer l'un d'eux → ValueError au moment de créer l'agent.
# ─────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════
# BLOC 4 : FONCTION create_agent — Assemblage des composants LangChain
#
# Rôle : Instancie et connecte les 3 pièces nécessaires à un agent :
#          1. Le LLM (Gemini) — le "cerveau"
#          2. Le PromptTemplate — le "protocole de raisonnement"
#          3. L'AgentExecutor — la "boucle de contrôle"
# ═══════════════════════════════════════════════════════════════════════

def create_agent() -> AgentExecutor:
    """
    Crée et configure l'AgentExecutor avec Gemini et les outils définis.

    Configure le LLM, le prompt ReAct, et l'orchestrateur LangChain.
    L'agent est recréé à chaque appel pour garantir un état propre
    sans mémoire des missions précédentes.

    Returns:
        AgentExecutor prêt à recevoir des missions

    Raises:
        ValueError: Si GEMINI_API_KEY n'est pas définie dans .env
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY introuvable.\n"
            "Vérifie que ton fichier .env contient : GEMINI_API_KEY=ta_clé"
        )

    # ─────────────────────────────────────────────────────────────────
    # PIÈCE 1 : Le LLM — ChatGoogleGenerativeAI
    #
    # ChatGoogleGenerativeAI vs genai.GenerativeModel (vu dans rag.py) :
    #
    #   rag.py utilise genai.GenerativeModel (SDK Google brut) :
    #     → Simple, direct, sans intégration LangChain
    #     → Idéal pour des appels LLM ponctuels
    #
    #   agent.py utilise ChatGoogleGenerativeAI (wrapper LangChain) :
    #     → Intégré dans l'écosystème LangChain
    #     → Nécessaire pour AgentExecutor, la mémoire, les callbacks, etc.
    #     → LangChain gère l'interface standard entre le LLM et l'agent
    #
    # MICRO-COURS : temperature — contrôle du "hasard" du LLM
    #
    #   temperature=0.0 : totalement déterministe — toujours le même output
    #                     pour le même input. Idéal pour du code, des calculs.
    #   temperature=0.1 : très stable, légèrement variable
    #                     → choisi ici car le raisonnement ReAct doit être
    #                        cohérent et prévisible, pas créatif
    #   temperature=0.7 : créatif, varié → pour la génération de texte libre
    #   temperature=1.0 : très aléatoire, imprévisible
    #
    # Analogie Java : comme le niveau de logs ou le seed d'un Random.
    #   new Random(42) → déterministe (toujours même séquence)
    #   new Random()   → aléatoire (seed basé sur le temps)
    # ─────────────────────────────────────────────────────────────────
    # Le LLM — même modèle que le Projet 1
    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL,
        google_api_key=api_key,
        temperature=0.1  # Faible température = raisonnement plus stable et prévisible
    )

    # ─────────────────────────────────────────────────────────────────
    # PIÈCE 2 : Le PromptTemplate — structure formelle du prompt ReAct
    #
    # PromptTemplate est différent des templates vus dans prompts.py :
    #   prompts.py : simple string avec {placeholders} → .format() manuel
    #   ici : objet LangChain avec validation des variables requises
    #
    # input_variables : liste des noms de placeholders que ce template attend.
    #   LangChain vérifie que ces 4 variables sont bien fournies avant
    #   de générer le prompt final. Si l'une manque → erreur claire.
    #
    # MICRO-COURS : list Python vs Java
    #
    #   En Java : new ArrayList<>(Arrays.asList("tools", "tool_names", ...))
    #   En Python : ["tools", "tool_names", "input", "agent_scratchpad"]
    #
    #   Les crochets [] créent une liste Python.
    #   Ici la liste contient 4 strings — les noms des placeholders du template.
    # ─────────────────────────────────────────────────────────────────
    # Le prompt ReAct structuré
    prompt = PromptTemplate(
        template=AGENT_PROMPT_TEMPLATE,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        # ↑ Les 4 placeholders que LangChain doit remplir dans le template
        #   "tools" et "tool_names" → injectés depuis AGENT_TOOLS
        #   "input" → fourni par l'utilisateur via invoke({"input": ...})
        #   "agent_scratchpad" → géré automatiquement par AgentExecutor
    )

    # ─────────────────────────────────────────────────────────────────
    # PIÈCE 3A : create_react_agent — l'agent ReAct (pas encore la boucle)
    #
    # create_react_agent() est une factory function LangChain.
    # Elle crée un objet "agent" (pas encore exécutable) qui encode :
    #   - Comment le LLM doit formater ses réponses (Thought/Action/...)
    #   - Comment parser le texte du LLM pour en extraire l'outil et ses args
    #   - Le lien entre le LLM, le prompt et les outils
    #
    # Analogie Java : c'est le "plan de travail" avant l'exécution.
    #   Comme configurer un Job Spring Batch sans encore le lancer.
    # ─────────────────────────────────────────────────────────────────
    # Création de l'agent ReAct
    agent = create_react_agent(
        llm=llm,          # Le cerveau : Gemini
        tools=AGENT_TOOLS, # Le menu d'outils disponibles
        prompt=prompt      # Le protocole de raisonnement ReAct
    )

    # ─────────────────────────────────────────────────────────────────
    # PIÈCE 3B : AgentExecutor — la boucle de contrôle (l'exécuteur)
    #
    # AgentExecutor est le "chef d'orchestre" qui gère la boucle ReAct :
    #   1. Donne l'input à l'agent → reçoit Thought + Action + Action Input
    #   2. Identifie l'outil demandé par son nom
    #   3. Appelle l'outil avec les arguments
    #   4. Récupère l'Observation (résultat de l'outil)
    #   5. Renvoie Observation à l'agent comme nouveau contexte
    #   6. Recommence jusqu'à "Final Answer" ou max_iterations
    #
    # Paramètres clés :
    #
    #   verbose=True :
    #     Affiche chaque étape du raisonnement dans le terminal.
    #     → Thought, Action, Observation visibles en temps réel.
    #     En Java : équivalent de logger.debug() sur chaque step.
    #     Mettre False en production pour réduire les logs.
    #
    #   max_iterations=MAX_ITERATIONS :
    #     Circuit-breaker — si l'agent boucle sans converger après 10
    #     itérations, AgentExecutor force l'arrêt et retourne ce qu'il a.
    #     Évite les boucles infinies coûteuses (chaque itération = appel API).
    #
    #   handle_parsing_errors=True :
    #     Si Gemini génère du texte qui ne suit pas exactement le format
    #     Thought/Action/..., AgentExecutor renvoie le texte problématique
    #     au LLM avec une demande de correction au lieu de crasher.
    #     En Java : équivalent d'un try/catch avec retry sur FormatException.
    # ─────────────────────────────────────────────────────────────────
    # L'orchestrateur qui gère la boucle
    return AgentExecutor(
        agent=agent,                           # L'agent ReAct configuré
        tools=AGENT_TOOLS,                     # Les outils que l'exécuteur peut appeler
        verbose=True,                          # Affiche Thought/Action/Observation dans le terminal
        max_iterations=MAX_ITERATIONS,         # Limite de sécurité : max 10 itérations
        handle_parsing_errors=True             # Resilience : corrige les erreurs de format LLM
    )


# ═══════════════════════════════════════════════════════════════════════
# BLOC 5 : FONCTION run_agent — Point d'entrée principal
#
# Rôle : Crée un agent frais, lui donne la mission, retourne le résultat.
#
# POURQUOI recréer l'agent à chaque appel ?
#   L'agent est créé SANS mémoire persistante entre les missions.
#   Chaque appel à run_agent() démarre avec une ardoise vierge.
#   Cela évite que les missions précédentes influencent les nouvelles.
#   (Pour un agent avec mémoire, il faudrait conserver l'AgentExecutor
#    entre les appels et lui ajouter un ConversationBufferMemory.)
# ═══════════════════════════════════════════════════════════════════════

def run_agent(mission: str) -> dict:
    """
    Point d'entrée principal — lance l'agent sur une mission.

    Crée un agent frais, lui donne la mission, et retourne
    le résultat final avec les étapes de raisonnement.

    Args:
        mission: La mission complète décrite en langage naturel

    Returns:
        Dict contenant :
        - answer: La réponse finale de l'agent
        - mission: La mission originale
        - success: True si l'agent a terminé sans erreur

    Example:
        result = run_agent(
            "Analyse les SOPs disponibles et identifie toutes les clauses "
            "mentionnant des délais de validation. Génère un rapport structuré."
        )
        print(result["answer"])
    """
    print(f"\n🤖 Mission reçue : {mission}\n")
    print("="*60)

    try:
        agent_executor = create_agent()  # Crée un agent frais sans état

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : agent_executor.invoke({"input": mission})
        #
        # invoke() est la méthode standard LangChain pour déclencher
        # une chaîne ou un agent. Elle accepte un DICT d'inputs.
        #
        # {"input": mission} → remplit le placeholder {input} du prompt
        #   (les autres placeholders — tools, tool_names, agent_scratchpad —
        #    sont remplis automatiquement par AgentExecutor)
        #
        # En Java : agentExecutor.execute(Map.of("input", mission))
        #
        # invoke() est SYNCHRONE (bloquant) — il attend que l'agent
        # ait terminé TOUTES ses itérations ReAct avant de retourner.
        # Le terminal affiche le raisonnement en temps réel grâce à verbose=True.
        #
        # Retour : un dict LangChain avec au minimum :
        #   {"output": "La réponse finale de l'agent (Final Answer)", ...}
        # ─────────────────────────────────────────────────────────────
        result = agent_executor.invoke({"input": mission})
        # ↑ Déclenche la boucle ReAct complète
        #   → le terminal va afficher Thought → Action → Observation → ...
        #   → s'arrête sur "Final Answer" ou après MAX_ITERATIONS

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : Booléens Python — True / False avec majuscule
        #
        # En Java : boolean success = true;  → minuscule
        # En Python : "success": True        → MAJUSCULE
        #
        # C'est l'une des différences de syntaxe les plus piégeuses !
        #   Python : True, False, None  ← majuscule obligatoire
        #   Java   : true, false, null  ← minuscule
        #
        # Si tu écris "true" en Python → NameError : name 'true' is not defined
        # ─────────────────────────────────────────────────────────────
        return {
            "answer":  result.get("output", "Aucune réponse générée."),
            # ↑ result.get("output", ...) : dict.get() avec valeur par défaut
            #   (pattern déjà vu dans rag.py)
            #   "output" = clé LangChain pour la réponse finale de l'agent
            "mission": mission,
            "success": True        # ← MAJUSCULE — booléen Python
        }

    except Exception as e:
        error_msg = f"Erreur durant l'exécution de l'agent : {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "answer":  error_msg,
            "mission": mission,
            "success": False   # ← MAJUSCULE — booléen Python
        }


# ═══════════════════════════════════════════════════════════════════════
# BLOC 6 : POINT D'ENTRÉE POUR TESTS EN LIGNE DE COMMANDE
# ═══════════════════════════════════════════════════════════════════════

# Test en ligne de commande
if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : Concaténation implicite de strings entre parenthèses
    #
    # En Java, pour une longue string tu écrirais :
    #   String mission = "Recherche les informations sur les délais " +
    #                    "de validation dans les SOPs...";
    #
    # En Python, il y a une autre façon : mettre plusieurs strings
    # littérales côte à côte entre parenthèses — Python les colle
    # automatiquement à la compilation :
    #
    #   mission = (
    #       "Recherche les informations sur les délais "
    #       "de validation dans les SOPs "
    #       "disponibles."
    #   )
    #   # → "Recherche les informations sur les délais de validation dans les SOPs disponibles."
    #
    # IMPORTANT : Cela ne fonctionne QU'avec des strings LITTÉRALES côte à côte.
    #   "a" "b"   → "ab"    ← fonctionne (concaténation à la compilation)
    #   "a" + "b" → "ab"    ← fonctionne (opérateur +, à l'exécution)
    #   x "b"     → SyntaxError ← ne fonctionne pas avec une variable
    #
    # Avantage : pas besoin du "+" de Java, code plus lisible pour les longs textes.
    # ─────────────────────────────────────────────────────────────────
    mission_test = (
        "Recherche les informations sur les délais de validation dans les SOPs "
        "disponibles. Génère ensuite un rapport structuré avec les résultats trouvés."
    )
    # ↑ Deux strings littérales entre parenthèses → collées automatiquement

    result = run_agent(mission_test)

    print("\n" + "="*60)
    print("RÉSULTAT FINAL :")
    print("="*60)
    print(result["answer"])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONCLUSION DU FICHIER                                               ║
# ║                                                                      ║
# ║  🎯 CONCEPTS PYTHON CLÉS VUS ICI                                     ║
# ║                                                                      ║
# ║  1. Pattern ReAct : Thought → Action → Observation (boucle)         ║
# ║     → le LLM raisonne en plusieurs étapes comme un humain           ║
# ║     → AgentExecutor gère la boucle, le parsing, les outils          ║
# ║                                                                      ║
# ║  2. True / False / None → MAJUSCULE en Python                       ║
# ║     → ≠ true/false/null Java (minuscule)                            ║
# ║     → erreur la plus fréquente des devs Java qui découvrent Python  ║
# ║                                                                      ║
# ║  3. temperature=0.1 → contrôle du hasard du LLM                    ║
# ║     → 0.0 = déterministe, 1.0 = très aléatoire                     ║
# ║     → faible pour du raisonnement, plus élevé pour de la créativité ║
# ║                                                                      ║
# ║  4. Concaténation implicite : ("partie 1 " "partie 2")              ║
# ║     → Python colle les strings littérales côte à côte              ║
# ║     → uniquement pour les littérales, pas les variables             ║
# ║                                                                      ║
# ║  5. agent_executor.invoke({"input": mission})                       ║
# ║     → déclenche la boucle ReAct complète et attend le résultat     ║
# ║     → retourne {"output": "réponse finale"}                        ║
# ║                                                                      ║
# ║  ⚠️  PIÈGES À ÉVITER                                                 ║
# ║                                                                      ║
# ║  - True/False/None en MAJUSCULE — jamais true/false/null           ║
# ║  - Les 4 placeholders {tools} {tool_names} {input} {agent_scratchpad}║
# ║    sont OBLIGATOIRES dans le prompt ReAct LangChain                 ║
# ║  - temperature=0 ≠ temperature non définie : toujours expliciter   ║
# ║  - invoke() est bloquant : dans une UI, prévoir un indicateur       ║
# ║    de chargement (Streamlit spinner, par exemple)                   ║
# ║                                                                      ║
# ║  🔗 CONNEXION AVEC L'ARCHITECTURE                                    ║
# ║                                                                      ║
# ║  Ce fichier est le pivot de la partie "Agent" :                    ║
# ║    ← src/tools.py   : fournit AGENT_TOOLS (les 4 outils)           ║
# ║    ← src/rag.py     : appelé indirectement via search_documents()  ║
# ║    → agent_app.py   : appelle run_agent() depuis l'UI Streamlit    ║
# ║                                                                      ║
# ║  DIFFÉRENCE RAG vs AGENT :                                          ║
# ║    app.py      → appelle ask()       → 1 réponse, 1 appel LLM      ║
# ║    agent_app.py → appelle run_agent() → N appels LLM (boucle ReAct)║
# ╚══════════════════════════════════════════════════════════════════════╝
