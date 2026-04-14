# src/agent.py

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from src.tools import AGENT_TOOLS

load_dotenv()

# --- Configuration ---
AGENT_MODEL = "gemini-2.5-flash"
MAX_ITERATIONS = 10  # Sécurité : l'agent s'arrête après 10 étapes max


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

    # Le LLM — même modèle que le Projet 1
    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL,
        google_api_key=api_key,
        temperature=0.1  # Faible température = raisonnement plus stable
    )

    # Le prompt ReAct structuré
    prompt = PromptTemplate(
        template=AGENT_PROMPT_TEMPLATE,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
    )

    # Création de l'agent ReAct
    agent = create_react_agent(
        llm=llm,
        tools=AGENT_TOOLS,
        prompt=prompt
    )

    # L'orchestrateur qui gère la boucle
    return AgentExecutor(
        agent=agent,
        tools=AGENT_TOOLS,
        verbose=True,        # Affiche le raisonnement dans le terminal
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True  # Évite les crashes si Gemini formate mal
    )


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
        agent_executor = create_agent()
        result = agent_executor.invoke({"input": mission})

        return {
            "answer": result.get("output", "Aucune réponse générée."),
            "mission": mission,
            "success": True
        }

    except Exception as e:
        error_msg = f"Erreur durant l'exécution de l'agent : {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "answer": error_msg,
            "mission": mission,
            "success": False
        }


# Test en ligne de commande
if __name__ == "__main__":
    mission_test = (
        "Recherche les informations sur les délais de validation dans les SOPs "
        "disponibles. Génère ensuite un rapport structuré avec les résultats trouvés."
    )

    result = run_agent(mission_test)

    print("\n" + "="*60)
    print("RÉSULTAT FINAL :")
    print("="*60)
    print(result["answer"])