# src/tools.py

import os
import json
from datetime import datetime
from langchain.tools import tool
from src.rag import ask


@tool
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
    try:
        result = ask(query)
        
        # Formate la réponse pour l'agent : réponse + sources
        sources_text = "\n".join([
            f"  - {chunk['source']}, page {chunk['page']}"
            for chunk in result["chunks"]
        ])
        
        return f"{result['answer']}\n\nSources utilisées :\n{sources_text}"
    
    except FileNotFoundError:
        return "Erreur : aucune base documentaire trouvée. Lance d'abord l'ingestion des PDFs."
    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"


@tool
def generate_report(content: str, report_title: str = "Rapport d'analyse") -> str:
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
        # Fix : si LangChain passe un dict, on extrait les valeurs
        if isinstance(content, dict):
            report_title = content.get("report_title", report_title)
            content = content.get("content", str(content))

        # Crée le dossier reports/ s'il n'existe pas
        os.makedirs("reports", exist_ok=True)
        
        # Génère un nom de fichier unique avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = report_title.lower().replace(" ", "_").replace("/", "-")
        filename = f"reports/{safe_title}_{timestamp}.md"
        
        # Construit le rapport complet avec header
        report_content = f"""# {report_title}

**Généré le** : {datetime.now().strftime("%d/%m/%Y à %H:%M")}  
**Système** : Mini DocuLens — Agent IA PharmaCo Belgium

---

{content}

---
*Rapport généré automatiquement par l'Agent IA Mini DocuLens*
"""
        
        # Écrit le fichier
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        return f"Rapport généré avec succès : {filename}"
    
    except Exception as e:
        return f"Erreur lors de la génération du rapport : {str(e)}"


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
        # Import ici pour ne pas bloquer si non installé
        from langchain_community.tools import DuckDuckGoSearchRun
        
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result
    
    except ImportError:
        return (
            "Outil search_web non disponible : installe 'duckduckgo-search' "
            "avec : pip install duckduckgo-search"
        )
    except Exception as e:
        return f"Erreur lors de la recherche web : {str(e)}"
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
        import matplotlib
        matplotlib.use("Agg")  # Mode sans interface graphique
        import matplotlib.pyplot as plt

        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/chart_{timestamp}.png"

        # Parse les données format "catégorie:valeur, catégorie:valeur"
        categories = []
        values = []

        for item in data.split(","):
            item = item.strip()
            if ":" in item:
                parts = item.split(":", 1)
                categories.append(parts[0].strip())
                try:
                    values.append(float(parts[1].strip()))
                except ValueError:
                    values.append(1.0)

        # Si pas de données parsables, crée un graphique exemple
        if not categories:
            categories = ["Donnée 1", "Donnée 2", "Donnée 3"]
            values = [3, 5, 2]

        # Génère le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(categories, values, color="#4A90D9", edgecolor="white")

        # Ajoute les valeurs sur les barres
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=11
            )

        ax.set_title(chart_title, fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Nombre de références", fontsize=11)
        ax.set_xlabel("Catégories", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        return f"Graphique généré avec succès : {filename}"

    except Exception as e:
        return f"Erreur lors de la génération du graphique : {str(e)}"

# Liste exportée pour agent.py
AGENT_TOOLS = [search_documents, generate_report, search_web, generate_chart]