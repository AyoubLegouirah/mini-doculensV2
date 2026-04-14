# agent_app.py

import os
import streamlit as st
from src.agent import run_agent

st.set_page_config(
    page_title="Agent IA — PharmaCo Belgium",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def check_vectorstore_ready() -> bool:
    return os.path.exists("chroma_db") and bool(os.listdir("chroma_db"))


def display_sidebar():
    with st.sidebar:
        st.markdown("## 🤖 Agent IA")
        st.caption("PharmaCo Belgium — Analyse réglementaire")
        st.divider()

        if check_vectorstore_ready():
            st.success("🟢 Base documentaire prête")
        else:
            st.error("🔴 Base documentaire manquante")
            st.warning("Lance d'abord Mini DocuLens (app.py) pour indexer tes documents PDF.")

        st.divider()
        st.markdown("### Missions exemples")
        st.caption("Clique pour charger dans le champ")

        missions = [
            "Identifie toutes les clauses mentionnant des délais de validation dans les SOPs disponibles. Génère un rapport structuré.",
            "Compare les exigences qualité de nos SOPs internes avec les guidelines ICH Q10. Liste les écarts et génère un rapport.",
            "Recherche toutes les procédures liées à la gestion des déviations. Résume les points clés et génère un rapport.",
        ]

        for i, mission in enumerate(missions):
            if st.button(f"Exemple {i+1}", key=f"mission_{i}", use_container_width=True):
                st.session_state["mission_input"] = mission

        st.divider()
        st.markdown("### À propos")
        st.caption("Cet agent utilise le pattern ReAct (LangChain) avec Gemini. Il raisonne, choisit ses outils, et produit un rapport final.")


def main():
    st.title("🤖 Agent IA — Analyse Réglementaire")
    st.caption("Donne une mission complexe à l'agent. Il raisonnera, cherchera dans les documents, et produira un rapport.")

    display_sidebar()
    st.divider()

    st.markdown("### Décris ta mission")
    st.caption("Sois précis sur ce que tu veux analyser et ce que doit contenir le rapport final.")

    default_mission = st.session_state.get("mission_input", "")

    mission = st.text_area(
        label="Mission",
        value=default_mission,
        height=120,
        placeholder="Ex: Analyse les SOPs disponibles et identifie toutes les clauses mentionnant des délais de validation.",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        launch_button = st.button(
            "🚀 Lancer l'agent",
            type="primary",
            use_container_width=True,
            disabled=not check_vectorstore_ready()
        )

    if launch_button and mission.strip():
        st.divider()
        st.markdown("### ⚙️ Exécution en cours...")
        st.caption("L'agent raisonne et appelle ses outils. Cela peut prendre 1 à 2 minutes.")

        progress_bar = st.progress(0, text="Initialisation de l'agent...")
        status_placeholder = st.empty()

        try:
            progress_bar.progress(20, text="🧠 L'agent réfléchit...")
            status_placeholder.info("L'agent analyse ta mission et choisit ses outils...")

            result = run_agent(mission)

            progress_bar.progress(100, text="✅ Mission terminée !")
            status_placeholder.empty()

            st.divider()

            if result["success"]:
                st.success("✅ Mission accomplie !")
                st.markdown("### 📋 Réponse de l'agent")
                st.markdown(result["answer"])

                # Rapports générés
                if os.path.exists("reports") and os.listdir("reports"):
                    st.divider()
                    st.markdown("### 📁 Rapports générés")

                    report_files = sorted(
                        [f for f in os.listdir("reports") if f.endswith(".md")],
                        reverse=True
                    )

                    for report_file in report_files[:3]:
                        report_path = os.path.join("reports", report_file)
                        with open(report_path, "r", encoding="utf-8") as f:
                            report_content = f.read()

                        with st.expander(f"📄 {report_file}", expanded=True):
                            st.markdown(report_content)
                            st.download_button(
                                label="⬇️ Télécharger",
                                data=report_content,
                                file_name=report_file,
                                mime="text/markdown",
                                key=f"download_{report_file}"
                            )

                # Graphiques générés
                if os.path.exists("reports"):
                    chart_files = sorted(
                        [f for f in os.listdir("reports") if f.endswith(".png")],
                        reverse=True
                    )
                    if chart_files:
                        st.divider()
                        st.markdown("### 📊 Graphiques générés")
                        for chart_file in chart_files[:3]:
                            chart_path = os.path.join("reports", chart_file)
                            st.image(chart_path, caption=chart_file, use_container_width=True)

            else:
                st.error("❌ Une erreur est survenue durant l'exécution.")
                st.code(result["answer"])

            # Affiche les graphiques même en cas d'erreur partielle
            if os.path.exists("reports"):
                chart_files = sorted(
                    [f for f in os.listdir("reports") if f.endswith(".png")],
                    reverse=True
                )
                if chart_files:
                    st.divider()
                    st.markdown("### 📊 Graphiques générés")
                    for chart_file in chart_files[:3]:
                        chart_path = os.path.join("reports", chart_file)
                        st.image(chart_path, caption=chart_file, use_container_width=True)

        except Exception as e:
            progress_bar.empty()
            status_placeholder.empty()
            st.error(f"Erreur inattendue : {str(e)}")

    elif launch_button and not mission.strip():
        st.warning("⚠️ Décris une mission avant de lancer l'agent.")


if __name__ == "__main__":
    main()