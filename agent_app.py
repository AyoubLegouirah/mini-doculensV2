# agent_app.py
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FICHE D'IDENTITÉ DU FICHIER                                         ║
# ║                                                                      ║
# ║  Rôle    : Interface Streamlit de l'agent ReAct.                    ║
# ║            Permet de donner des missions complexes à l'agent,       ║
# ║            suivre son raisonnement, et télécharger ses livrables    ║
# ║            (rapports Markdown + graphiques PNG).                    ║
# ║                                                                      ║
# ║  En Java : @Controller pour un workflow long — comme une vue        ║
# ║            Thymeleaf avec polling d'un job Spring Batch asynchrone. ║
# ║                                                                      ║
# ║  En Angular : Composant "dashboard" avec :                          ║
# ║               - Formulaire de lancement de job (mission)            ║
# ║               - Barre de progression pendant l'exécution            ║
# ║               - Section résultats avec téléchargement               ║
# ║                                                                      ║
# ║  Lancement : streamlit run agent_app.py                             ║
# ║              (port différent de app.py si les deux tournent)        ║
# ║                                                                      ║
# ║  Flux    :                                                           ║
# ║    Mission (str) → run_agent() → result {answer, success}          ║
# ║      → affichage réponse                                            ║
# ║      → lecture reports/*.md  → affichage + téléchargement          ║
# ║      → lecture reports/*.png → affichage graphiques                 ║
# ║                                                                      ║
# ║  Dépendances :                                                       ║
# ║    ← src/agent.py : run_agent()                                     ║
# ║    ← reports/     : lit les fichiers produits par les outils        ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════
# BLOC 1 : IMPORTS ET CONFIGURATION DE PAGE
# ═══════════════════════════════════════════════════════════════════════

import os
import streamlit as st
from src.agent import run_agent  # Point d'entrée de l'agent ReAct (src/agent.py)

# Configuration de la page — DOIT être le premier appel Streamlit
st.set_page_config(
    page_title="Agent IA — PharmaCo Belgium",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ═══════════════════════════════════════════════════════════════════════
# BLOC 2 : FONCTION check_vectorstore_ready — Vérification de la base
#
# Rôle : Vérifie que ChromaDB existe ET contient des fichiers.
#        Utilisée à deux endroits : dans la sidebar et pour désactiver
#        le bouton de lancement si la base n'est pas prête.
# ═══════════════════════════════════════════════════════════════════════

def check_vectorstore_ready() -> bool:
    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : bool() — conversion explicite en booléen
    #
    # Dans app.py on avait : os.path.exists(...) and os.listdir(...)
    # Ici on a : os.path.exists(...) and bool(os.listdir(...))
    #
    # os.listdir() retourne une LISTE de noms de fichiers.
    # Une liste non vide est "truthy" en Python (elle vaut True dans un contexte booléen).
    # Une liste vide [] est "falsy" (vaut False).
    #
    # bool([]) → False    (liste vide)
    # bool(["fichier.txt"]) → True  (liste non vide)
    #
    # bool() est une conversion EXPLICITE — équivalent Java : !list.isEmpty()
    # En pratique, "and os.listdir(...)" et "and bool(os.listdir(...))"
    # fonctionnent pareil ici, mais bool() rend l'intention plus claire.
    #
    # En Java :
    #   return Files.exists(Path.of("chroma_db")) &&
    #          !new File("chroma_db").list().isEmpty();
    # ─────────────────────────────────────────────────────────────────
    return os.path.exists("chroma_db") and bool(os.listdir("chroma_db"))
    # ↑ Retour direct d'une expression booléenne — pas besoin de variable intermédiaire
    #   En Java : return condition1 && condition2;


# ═══════════════════════════════════════════════════════════════════════
# BLOC 3 : FONCTION display_sidebar — Contenu de la barre latérale
#
# MICRO-COURS : Extraire le code Streamlit dans des fonctions
#
# Dans app.py, tout le code était au niveau du module (pas dans des fonctions).
# Ici, la sidebar est encapsulée dans display_sidebar().
#
# LES DEUX APPROCHES SONT VALIDES en Streamlit.
# Mettre du code Streamlit dans des fonctions :
#   ✅ Améliore la lisibilité pour les interfaces complexes
#   ✅ Permet de réutiliser des blocs UI
#   ✅ main() devient plus lisible (structure claire)
#
# RAPPEL : la fonction est appelée depuis main() à chaque rechargement.
# Le comportement est identique — juste mieux organisé.
# ═══════════════════════════════════════════════════════════════════════

def display_sidebar():
    with st.sidebar:  # Tout ce bloc va dans la barre latérale gauche
        st.markdown("## 🤖 Agent IA")
        # ↑ "##" = H2 en Markdown — un niveau en dessous de st.title() (H1)

        st.caption("PharmaCo Belgium — Analyse réglementaire")
        # ↑ MICRO-COURS : st.caption() — texte gris plus petit
        #   En HTML/CSS : <p style="color: gray; font-size: 0.8em;">
        #   En Angular  : <mat-hint> ou <small class="text-muted">
        #   Utilisé pour les sous-titres discrets, les notes explicatives.

        st.divider()

        # Indicateur d'état de la base vectorielle
        if check_vectorstore_ready():
            st.success("🟢 Base documentaire prête")
        else:
            st.error("🔴 Base documentaire manquante")
            st.warning("Lance d'abord Mini DocuLens (app.py) pour indexer tes documents PDF.")

        st.divider()
        st.markdown("### Missions exemples")
        st.caption("Clique pour charger dans le champ")

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : Liste de strings comme source de données
        #
        # En Java : List<String> missions = Arrays.asList("mission1", "mission2", "mission3");
        # En Python : missions = ["mission1", "mission2", "mission3"]
        #
        # C'est une liste de chaînes sur plusieurs lignes.
        # Python permet d'indenter les éléments d'une liste sans problème.
        # La virgule après le dernier élément est optionnelle en Python
        # (contrairement à certains langages).
        # ─────────────────────────────────────────────────────────────
        missions = [
            "Identifie toutes les clauses mentionnant des délais de validation dans les SOPs disponibles. Génère un rapport structuré.",
            "Compare les exigences qualité de nos SOPs internes avec les guidelines ICH Q10. Liste les écarts et génère un rapport.",
            "Recherche toutes les procédures liées à la gestion des déviations. Résume les points clés et génère un rapport.",
        ]

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : key= — identifiant unique des widgets en boucle
        #
        # PROBLÈME STREAMLIT :
        #   Quand tu crées plusieurs widgets du même type dans une boucle
        #   (ici 3 boutons), Streamlit doit pouvoir les distinguer.
        #   Sans key=, Streamlit lève une DuplicateWidgetID exception.
        #
        # SOLUTION : donner un identifiant unique à chaque widget via key=
        #   key=f"mission_{i}" → "mission_0", "mission_1", "mission_2"
        #
        # Analogie Angular/HTML :
        #   <button id="mission-0">...</button>
        #   <button id="mission-1">...</button>
        #   L'id HTML doit être unique dans la page.
        #
        # RÈGLE : Toujours mettre key= sur les widgets créés dans une boucle.
        #
        # Note : on utilise i+1 dans le label ("Exemple 1", "Exemple 2"...)
        # pour l'affichage, mais l'index Python commence à 0 (enumerate sans start=).
        # ─────────────────────────────────────────────────────────────
        for i, mission in enumerate(missions):   # enumerate sans start= → i commence à 0
            if st.button(f"Exemple {i+1}", key=f"mission_{i}", use_container_width=True):
                # ↑ i+1 dans le label : affiche "Exemple 1", "Exemple 2", "Exemple 3"
                # ↑ key=f"mission_{i}" : clé unique "mission_0", "mission_1", "mission_2"

                # Stocke la mission dans session_state → prérempli le textarea
                st.session_state["mission_input"] = mission
                # ↑ Au prochain rechargement (déclenché par ce clic),
                #   main() lira st.session_state.get("mission_input", "")
                #   et initialisera le textarea avec ce texte.

        st.divider()
        st.markdown("### À propos")
        st.caption("Cet agent utilise le pattern ReAct (LangChain) avec Gemini. Il raisonne, choisit ses outils, et produit un rapport final.")


# ═══════════════════════════════════════════════════════════════════════
# BLOC 4 : FONCTION main — Interface principale
#
# Rôle : Construit l'interface principale de la page :
#        titre, textarea de mission, bouton de lancement,
#        barre de progression, et affichage des résultats.
# ═══════════════════════════════════════════════════════════════════════

def main():
    st.title("🤖 Agent IA — Analyse Réglementaire")
    st.caption("Donne une mission complexe à l'agent. Il raisonnera, cherchera dans les documents, et produira un rapport.")

    display_sidebar()  # Affiche le contenu de la sidebar (appel de fonction normale)
    st.divider()

    st.markdown("### Décris ta mission")
    st.caption("Sois précis sur ce que tu veux analyser et ce que doit contenir le rapport final.")

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : st.session_state.get() — lire avec valeur par défaut
    #
    # st.session_state est un dict spécial Streamlit.
    # On peut l'utiliser comme n'importe quel dict Python :
    #   st.session_state["clé"] = valeur   → écriture
    #   st.session_state["clé"]            → lecture (KeyError si absent)
    #   st.session_state.get("clé", "")    → lecture sécurisée avec défaut
    #
    # Ici, si l'utilisateur n'a pas encore cliqué sur un bouton exemple,
    # "mission_input" n'existe pas dans session_state.
    # .get("mission_input", "") retourne "" plutôt que de lever une KeyError.
    #
    # En Java : (String) session.getAttribute("mission_input") avec null check
    # ─────────────────────────────────────────────────────────────────
    default_mission = st.session_state.get("mission_input", "")
    # ↑ "" si aucun exemple cliqué, sinon la mission de l'exemple cliqué

    # ─────────────────────────────────────────────────────────────────
    # MICRO-COURS : st.text_area() vs st.text_input()
    #
    # st.text_input() : champ de saisie UNE SEULE LIGNE (vu dans app.py)
    #   En HTML : <input type="text">
    #   Utilisé pour : mots-clés, noms, courtes questions
    #
    # st.text_area() : champ de saisie MULTI-LIGNES
    #   En HTML/Angular : <textarea> ou <mat-form-field><textarea matInput>
    #   Utilisé pour : textes longs, missions, descriptions, code
    #
    # Paramètres spécifiques à text_area :
    #   value=default_mission : valeur initiale (prérempli depuis session_state)
    #   height=120            : hauteur en pixels
    # ─────────────────────────────────────────────────────────────────
    mission = st.text_area(
        label="Mission",
        value=default_mission,   # Prérempli si un exemple a été cliqué
        height=120,              # Hauteur fixe en pixels
        placeholder="Ex: Analyse les SOPs disponibles et identifie toutes les clauses mentionnant des délais de validation.",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : disabled= — désactiver un widget conditionnellement
        #
        # En Angular : <button [disabled]="!isReady()">Lancer</button>
        # En HTML    : <button disabled>Lancer</button>
        #
        # Streamlit : disabled=True → bouton grisé, non cliquable
        #             disabled=False → bouton actif
        #
        # Ici : disabled=not check_vectorstore_ready()
        #   Si la base vectorielle N'EST PAS prête → disabled=True (bouton grisé)
        #   Si la base vectorielle EST prête       → disabled=False (bouton actif)
        #
        # "not check_vectorstore_ready()" : l'opérateur "not" inverse le booléen
        #   check_vectorstore_ready() retourne True  → not True  = False (actif)
        #   check_vectorstore_ready() retourne False → not False = True  (désactivé)
        #
        # INTÉRÊT : l'expression est évaluée à CHAQUE rechargement Streamlit
        # → le bouton se réactive automatiquement quand la base est prête,
        # sans que l'utilisateur ait besoin de rafraîchir la page.
        # ─────────────────────────────────────────────────────────────
        launch_button = st.button(
            "🚀 Lancer l'agent",
            type="primary",
            use_container_width=True,
            disabled=not check_vectorstore_ready()  # Grisé si ChromaDB absent
        )

    if launch_button and mission.strip():
        st.divider()
        st.markdown("### ⚙️ Exécution en cours...")
        st.caption("L'agent raisonne et appelle ses outils. Cela peut prendre 1 à 2 minutes.")

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : st.progress() — barre de progression
        #
        # En Angular Material : <mat-progress-bar mode="determinate" [value]="progress">
        # En HTML/CSS : <progress value="20" max="100">
        #
        # st.progress(valeur, text="message") :
        #   valeur : nombre entre 0 et 100 (pourcentage)
        #   text   : message affiché à côté de la barre
        #
        # La barre est MISE À JOUR en appelant progress_bar.progress(nouvelle_valeur).
        # C'est l'un des rares cas où Streamlit permet une mise à jour partielle
        # sans tout recharger.
        # ─────────────────────────────────────────────────────────────
        progress_bar = st.progress(0, text="Initialisation de l'agent...")
        # ↑ Crée une barre à 0% avec le message

        # ─────────────────────────────────────────────────────────────
        # MICRO-COURS : st.empty() — placeholder mutable
        #
        # st.empty() crée un espace VIDE dans l'interface qui peut être
        # rempli ou modifié PLUS TARD dans le script.
        #
        # En Angular : <ng-container *ngIf="..."> ou un <div> qu'on cache/montre
        # En JavaScript : document.getElementById("placeholder").innerHTML = "..."
        #
        # Pourquoi ici ?
        #   On veut afficher un message pendant l'exécution de l'agent,
        #   puis le VIDER quand c'est terminé (status_placeholder.empty()).
        #
        #   Sans st.empty(), chaque st.info() ajouterait un nouveau message
        #   dans la page sans effacer le précédent.
        #
        #   Avec st.empty(), on a UN SEUL espace qu'on remplace :
        #     status_placeholder.info("En cours...")  → affiche le message
        #     status_placeholder.empty()              → efface ce même espace
        # ─────────────────────────────────────────────────────────────
        status_placeholder = st.empty()
        # ↑ Placeholder vide — sera rempli/modifié dynamiquement

        try:
            progress_bar.progress(20, text="🧠 L'agent réfléchit...")
            # ↑ Met à jour la barre à 20% avec nouveau message
            status_placeholder.info("L'agent analyse ta mission et choisit ses outils...")
            # ↑ Affiche un message dans le placeholder (remplace le vide)

            result = run_agent(mission)  # ← APPEL BLOQUANT : peut prendre 1-2 minutes
            # ↑ La barre de progression ne bouge plus pendant run_agent()
            #   car Python est mono-thread et attend la fin de run_agent().
            #   Le progress à 20% donne juste un retour visuel initial.

            progress_bar.progress(100, text="✅ Mission terminée !")
            # ↑ Met à jour la barre à 100% quand l'agent a terminé
            status_placeholder.empty()
            # ↑ Efface le message "L'agent analyse..." — l'espace est vidé

            st.divider()

            # ─────────────────────────────────────────────────────────
            # Affichage conditionnel selon le succès ou l'échec
            # result["success"] est un booléen (True/False) défini dans run_agent()
            # ─────────────────────────────────────────────────────────
            if result["success"]:
                st.success("✅ Mission accomplie !")
                st.markdown("### 📋 Réponse de l'agent")
                st.markdown(result["answer"])

                # ─────────────────────────────────────────────────────
                # Affichage des rapports Markdown générés par generate_report()
                # ─────────────────────────────────────────────────────
                # Rapports générés
                if os.path.exists("reports") and os.listdir("reports"):
                    st.divider()
                    st.markdown("### 📁 Rapports générés")

                    # ─────────────────────────────────────────────────
                    # MICRO-COURS : sorted() — trier une liste
                    #
                    # En Java :
                    #   List<String> sorted = new ArrayList<>(files);
                    #   Collections.sort(sorted, Collections.reverseOrder());
                    #
                    # En Python :
                    #   sorted_list = sorted(ma_liste, reverse=True)
                    #
                    # sorted() est une fonction BUILT-IN Python (comme len, print...).
                    # Elle retourne une NOUVELLE liste triée sans modifier l'originale.
                    # (≠ liste.sort() qui modifie la liste en place et retourne None)
                    #
                    # reverse=True : ordre décroissant (Z→A, 9→0)
                    # reverse=False (défaut) : ordre croissant (A→Z, 0→9)
                    #
                    # Ici : les fichiers sont nommés avec un timestamp "20241215_143022"
                    # Tri décroissant → le plus RÉCENT en premier
                    # ─────────────────────────────────────────────────
                    report_files = sorted(
                        [f for f in os.listdir("reports") if f.endswith(".md")],
                        # ↑ List comprehension : liste des fichiers .md dans reports/
                        reverse=True  # Plus récent d'abord (timestamp dans le nom)
                    )

                    # ─────────────────────────────────────────────────
                    # MICRO-COURS : Slicing de liste — liste[:3]
                    #
                    # En Java : list.subList(0, Math.min(3, list.size()))
                    # En Python : liste[:3]
                    #
                    # Le "slicing" (découpage) extrait une SOUS-LISTE :
                    #   liste[start:stop]  → de l'index start (inclus) à stop (exclu)
                    #   liste[:3]          → les 3 PREMIERS éléments (index 0, 1, 2)
                    #   liste[2:]          → du 3ème à la fin
                    #   liste[1:4]         → index 1, 2, 3
                    #   liste[-1]          → le DERNIER élément
                    #   liste[-3:]         → les 3 DERNIERS éléments
                    #
                    # Exemples simples :
                    #   nombres = [10, 20, 30, 40, 50]
                    #   nombres[:3]   → [10, 20, 30]
                    #   nombres[2:]   → [30, 40, 50]
                    #   nombres[-1]   → 50
                    #   nombres[-2:]  → [40, 50]
                    #
                    # [:3] ici : affiche max 3 rapports pour ne pas surcharger l'UI
                    # Si l'agent a généré 5 rapports, on n'en affiche que les 3 plus récents
                    # ─────────────────────────────────────────────────
                    for report_file in report_files[:3]:  # Au maximum les 3 rapports les plus récents
                        report_path = os.path.join("reports", report_file)

                        # ─────────────────────────────────────────────
                        # MICRO-COURS : open() en mode lecture "r"
                        #
                        # Dans tools.py on a vu "w" (écriture).
                        # Ici on utilise "r" (read = lecture).
                        #
                        # En Java :
                        #   try (BufferedReader br = new BufferedReader(
                        #            new FileReader(reportPath, StandardCharsets.UTF_8))) {
                        #       String content = br.lines().collect(Collectors.joining("\n"));
                        #   }
                        #
                        # En Python :
                        #   with open(report_path, "r", encoding="utf-8") as f:
                        #       report_content = f.read()
                        #
                        # f.read() : lit TOUT le contenu du fichier en une seule string
                        # Alternative : f.readlines() → retourne une liste de lignes
                        #               for line in f: → itère ligne par ligne (économe en mémoire)
                        #
                        # TOUJOURS spécifier encoding="utf-8" pour les fichiers texte
                        # → évite les problèmes d'accents (é, à, ç...) sur Windows
                        # ─────────────────────────────────────────────
                        with open(report_path, "r", encoding="utf-8") as f:
                            report_content = f.read()  # Lit tout le fichier Markdown en une string

                        # ─────────────────────────────────────────────
                        # MICRO-COURS : expanded=True dans st.expander()
                        #
                        # Dans display_sources() (app.py), on avait st.expander("titre")
                        # → sections FERMÉES par défaut, l'utilisateur déplie.
                        #
                        # Ici : st.expander("titre", expanded=True)
                        # → sections OUVERTES par défaut, visibles immédiatement.
                        #
                        # Choix UX : les rapports sont le résultat principal de l'agent
                        # → on les montre ouverts. Les sources RAG (app.py) sont
                        # secondaires → on les cache par défaut.
                        # ─────────────────────────────────────────────
                        with st.expander(f"📄 {report_file}", expanded=True):
                            # ↑ expanded=True : le panneau est ouvert par défaut
                            st.markdown(report_content)  # Affiche le Markdown du rapport rendu en HTML

                            # ─────────────────────────────────────────
                            # MICRO-COURS : st.download_button() — téléchargement
                            #
                            # En HTML : <a href="/reports/fichier.md" download>Télécharger</a>
                            # En Angular : HttpClient + Blob + window.URL.createObjectURL()
                            #
                            # Streamlit gère tout : le fichier est envoyé au navigateur
                            # directement depuis la mémoire Python.
                            #
                            # Paramètres :
                            #   label      : texte du bouton
                            #   data       : contenu du fichier (string ou bytes)
                            #   file_name  : nom du fichier téléchargé
                            #   mime       : type MIME ("text/markdown", "image/png", "application/pdf"...)
                            #   key        : OBLIGATOIRE car dans une boucle — doit être unique
                            #
                            # PIÈGE : sans key= dans une boucle → DuplicateWidgetID error
                            # ─────────────────────────────────────────
                            st.download_button(
                                label="⬇️ Télécharger",
                                data=report_content,          # Contenu string du fichier Markdown
                                file_name=report_file,        # Nom du fichier tel que téléchargé
                                mime="text/markdown",         # Type MIME pour le navigateur
                                key=f"download_{report_file}" # Clé unique → évite DuplicateWidgetID
                            )

                # ─────────────────────────────────────────────────────
                # Affichage des graphiques PNG générés par generate_chart()
                # ─────────────────────────────────────────────────────
                # Graphiques générés
                if os.path.exists("reports"):
                    chart_files = sorted(
                        [f for f in os.listdir("reports") if f.endswith(".png")],
                        reverse=True  # Plus récent en premier
                    )
                    if chart_files:
                        st.divider()
                        st.markdown("### 📊 Graphiques générés")

                        # ─────────────────────────────────────────────
                        # MICRO-COURS : st.image() — afficher une image
                        #
                        # En HTML/Angular : <img src="..." [alt]="caption">
                        #
                        # st.image() accepte :
                        #   - Un chemin de fichier local (str) → Streamlit lit le fichier
                        #   - Un objet PIL Image
                        #   - Un tableau numpy (matrice de pixels)
                        #   - Une URL (str commençant par "http")
                        #
                        # use_container_width=True : l'image s'adapte à la largeur disponible
                        #   En CSS : width: 100%;
                        # caption : texte affiché sous l'image
                        # ─────────────────────────────────────────────
                        for chart_file in chart_files[:3]:  # Max 3 graphiques récents
                            chart_path = os.path.join("reports", chart_file)
                            st.image(chart_path, caption=chart_file, use_container_width=True)

            else:
                # L'agent a retourné success=False (exception capturée dans run_agent)
                st.error("❌ Une erreur est survenue durant l'exécution.")

                # ─────────────────────────────────────────────────────
                # MICRO-COURS : st.code() — affichage monospace (bloc de code)
                #
                # En HTML : <pre><code>texte...</code></pre>
                #
                # st.code(texte) : affiche le texte dans un encadré
                # avec police monospace, fond gris, et bouton "Copier".
                # Utilisé ici pour afficher le message d'erreur lisiblement.
                # ─────────────────────────────────────────────────────
                st.code(result["answer"])  # Affiche le message d'erreur en monospace

            # ─────────────────────────────────────────────────────────
            # Affiche les graphiques même si success=False
            # (l'agent a pu générer un graphique avant de planter)
            # ─────────────────────────────────────────────────────────
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
            # En cas d'erreur non gérée (ex: problème réseau, API down...)
            progress_bar.empty()          # Efface la barre de progression
            status_placeholder.empty()    # Efface le message de statut
            st.error(f"Erreur inattendue : {str(e)}")

    elif launch_button and not mission.strip():
        # L'utilisateur a cliqué "Lancer" sans écrire de mission
        st.warning("⚠️ Décris une mission avant de lancer l'agent.")


# ═══════════════════════════════════════════════════════════════════════
# BLOC 5 : POINT D'ENTRÉE
#
# MICRO-COURS : Pattern "if __name__ == '__main__': main()"
#
# On a vu ce pattern dans ingest.py et rag.py.
# Ici il est utilisé AVEC une fonction main() explicite.
#
# Deux styles coexistent dans ce projet :
#
#   Style 1 — app.py : code au niveau du module + if __name__ == "__main__"
#     Tout le code Streamlit est directement dans le module.
#     Streamlit l'exécute quand il lance le script.
#
#   Style 2 — agent_app.py : tout dans main() + if __name__ == "__main__": main()
#     Le code Streamlit est encapsulé dans main().
#     Plus proche du style Java (public static void main).
#
# LES DEUX FONCTIONNENT : Streamlit ne fait pas de distinction.
# Le Style 2 (avec main()) est préféré pour les fichiers plus complexes
# car il structure mieux le code et facilite les tests.
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONCLUSION DU FICHIER ET DU PROJET                                  ║
# ║                                                                      ║
# ║  🎯 CONCEPTS PYTHON CLÉS VUS ICI                                     ║
# ║                                                                      ║
# ║  1. bool(liste) → True si non vide, False si vide                   ║
# ║     Conversion explicite — équivalent de !list.isEmpty() Java       ║
# ║                                                                      ║
# ║  2. liste[:3] → slicing, extrait les 3 premiers éléments            ║
# ║     liste[-1] → dernier élément                                     ║
# ║     liste[1:4] → éléments d'index 1, 2, 3                          ║
# ║                                                                      ║
# ║  3. sorted(liste, reverse=True) → liste triée décroissante          ║
# ║     Retourne une NOUVELLE liste (ne modifie pas l'originale)        ║
# ║     ≠ liste.sort() qui modifie en place et retourne None            ║
# ║                                                                      ║
# ║  4. f.read() → lit tout le contenu d'un fichier en une string       ║
# ║     open(path, "r", encoding="utf-8") → toujours UTF-8 pour le texte║
# ║                                                                      ║
# ║  5. key= obligatoire pour les widgets en boucle                     ║
# ║     Sans key= → DuplicateWidgetID exception Streamlit               ║
# ║                                                                      ║
# ║  ⚠️  PIÈGES À ÉVITER                                                 ║
# ║                                                                      ║
# ║  - sorted() ≠ liste.sort() : l'un retourne une nouvelle liste,     ║
# ║    l'autre modifie en place et retourne None                        ║
# ║    → "liste2 = sorted(liste1)" correct                              ║
# ║    → "liste2 = liste1.sort()" → liste2 vaut None !                 ║
# ║  - st.empty() n'est pas un widget caché — c'est un espace réservé  ║
# ║    remplaçable dynamiquement (différent de hidden/visible Angular)  ║
# ║  - [:3] sur une liste de 2 éléments → [elem1, elem2] (pas d'erreur)║
# ║    Python ne lève pas d'IndexError pour les slices hors limites     ║
# ║                                                                      ║
# ║  🔗 CONNEXION AVEC L'ARCHITECTURE — VUE GLOBALE DU PROJET           ║
# ║                                                                      ║
# ║  PIPELINE DE DONNÉES COMPLET :                                       ║
# ║                                                                      ║
# ║  src/prompts.py → constantes de prompts LLM                        ║
# ║       ↓                                                             ║
# ║  src/ingest.py  → PDF → chunks → vecteurs → chroma_db/             ║
# ║       ↓                                                             ║
# ║  src/rag.py     → question → ChromaDB → contexte → Gemini → réponse║
# ║       ↓              ↑ utilise prompts.py                          ║
# ║  src/tools.py   → outils @tool utilisant ask() + matplotlib        ║
# ║       ↓                                                             ║
# ║  src/agent.py   → boucle ReAct : mission → N outils → livrable     ║
# ║                                                                      ║
# ║  INTERFACES UTILISATEUR :                                            ║
# ║  app.py        → UI Streamlit RAG  (questions simples)             ║
# ║  agent_app.py  → UI Streamlit Agent (missions complexes)           ║
# ╚══════════════════════════════════════════════════════════════════════╝
