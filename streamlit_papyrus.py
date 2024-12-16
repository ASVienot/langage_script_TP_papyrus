import streamlit as st
import pandas as pd 
import numpy as np
from annotated_text import annotated_text
import ast 
# import matplotlib as plt
import matplotlib.pyplot as plt
import unicodedata
from collections import Counter

def lecture_df(path): 
    """Cette fonction lit le csv clean_papyrus-corpus.csv ou clean_papyrus-corpus_augmente.csv"""
    df = pd.read_csv(path)
    return df

def transforme_en_list(string):
    """ Transforme une string au format list en vrai object python list """
    list = string.strip("[").strip("]").replace("'","").strip("(").strip(")").split(",")
    return list

def normalize(text):
    """Cette fonction normalise le texte en enlevant les diactitiques et en le mettant en minuscule"""
    # enlever les diacritics / mettre en minuscule 
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c) and c not in "{}"])
    return text.replace('*','').lower()
    # result = u"".join([unicodedata.normalize("NFD", x)[0] for x in word])
    # # enlever les <>, (), {} ?[]?
    # # result = result.replace("<",'').replace(">",'').replace("{",'').replace("}",'').replace("(",'').replace(")",'')
    # return result

def annotation(text, dico_irreg): 
    """
    Cette fonction annote un texte selon un dictionnaire qui retourne une liste de mots et de tuples (mot et mot modifie)
    """
    list_text = (normalize(text)).split()
    list_annotated = []
    for word1 in list_text: 
        word = normalize(word1)
        if word in dico_irreg.keys(): 
            list_annotated.append((f"{word} ", dico_irreg[word]))
        else: 
            list_annotated.append(f"{word} ")
    return list_annotated

def page_acceuil():
    """Affichage de la Page d'acceuil - quand aucun papyrus n'est selectionné  """
# Page acceuil sans select papyrus
    st.image("crosby-schoyen-mississippi-codex-cnn.jpg", caption = "Le Crosby-Schøyen Codex")
    st.markdown("""Cette application permet de récupérer les informations suivantes sur les papyrus de notre base de données:   
                :scroll: Date,    
                :scroll: Provenance,     
                :scroll: Listes des personnes mentionnées,      
                :scroll: Listes des lieux mentionnées,    
                :scroll: Listes de coordonnées géographiques des lieux mentionnés,       
                :scroll: Texte annoté avec les irrégularités graphiques  """)
    st.markdown("""Pour accéder aux papyrus désirés, il est possible de passer le corpus par des filtres.    
                Les fitres qui ont été implémenter sont:    
                :scroll: Un filtre de provenance,  
                :scroll: Un filtre sur un interval de dates,  
                :scroll: Un filtre sur le genre du document.   """)
    st.markdown("""Une page de visualisation des données est aussi disponible.     """)

def affichage_papy(data, papyrus_selected, data1):
    """Affichage des informations sur le papyrus selectionné"""

    st.markdown(f"<h1 style='text-align: center;'> Papyrus n°{papyrus_selected}</h1>", unsafe_allow_html=True)

    # Affichage des données sur les papyrus
    list_label = ["Date", "Provenance","Personnes", "Lieux","Genre"]
    papyrus = data.loc[data["ID"] == papyrus_selected]
    papy_date = str(papyrus["Date"].values[0])
    papy_people = papyrus["People List"].values[0]

    for label in list_label: 
        col1, col2 = st.columns(2)

        with col1: 
            st.markdown(f"**{label}**")

        with col2: 
            if label == "Date": 
                st.write(papy_date)
            elif label == "Provenance": 
                st.write(papyrus["Provenance"].values[0])
            elif label == "Personnes":
                with st.expander("Personnes dans le papyrus"):
                    for people in set(papy_people): 
                        if people:
                            autre_papy = data1[data1["People List"].apply(lambda x: people in x)]
                            results = (autre_papy["ID"].tolist())
                            if st.button(people, icon="👤", use_container_width=True):
                                st.markdown(results)
                        else: 
                            st.write("Pas d'informations dans cette rubrique")              
            elif label == "Lieux":
                # transforme en dico
                dico_lieux = papyrus["Places List"].values[0]
                with st.expander("Lieux mentionnés dans le papyrus"):
                    if len(set(dico_lieux.keys()))>0:
                        for place in set(dico_lieux.keys()):
                            lieux_papy = data1[data1["Places List"].apply(lambda x: place in x.keys())]
                            lieux = lieux_papy["ID"].tolist()
                            if st.button(place, icon ="🌐",use_container_width=True):
                                st.markdown(lieux)
                    else: 
                        st.write("Pas d'informations dans cette rubrique")
            elif label == "Genre": 
                genre = str(papyrus["Content (beta!)"].values[0])
                st.write(genre)
            else: 
                st.write("Pas d'informations dans cette rubrique")

    st.write("**Text annotated with textual irregularities**")
    text = papyrus['Text Clean'].values[0]
    list_text_annotated = annotation(text, (ast.literal_eval(papyrus["Clean irregularities"].values[0]))) 
    annotated_text(list_text_annotated)

def visualisation(data):
    """ Affichage de plots sur nos données sur les papyrus """
    
    # plot des villes de provenance
    st.markdown("### 1) Distribution des villes d'où viennent les papyri")
    list_provenance = data['Ville Provenance'].tolist()
    provenance = Counter(list_provenance)
    fig = plt.figure(figsize=(8,8))
    plt.bar(provenance.keys(), provenance.values())
    plt.xticks(rotation=90)
    plt.xlabel("Villes")
    plt.ylabel("Nombres de papyrus")
    st.pyplot(fig)

    #plot des genres
    st.markdown("### 2) Distribution des genres de documents dans le corpus")
    fig2= plt.figure(figsize=(8,8))
    genre = Counter(data["Clean genre"].tolist())
    plt.pie(genre.values(), labels=genre.keys(), autopct='%1.1f%%')
    st.pyplot(fig2)

    # Map des lieux mentionnés 
    df_coord = lecture_df("tableau_coord_geo.csv")
    st.markdown("### 3) Carte des lieux mentionnés dans les papyrus")
    st.map(df_coord, latitude="Lat", longitude="Long")    
    st.markdown("""On peut voir sur la carte ci-desssus que la plus part des papyrus mentionnent des lieux proches des lieux d'origines de nos papyrus.   
                On voit par exemple qu'un grand nombre de lieux mentionnés se trouvent en Egypte sur les bords du Nil.   
                Les autres points sont en Turquie et un des points est en Italie à Rome. """)

# Date and uncertain portion
    st.markdown("### 4) Proportion de texte incertain en fonction de la date d'écriture")
    uncertain = data["Uncertain Portion"].tolist()
    dates = data["Clean dates intervals"].tolist()
    moyenne = [(int(debut) + int(fin)) / 2 for debut, fin in dates]
    fig3= plt.figure(figsize=(8,8))
    plt.scatter(moyenne,uncertain)
    plt.xlabel("Dates")
    plt.ylabel("Pourcentages d'incertitude")
    st.pyplot(fig3)

# Genre and uncertain portion
    st.markdown("### 4) Proportion de texte incertain en fonction du genre du papyrus ")
    uncertain = data["Uncertain Portion"].tolist()
    genre = data["Clean genre"].tolist()
    fig4= plt.figure(figsize=(8,8))
    plt.scatter(uncertain, genre)
    plt.ylabel("Genres")
    plt.xlabel("Pourcentages d'incertitude")
    st.pyplot(fig4)

def creation_page(data1):
    """
    Creation de la page 
    """
    st.markdown("<h1 style='text-align: center;'>La Chasse au Papyrus</h1>", unsafe_allow_html=True) # Pourquoi unsafe ? 
    st.sidebar.image("Mercote-removebg-preview.png", caption = "Mercote")
    # queslques traitement pour que les dates et les gens et les lieux soient propres
    data1["Clean dates intervals"] = data1["Clean dates intervals"].map(transforme_en_list)
    data1["Coord Geo"]= data1["Coord Geo"].map(transforme_en_list)
    data1["People List"] = data1["People List"].map(transforme_en_list)
    data1["Places List"] = data1["Places List"].map(ast.literal_eval)
    data = data1

    graph_papy = st.sidebar.selectbox("Graphiques ou papyrus", ("Visualisation des données", "Recherche de papyrus"), index=None, placeholder="Choisissez ce que vous voulez voir" )
    
    if graph_papy == "Visualisation des données":
        visualisation(data1)

    else:  
        # Option filtre provenance
        ville_provenance = st.sidebar.selectbox(
        "Filtre sur la provenance du papyrus",
        (set(data["Ville Provenance"].tolist())), index=None, placeholder ="Choisissez une ville" )
    
        # Option filtre date 
        intervals = data["Clean dates intervals"].tolist()
        # Trouver le max et le min 
        list_date_dens = []
        for interval in intervals : 
            if len(interval) == 1:
                list_date_dens.extend([int(interval[0])])
            elif len(interval) == 2: 
                list_date_dens.extend([int(interval[0]), int(interval[1])])  
        min_date = min(list_date_dens)
        max_date = max(list_date_dens)
        # Creer la slidebar de la date
        interval = st.sidebar.slider("Filtre sur l'interval de date d'écriture du papyrus", min_value=min_date, max_value=max_date, value=(min_date, max_date), step=1, key=None, help=None, on_change=None, args=None, kwargs=None, label_visibility="visible")
        
        # Option filtre genre 
        genre = st.sidebar.selectbox("Filtrer sur le genre du papyrus", (set(data["Clean genre"].tolist())), index=None, placeholder ="Choisissez une genre" ) 

        # si un ville est selectionné filtre sur la ville
        if ville_provenance: 
            data = data[data["Ville Provenance"]== ville_provenance]
        # si un interval de dates est selectionne on filtre les papyrus 
        if interval: 
            debut, fin = interval
            data = data[data["Clean dates intervals"].apply(lambda year_list: int(year_list[0]) >= debut and int(year_list[1]) <= fin)]
        if genre: 
            data = data[data["Clean genre"] == genre]
    
        # Bar de selection des papyrus
        papyrus_selected = st.sidebar.selectbox("Papyrus", (data["ID"].tolist()), index =None, placeholder ="Choisissez un papyrus")


        if papyrus_selected: 
            affichage_papy(data, papyrus_selected ,data1)    
        else: 
            page_acceuil()


def main():
    data = lecture_df("clean_papyrus-corpus_augmente.csv")
    creation_page(data)
    
if __name__ == '__main__':
    main()