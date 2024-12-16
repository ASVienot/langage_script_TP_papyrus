import pandas as pd
import matplotlib.pyplot as plt 
from collections import Counter
import re 
from unidecode import unidecode
import unicodedata
from transformers import pipeline
import difflib as dl
import numpy as np 
import ast 
import seaborn as sns


def lecture_csv(path):
    """Lecture d'un chemin vers un fichier csv en une dataframe"""
    df = pd.read_csv(path)
    return df

def transforme_en_list(string):
    """transforme un object string au format liste en un objet liste"""
    list = string.strip("[").strip("]").replace("'","").split(",")
    return list

def remove_empty(df):
    """Retire les lignes qui n'ont pas de texte """
    df.dropna(subset=['Full Text'], inplace=True)
    return df 

def graph_genre(clean_df): 
    """Pie plot du genre des papyrus"""
    list_genre = clean_df['Content (beta!)'].tolist()
    list_genre_clean =[genre.lower().split(':', 1)[0].split('see',1)[0].split(' ',1)[0] for genre in list_genre]
    #TEST SI PLUS QU'UN MOT 
    # for genre in list_genre_clean: 
    #     if len(genre.split()) > 1: 
    #         print("merde")
    clean_df["Clean genre"] = list_genre_clean
    genre = Counter(list_genre_clean)
    # print(genre)
    plt.title('Distribution des genres de documents dans le corpus')
    plt.pie(genre.values(), labels=genre.keys(), autopct='%1.1f%%')
    plt.show()

def lieu(clean_df): 
    """Crée un graphique représentant la distribution des villes d'ou viennent les papyri"""
    list_provenance = clean_df['Provenance'].tolist()
    # print(list_provenance)
    list_provenance_clean =[provenance.lower().split(' ',1)[0].split('-', 1)[0].split('?', 1)[0] for provenance in list_provenance]
    clean_df["Ville Provenance"] = list_provenance_clean
    # print(list_provenance_clean)
    provenance = Counter(list_provenance_clean)
    
    ville_max = max(provenance, key= lambda x: provenance[x])

    fig = plt.figure(figsize = (5, 10))

    plt.bar(provenance.keys(), provenance.values(), width = 0.5)
    
    plt.xticks(rotation=90)
    plt.xlabel("Villes")
    plt.ylabel("Nombres de papyrus")
    plt.title("Distribution des villes d'ou viennent les papyri")
    plt.show()

    print(f"J'en déduis que les papyrus viennent en grande partie de la ville de {ville_max}.")

def reutilises(clean_df):
    """Imprime le nombre de papyrus réutilisés dans le corpus"""
    reu = clean_df[['Reuse note', 'Reuse type']].notna().any(axis=1).sum()
    print(f"Le nombre de papyrus réutilisé est {reu}.")        

def dates(clean_df):
    """Crée un plot de densité pour des intervals de dates"""
    list_dates_brut = clean_df['Date'].tolist()
    # print(list_dates_brut)
    intervals = []
    pattern = "AD \d{3}( [A-Z][a-z]{2} \d{2} - \d{3}| - \d{3}|$| )"
    pattern_chiffres = "\d{3}"
    for date in list_dates_brut:
        date_clean = date.replace('about', ' ').replace('?', ' ').replace('cf', ' ').replace('or', ' ').replace('after', ' ')
        matchs = re.search(pattern, date_clean)
        # print(matchs.group(0))
        dates = re.findall(pattern_chiffres, matchs.group(0))
        # print(dates)
        intervals.append(dates)
    # print(len(intervals))
    # print(type(intervals[0]))


    
    clean_df['Clean dates intervals'] = [(interval[0], interval[0]) if len(interval) == 1 else tuple(interval) for interval in intervals]
    
    list_date_dens = []
    # creation de la liste de données avec intérieur des intervals 
    for interval in intervals : 
        if len(interval) == 1:
            points = (np.arange(int(interval[0]), int(interval[0])+1, 1)).tolist()
            list_date_dens.extend(points)
        elif len(interval) ==2: 
            points= np.arange(int(interval[0]), int(interval[1])+1, 1).tolist()
            list_date_dens.extend(points)       
    # print(list_date_dens)

    # plot densité 
    sns.kdeplot(list_date_dens, common_norm=True, fill=True)
    plt.title("Density of papyrus peer years",)
    plt.xlabel("Year")
    plt.ylabel("Density")
    plt.grid()
    plt.show()

def nettoyage(text):
    """ Retire les chiffres et les caractères demandés d'un texte"""
    pattern_gap = "\|gap=.*\|" 
    text_clean = re.sub(pattern_gap, '', text)
    pattern_chiffre = r'[0-9]'
    text_clean = re.sub(pattern_chiffre,'', text_clean)
    # text_clean.strip()
    text_clean = text_clean.replace("†", "").replace("⳨", "")

    return text_clean

def uncertain(text): 
    """Calcule le pourcentage de caractères incertains (caractères avec points dessous)"""
    uncertain = []
    # pattern_point= r".\u0323"
    # uncertain_point = re.findall(pattern_point, text)
    # pattern_crochet = r"[\(\[]([α-ωΑ-Ω]+)[\)\]]" # modern greek
    # Ancient greek
    # pattern_crochet = r"\[(.*?)\]|\((.*?)\)"
    pattern = r"\[(.*?)\]|\((.*?)\)|(.\u0323)"

    # uncertain_crochet = re.findall(r"\[(\u0370-\u03FF\u1F00-\u1FFF]+)\]|\(([\u0370-\u03FF\u1F00-\u1FFF]+)\)", text)
    # uncertain_crochet = re.findall(pattern_crochet, text)
    # character_counts = [len(match[0] or match[1]) for match in uncertain_crochet]
    # uncertain = uncertain_crochet + uncertain_point
    uncertain = list(sum(re.findall(pattern, text), ()))
    # print(re.findall(pattern, text))

    nb = 0
    # print(uncertain)
    for word in uncertain: 
        # print(word)
        nb += len(word)
    proportion = (nb/(len(text)))*100
    return proportion

def remove_crochet(text): 
    """Retire les crochets et les parenthèses"""
    text_clean = text.replace("(", "").replace(")", "").replace("[", "").replace("]","")
    return text_clean

def clean_people(list_people):
    """Nettoie une liste de personnes au format string et renvoie une bonne liste de personnes"""

    # enlever les chiffres devant les noms 
    # retirer les 'Subscribe to export the table'| ['\r\n        \t\t\t\t\tWe currently do not have any people attestations for this text.']
    list_people = list_people.strip("[").strip("]").replace("'","").split(",")
    pattern_chiffre = r'[0-9]'
    clean_people = []
    # print(list_people)
    # greek_letters = r"[\u0370-\u03FF\u1F00-\u1FFF]+"

    for nom in list_people: 
        # print(nom)
        nom = re.sub(pattern_chiffre,'', nom.lower())
        nom = normalize2(nom)
        if nom != ' subscribe to export the table' and nom != '\\r\\n        \\t\\t\\t\\t\\twe currently do not have any people attestations for this text.':
            clean_people.append(nom.strip())

        # lettre_greek = re.findall(greek_letters, nom)
        # if lettre_greek:
        #     clean_people.append(nom)
    # print(clean_people)

    return clean_people

def clean_places(dico_places): 
    """Nettoie un object string au format dico en une liste de clefs de ce dico normalisé et sans les caractères pas top"""
    dico_places = ast.literal_eval(dico_places)
    list_places = [normalize2(remove_crochet(x)) for x in dico_places.keys()]
    # print(list_places)
    return list_places

def ner(clean_df):
    """Applique le NER UGARIT sur le texte et crée trois colonnes"""
    ner = pipeline('ner', model="UGARIT/grc-ner-bert", aggregation_strategy = 'first')
    
    list_people_total = []
    list_loc_total = []
    list_other_total = []

    for i, row in clean_df.iterrows(): 
        list_people = []
        list_loc = []
        list_other = []
        ner_row = ner(row['Text Clean'])
        for dico_entity in ner_row: 
            if dico_entity['entity_group'] == 'PER':
                list_people.append(dico_entity['word'])
                # print(list_people)
            elif dico_entity['entity_group'] == 'LOC':
                list_loc.append(dico_entity['word'])
            elif dico_entity['entity_group'] == 'MISC':
                list_other.append(dico_entity['word'])
        list_people_total.append(list_people)
        list_loc_total.append(list_loc)
        list_other_total.append(list_other)
    # print(list_people)
    clean_df['People Ugarit'] = list_people_total
    clean_df['Places Ugarit'] = list_loc_total
    clean_df['Other Ugarit'] = list_other_total

def f_mesure(list_actual, list_predicted):
    """Calcule de la f-mesure à partir d'une liste de valeur actueles et une liste de valeur prédites"""
    vp_people =0
    # vn_people =0
    fp_people =0
    fn_people =0
    
    for people in list_actual: 
        if people in list_predicted: 
            vp_people += 1
            list_predicted.remove(people)
            list_actual.remove(people)
    
    fp_people = len(list_predicted)
    fn_people = len(list_actual)

    f_mesure = (2*vp_people) / ((2*vp_people) + fp_people + fn_people)
    return f_mesure

def tolerant(clean_df):
    """Mesure de la f-mesure tolérante""" 
    list_tout = []
    list_tout_ner = [] 
    
    list_list_people = clean_df['People List'].tolist()
    for list in list_list_people: 
        list_tout.extend(list)

    list_ner_people = clean_df['People Ugarit'].tolist()
    for list in list_ner_people: 
        list_tout_ner.extend(list)


    list_places = clean_df['Clean Places List'].tolist()
    for list in list_places: 
        list_tout.extend(list)
    
    list_ner_places = clean_df['Places Ugarit'].tolist()
    for list in list_ner_places: 
        list_tout_ner.extend(list)
    
    
    list_ner_misc = clean_df['Other Ugarit'].tolist()
    for list in list_ner_misc: 
        list_tout_ner.extend(list)
    
    f_mesure_tolerant = f_mesure(list_tout, list_tout_ner)
    
    return f_mesure_tolerant

def severe(clean_df): 
    """Calcul de la f-mesure sévère"""
    list_list_people = clean_df['People List'].tolist()
    true_people =[]
    for list in list_list_people: 
        true_people.extend(list)
    
    list_ner_people = clean_df['People Ugarit'].tolist()
    ner_people = []
    for list in list_ner_people: 
        ner_people.extend(list)


    list_places = clean_df['Clean Places List'].tolist()
    true_places =[]
    for list in list_places: 
        true_places.extend(list)
    
    list_ner_places = clean_df['Places Ugarit'].tolist()
    ner_places = []
    for list in list_ner_places: 
        ner_places.extend(list)
    
    
    list_ner_misc = clean_df['Other Ugarit'].tolist()
    ner_misc = []
    for list in list_ner_misc: 
        ner_misc.extend(list)

    # print(f"people : {true_people[0]} \n {ner_people[0]}")
    
    f_mesure_people = f_mesure(true_people, ner_people)
    f_mesure_places = f_mesure(true_places, ner_places)
    
    return f_mesure_people, f_mesure_places

def normalize(word):
    # enlever les diacritics
    result = u"".join([unicodedata.normalize("NFD", x)[0] for x in word])
    # enlever les <>, (), {} ?[]?
    result = result.replace("<",'').replace(">",'').replace("{",'').replace("}",'').replace("(",'').replace(")",'').replace("*","")
    return result

def normalize2(text):
    # enlever les diacritics / mettre en minuscule 
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c) and c not in "{}"])
    return text.replace('*','').lower()
    # result = u"".join([unicodedata.normalize("NFD", x)[0] for x in word])
    # # enlever les <>, (), {} ?[]?
    # # result = result.replace("<",'').replace(">",'').replace("{",'').replace("}",'').replace("(",'').replace(")",'')
    # return result

def diff(list_diff): 
    list_old = []
    list_new = []
    # ligne par ligne
    list_list_dico_diff = [] # list des list de dico de changements
    for diffs in list_diff:
        dico_diff = {}
        # list_dico =[] # list des dicos pour ce papyrus
        # transforme la string en liste 
        list_diffs = diffs.strip("[").strip("]").replace("'","").split(", ")
    # differences par differences
        # print(f"diffs : {diffs}")
        for diff in list_diffs:
            
            if diff:  
                diff_sans_diacr = normalize(diff)
                # print(f"diff : {diff}")
                match = re.split(": read ", diff_sans_diacr)
                # print(match)
                if match:
                    old, new = diff_lib(match[0], match[1])
                    
                    dico_diff[match[0].strip()] = match[1].strip()
                    # list_dico.append(dico_diff)
                    # list_old.append(old)
                    # list_new.append(new) 
                    if len(new) < 3: 
                        list_old.append(old)
                        list_new.append(new)
        list_list_dico_diff.append(dico_diff)
        # print(list_list_dico_diff)     
    sound_change_df = pd.DataFrame({'old':list_old, 'new':list_new})
    # print(sound_change_df)
    # print(list_old, list_new)
    return sound_change_df, list_list_dico_diff

def diff_lib(old, new): 
    old_sound = ""
    new_sound = ""
    diff = dl.ndiff(old, new)
    for change in diff:
        if change[0] == "-":
            old_sound+=change[2:]
        elif change[0] == "+":
            new_sound+=change[2:]
        
    return old_sound, new_sound

def plus_trente(sound_change_df):
    count_old = sound_change_df['old'].value_counts()
    # trente_plus = count_old[count_old > 30].index # Celles qui ont bien plus que 30 
    trente_plus = count_old.head(8).index # Les memes 8 que le prof
    # print(count_old)
    # dico avec old : [ list des valeurs possibles de new]
    dictionnaire_valeur_old = {old_sound: sound_change_df[sound_change_df["old"] == old_sound]["new"].tolist() for old_sound in trente_plus}
    # dictionnaire_valeur_old = {key:value for key, value in dictionnaire_valeur_old1.items() if len(key) < 3 }
    # print(dictionnaire_valeur_old)

    # Create the size of the figure 
    # le nombre de figure qu'on veut 
    nb_fig = len(dictionnaire_valeur_old)
    # on prend 2 colonnes 
    n_columns = 2 
    # on calcul le nombre de lignes qu'on veut
    n_rows = nb_fig // n_columns 

    if nb_fig % n_columns != 0:
        n_rows += 1 
    # print(n_rows, n_columns)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(15, 15))
    # Adapter figsize
    
    # print(axes)
    axes = axes.flatten()  # Flatten to easily iterate over all axes
    # print(axes)

    # dico = {'ς': ['ι', 'υ', 'δι', '', 'δι', 'ν', 'υ', 'ιυ', 'δι', 'ι', 'δι', '', 'σ', '', '', '', '', '', '', 'υ', '', '', '', '', '', '', 'υ', 'δι', '', 'ν', 'υ', '', '', '', '', 'ι', '', 'υ'], 'ι': ['υ', 'υ', 'η', 'ευ', '', '', 'υ', 'υ', '', 'ε', '', '', 'εη', 'η', '', 'υ', 'ος', 'ος', 'λ', '', 'υ', 'υ', 'μη', 'υ', '', 'υ', 'υ', 'υ', '', 'η', 'η', 'η', 'η', 'η', 'η', 'ε', 'εθ']}
    i = 0
    for old in dictionnaire_valeur_old.keys(): 
        count_new = Counter(dictionnaire_valeur_old[old])
        # print(count_new)
        ax = axes[i]
        # plt.pie(count_new.values(), labels=count_new.keys(), autopct='%1.1f%%')
        # plt.show()
        ax.pie(count_new.values(), labels = count_new.keys(), autopct='%1.1f%%')  
        ax.set_title(f"Old Sound: {old}")
        i+=1

    # # Hide any remaining empty subplots (if any)
    # for j in range(i + 1, n_rows * n_cols):
    #     fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def main(): 
    # PARTIE 4
    print('Partie 4 - Chargement et nettoyage du dataset')

    df = lecture_csv("papyrus_corpus.csv")
    
    print(df.head(4)) # Imprimer les premieres lignes pour regarder 
    taille_corpus_origine = len(df)
    print(f"En regardant les 4 premières lignes de notre dataframe on voit que pour ces papyrus les textes n'ont pas été scrapé.\nLe nombre de fichiers avant traitement: {taille_corpus_origine}.")
    
    clean_df = remove_empty(df)
    
    # print(f"Après avoir retiré les lignes ou la case full text est vide on a {len(clean_df)} papyrus. \nOn a retiré {taille_corpus_origine-(len(clean_df))} lignes au tableau, il s'agit des textes qui n'ont pas été capturés par le scrapping.")

    clean_df.sort_values('ID', ascending=True, inplace=True)
    print(f"On tri ensuite les valeurs des ID par ordre croissant.")

    # PARTIE 5
    print('\nPartie 5 - Etude du corpus, genre lieu et date')

    # Graphes des genres 
    graph_genre(clean_df)
    
    # Papyrus réutilisés 
    reutilises(clean_df)

    # Graphe des provenances
    lieu(clean_df)
    
    # Graphe des dates 
    dates(clean_df)


    # PARTIE 6
    print('\nPartie 6 - Nettoyage du texte grec')
    
    clean_df['Text Clean'] = clean_df['Full Text'].map(nettoyage)
    
    clean_df['Uncertain Portion'] = clean_df['Full Text'].map(uncertain)
    
    nb_bcp_uncertain = len(clean_df[clean_df['Uncertain Portion'] >= 30])
    print(f"Il y a {nb_bcp_uncertain} papyrus qui ont plus d'un tiers de texte incertain.")
    
    clean_df['Text Clean'] = clean_df['Text Clean'].map(remove_crochet)

    # PARTIE 7
    print('\nPartie 7 - Identifier les noms de personnes et de lieux')
    
    print("Il y a beaucoup de caractères en trop mais aussi des phrases dans les listes de personnes.\nOn applique une fonction qui nettoie ces listes.")
    clean_df['People List'] = clean_df['People List'].map(clean_people)
    clean_df['Text Clean'] = clean_df['Text Clean'].map(normalize2)

    clean_df["Clean Places List"] = clean_df["Places List"].map(clean_places)

    ner(clean_df)
    print("Les résultats ont au premier abord l'air de ne pas être trop mauvais.")
    f_score_tolerant = tolerant(clean_df)
    f_mesure_people, f_mesure_places = severe(clean_df)
    print(f"Le f-score tolerant du systeme NER Ugarit est de {f_score_tolerant}")
    print(f"Le f-score sévère du systeme NER Ugarit est de {f_mesure_people} pour les personnes et {f_mesure_places} pour les lieux" )
    
    # PARTIE 8
    print("\nPartie 8 - Etude des fautes de graphie")

    # clean_df['Clean Irregularities'] = clean_df['Text Irregularities'].map(normalize)
    list_diff = clean_df['Text Irregularities'].tolist()
    
    # Nouveau dataframe avec les changements de graphèmes 
    sound_change_df, list_dico_diffs = diff(list_diff)
    
    # Creation dataframe 
    count_sound_df = sound_change_df[['old', 'new']].value_counts().reset_index(name='count')
    clean_df["Clean irregularities"] = list_dico_diffs
    
    # 10 changements les plus courants 
    # print(f"Les 10 changements les plus courants sont : \n{count_sound_df.head(10)}")
    top_10 = zip(count_sound_df['old'].head(10).to_list(), count_sound_df['new'].head(10).to_list())
    print(f" Les 10 changements les plus courants sont : {list(top_10)}")
    
    
    # Création graphique des graphèmes anciens modifiés plus de 30 fois 
    # Trouve les changements + de 30 
    plus_de_trente = count_sound_df[count_sound_df['count']>30]
    plus_de_trente_tuples = zip(plus_de_trente['old'].to_list(), plus_de_trente['new'].to_list())
    plus_trente(sound_change_df)    
    # print(f'Les graphèmes du grecs classiques modifiés plus de 30 fois sont :\n {plus_de_trente}')
    print(f'Les graphèmes du grecs classiques modifiés plus de 30 fois sont :')
    for tuple in plus_de_trente_tuples: 
        print(f"\"{tuple[0]}\" -> \"{tuple[1]}\"", end=", ")


    print("\n\nTransition")
    print(f"Après tous ces traitements nous avons les colonnes suivantes: \n{clean_df.columns.values}")
    # columns_inutiles = ['Recto/Verso','Reuse note','Reuse type', 'Ro','Note','Culture & genre','Material','Authors / works', 'Book form']
    columns_inutiles = ['Recto/Verso','Reuse note','Reuse type', 'Ro','Note','Culture & genre','Material','Authors / works', 'Book form', 'People Ugarit','Places Ugarit','Other Ugarit']

    print(f"Je pense que les colonnes {columns_inutiles} ne sont pas utiles pour notre analyse.")
    for column in columns_inutiles: 
        clean_df = clean_df.drop(column, axis= 1)
    clean_df.to_csv("clean_papyrus-corpus.csv", index=False)

if __name__ == '__main__':
    main()