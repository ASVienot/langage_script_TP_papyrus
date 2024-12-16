import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import ast 

def lecture_csv(path):
    """lecture du csv en dataframe"""
    df = pd.read_csv(path)
    return df

def recup_site(url): 
    """recuperation de la soup d'une url"""
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup

def recup_georef(data):
    """scrapping de la georef"""
    list_dic_lieux = []
    list_lieux = data["Places List"].tolist()
    for lieux in list_lieux:
        dico_lieux = ast.literal_eval(lieux) 
        list_dic_lieux.append(dico_lieux)
    data['Georef']= list_dic_lieux
    
    return list_dic_lieux


    # provenance = Counter(list_provenance_clean)
    
    # ville_max = max(provenance, key= lambda x: provenance[x])

    # fig = plt.figure(figsize = (5, 10))

    # plt.bar(provenance.keys(), provenance.values(), width = 0.5)
    
    # plt.xticks(rotation=90)
    # plt.xlabel("Villes")
    # plt.ylabel("Nombres de papyrus")
    # plt.title("Distribution des villes d'ou viennent les papyri")
    # plt.show()

def get_geo(list_dic_georef):
    """scrapping des coordonnées géographiques depuis une liste de dictionnaire avec les georef
    retourne une liste avec pour chaque papyrus une liste de coordonnees """
    list_list_coord = []
    list_new_df = []
    for i, dic_lieux in enumerate(list_dic_georef): 
        urls_loc = []
        base_url_ref = "https://www.trismegistos.org/georef/"
        for geo in dic_lieux.values(): 
            loc = geo.strip('getgeo(').strip(")")
            urls_loc.append(f"{base_url_ref}{loc}")
        print(i)

        list_coord =[]
        for ref in urls_loc: 
            soup2 = recup_site(ref)
            div = soup2.find('div', id="right-infobox")
            for ds in div.find_all('p'):
                if "Lat,Long" in ds.text:
                    coord = (ds.text).replace(" (Lat,Long)", "").strip().split(",")
                    lat = coord[0]
                    long = coord[1]
                    list_new_df.append((lat,long))
                    list_coord.append((lat,long))

        list_list_coord.append(list_coord)
    coord_df = pd.DataFrame(list_new_df, columns=["Lat", "Long"])
    coord_df.to_csv("tableau_coord_geo.csv")

    return list_list_coord

def main():
    df = lecture_csv("clean_papyrus-corpus.csv")
    list_dic_georef = recup_georef(df)
    df["Coord Geo"] = get_geo(list_dic_georef)
    df.to_csv("clean_papyrus-corpus_augmente.csv")
    
if __name__ == '__main__':
    main()