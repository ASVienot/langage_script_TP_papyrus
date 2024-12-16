import requests
from bs4 import BeautifulSoup
import pandas as pd

def lire_urls(liste):
    """Cette fonction prend une liste d'id et les transforment en liste d'url"""
    urls = []
    for id in liste: 
            num = id.removeprefix('TM')
            urls.append(f'https://www.trismegistos.org/text/{num}')
    return urls

def recup_site(url): 
    """cette fonction prend une url et en retourn la soup """
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup

def recup_list_url(): 
    """Cette fonction prend la csv et retourne une list des ids"""
    df = pd.read_csv("papyrus_metadata.csv")
    id = df['ID'].tolist()
    return id 

def scrap_papyrus(url):
    """Cette fonction scrap une url pour créer un dictionnaire avce toutes les informations sur les papyri""" 
    dico_papy = {}
    soup = recup_site(url)
    div_details = soup.find('div', id='text-details')

    dico_papy = {"ID" : "", "Date" : "", "Provenance" : "", "Language/script" : "", "Material" : "",
                    "Content" : "", "Archive": [],"Publications" : [], "Collections" : [], "Text" : "",
                    "People" : [], "Places" : [], "Text irregularities" : [], "Geo" : ""}

    id = url.split('/')[-1]
    dico_papy['ID'] = id

    for child in div_details: 
        if "Date" in child.text: 
            dico_papy['Date'] = child.text
        elif "Provenance" in child.text: 
            dico_papy['Provenance'] = child.text
        elif "Language" in child.text: 
            dico_papy['Language/script'] = child.text
        elif "Material" in child.text: 
            dico_papy['Material'] = child.text
        elif "Content" in child.text: 
            dico_papy['Content'] = child.text   

    pub=""
    publication = soup.find('div', id='text-publs')
    for child in publication: 
        pub += child.text
    dico_papy['Publications'] = pub

    archives = []
    archive = soup.find('div', id='text-coll')
    for child in archive.find_all('p'):
        if child: 
            archives.append(child.text)
    dico_papy['Collections'] = archives

    collec = []
    collection = soup.find('div', id='text-arch')
    for child in collection.find_all('p'):
        if child: 
            collec.append(child.text)
    dico_papy['Archive'] = collec


    words = ""
    mots = soup.find('div', id='words')
    for child in mots:
        words+= child.text
    dico_papy['Text'] = words

    liste_people =[]
    people = soup.find('div', id='people')
    for child in people.find_all('li', class_ = 'item-large'): 
        liste_people.append(child.text)
    dico_papy['People'] = liste_people

    liste_places =[]
    list_geo = []
    places = soup.find('div', id='places')
    for child in places.find_all('li', class_ = 'item-large'): 
        liste_places.append(child.text)
        list_geo.append(child['onclick'])
    dico_papy['Places'] = liste_places

    liste_irregularites =[]
    irregularite = soup.find('div', id='texirr')
    for child in irregularite.find_all('li', class_ = 'item-large'): 
        liste_irregularites.append(child.text)
    dico_papy['Text irregularities'] = liste_irregularites

    # GEO Bonnus

    urls_loc = []
    base_url_ref = "https://www.trismegistos.org/georef/"
    for geo in list_geo: 
        loc = geo.strip('getgeo(').strip(")")
        urls_loc.append(f"{base_url_ref}{loc}")
    # print(urls_loc)

    list_coord =[]
    for ref in urls_loc: 
        soup2 = recup_site(ref)
        div = soup2.find('div', id="right-infobox")
        for ds in div.find_all('p'):
            if "Lat,Long" in ds.text:
                coord = ds.text
                list_coord.append(coord)

    dico_papy['Geo'] = list_coord

    return dico_papy

def main():
    list_id = recup_list_url()
    list_urls = lire_urls(list_id)
    dico_tout_papyrus = {"ID" : [], "Date" : [], "Provenance" : [], "Language/script" : [], "Material" : [],
                    "Content" : [],"Archive": [] ,"Publications" : [], "Collections" : [], "Text" : [],
                    "People" : [], "Places" : [], "Text irregularities" : [], "Geo" : []}
    
    for url in list_urls[:8]:
        dico_papy = scrap_papyrus(url)
        for key in dico_papy.keys(): 
            dico_tout_papyrus[key].append(dico_papy[key])

    data = pd.DataFrame(dico_tout_papyrus)
    data.to_csv("scraper.csv")
    print(dico_tout_papyrus)

if __name__ == '__main__':
    main()
