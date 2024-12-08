# langage_script_TP_papyrus

Dans ce TP pour le cours de langage de script, j'ai écrit 3 scripts python (.py).  

## scraping.py 
Dans ce premier script, le site internet trismegistos qui répertorie les papyrus est scraper afin de récupérer un corpus de papyrus enrichi. 
"https://www.trismegistos.org/text/"

## corpus_analysis.py 
Ce deuxième script commence par afficher des **graphes** de visualisation des données sur les papyrus. Il enrichi ensuite les données. ++

## streamlit_papyrus.py
Ce troisième script crée et affiche une interface utilisateur pour "La chasse au papyrus".
Dans cet interface, on trouve une option de visualisation des données analysées dans le deuxième script et une option d'affichage de papyrus.
Pour afficher les papyrus, trois filtres sont disponnibles: 
- :scroll: Un filtre sur la provenance des papyrus,  
- :scroll: Un filtre sur un interval de dates,  
- :scroll: Un filtre sur le genre du document. 
Après avoir choisi un papyrus, les données suivantes sur les papyrus sont affichées: 
- Date,    
- Provenance,     
- Listes des personnes mentionnées et les ids papyrus dans lesquels ces personnes apparaissent,      
- Listes des lieux mentionnées et les ids papyrus dans lesquels ces lieux apparaissent,    
- Texte annoté avec les irrégularités graphiques, etc.