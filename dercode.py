# 2025-06-29
# Ali Ufuk Bozkurt, Janusz Guzman
# Endprojekt Programmieren (Unfallatlas-Analyse)


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from overpass import API
from time import sleep
from sys import exc_info

# FUNKTIONEN ZUM ERSTMALIGEN AUSFÜHREN (hier an-/ausschalten mit 1 bzw. 0)
kaUnfallorteFehlt = 1

kaUnfallorte2023Fehlt = 1    # Achtung: stabile Internetverbindung notwendig, zeitaufwendig

optimiereKMeans = 1
erzwingeAnderesK = 1    # es wird k = 308 ("nicht optimal") genommen, damit die exportierte Karte besser aussieht

korrelationGesucht = 1


# Generieren von KA_Unfallorte.csv
if kaUnfallorteFehlt == 1:
    typ = {1: str, 23: str}    # Bugfix
    
    # Einladen der Unfalldaten der letzten Jahre
    df1 = pd.read_csv('Unfallorte2023_LinRef.csv', sep=';', encoding='utf-8', dtype=typ)
    df2 = pd.read_csv('Unfallorte2022_LinRef.csv', sep=';', encoding='utf-8', dtype=typ)
    df3 = pd.read_csv('Unfallorte_2021_LinRef.txt', sep=';', encoding='utf-8', dtype=typ)
    df4 = pd.read_csv('Unfallorte2020_LinRef.csv', sep=';', encoding='utf-8', dtype=typ)
    df5 = pd.read_csv('Unfallorte2019_LinRef.txt', sep=';', encoding='utf-8', dtype=typ)
    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

    # Filtern nach Unfällen in Karlsruhe (siehe Gemeindeschlüssel)
    dfGefiltert = df[(df['ULAND'] == 8) & (df['UREGBEZ'] == 2) & (df['UKREIS'] == 12) & (df['UGEMEINDE'] == 0)].copy()
    print(dfGefiltert.shape)    # Ermitteln der Anzahl Unfälle in Karlsruhe in den behandelten Jahren

    # Ersetzen der deutschen Komma-Schreibweise durch die Python-Schreibweise
    dfGefiltert['XGCSWGS84'] = dfGefiltert['XGCSWGS84'].str.replace(',', '.').astype(float)
    dfGefiltert['YGCSWGS84'] = dfGefiltert['YGCSWGS84'].str.replace(',', '.').astype(float)

    # Exportieren der Karlsruher Unfalldaten
    dfGefiltert.to_csv(r"KA_Unfallorte.csv", index=False)

    print("KA_Unfallorte.csv generiert")

# Einladen der Karlsruher Unfalldaten
dfKA = pd.read_csv('KA_Unfallorte.csv', encoding='utf-8')

# Ermitteln des Tempolimits an Unfallorten (Generiere KA_Unfallorte_2023.csv)
# (nur neueste Daten, also von 2023, genommen)
# siehe https://leshaved.wordpress.com/2015/12/01/getting-nearest-roads-from-osm-in-python/
if kaUnfallorte2023Fehlt == 1:
    api = API()

    dfGefiltert2 = dfKA[(dfKA['UJAHR'] == 2023)].copy()
    dfTest = dfGefiltert2.head(20).copy()    # Testen mit 20 Unfallorten

    # for index, row in dfTest.iterrows():    # test
    for index, row in dfGefiltert2.iterrows():
        try:
            breitengrad = row['YGCSWGS84']
            laengengrad = row['XGCSWGS84']
            # Finden linienförmiger Straßenobjekte im Umkreis von 10 Metern des Unfalls
            abfrage = api.Get(f'way["highway"](around:10,{breitengrad},{laengengrad});')
            # print("Raw response:", response)    # debug
            # print(f"Für {breitengrad}, {laengengrad} Straßen mit folgenden OSM-Objekt-IDs und Tempolimits gefunden:")    # debug
            if 'features' in abfrage:
                waynummer = 0
                for way in abfrage['features']:
                    waynummer = waynummer + 1
                    # print(way.get('id'), way.get('properties').get('maxspeed'), waynummer)    # debug
                    if waynummer == 1:    # Betrachte nur das 1. Objekt (nach welchem Kriterium werden die eigentlich sortiert?
                                          # Nach Distanz wäre schwierig, ist aber eh nicht wirklich nötig bei Umkreis von 10 Metern)
                                          # siehe https://community.openstreetmap.org/t/distance-point-to-nearest-road/83998
                        # dfTest.at[row.name, 'tempolimit'] = way.get('properties').get('maxspeed')
                        dfGefiltert2.at[row.name, 'tempolimit'] = way.get('properties').get('maxspeed')
                        print(way.get('id'), way.get('properties').get('maxspeed'), index)
            else:
                print("Keine Straße da gefunden")
            # sleep(0.1)    # falls die API mit zu vielen Abfragen auf einmal ein Problem haben sollte
        except Exception as e:
            print(f"Fehler: {e}")
    
    # print(dfTest[['UKATEGORIE', 'YGCSWGS84', 'XGCSWGS84', 'tempolimit']])    # test
    print(dfGefiltert2[['UKATEGORIE', 'YGCSWGS84', 'XGCSWGS84', 'tempolimit']])
    dfGefiltert2.to_csv(r"KA_Unfallorte_2023.csv", index=False)

    print("KA_Unfallorte_2023.csv generiert")

# Ermitteln der Korrelation zwischen Tempolimit und Unfallschwere (nur Daten von 2023)
if korrelationGesucht == 1:
    df2023 = pd.read_csv("KA_Unfallorte_2023.csv")
    # Anzeigen aller vorgekommenen Werte (20 und 40 km/h werden vernachlässigt werden)
    # print(df2023['tempolimit'].value_counts(dropna=False))
    # Angleichen ähnlicher Schrittgeschwindigkeiten, siehe https://wiki.openstreetmap.org/wiki/Key:maxspeed#Values
    df2023.loc[df2023['tempolimit'].isin(['7', '10', 'walk']), 'tempolimit'] = 7

    # Zählen der Unfälle nach Kategorie und Tempolimit (tabellarische Form für Säulendiagramm)
    unfallvorkommen = df2023.groupby(['tempolimit', 'UKATEGORIE']).size().unstack(fill_value=0)
    print(unfallvorkommen)

    # in Prozenten, damit Säulen für jedes Tempolimit gleich groß (vergleichbar) sind
    verhaeltnisse = unfallvorkommen.div(unfallvorkommen.sum(axis=1), axis=0) * 100
    verhaeltnisse.index = pd.to_numeric(verhaeltnisse.index, errors='coerce')    # wegen "none" (OSM-legitimer String für unbegrenztes Tempolimit != None bei Python)
    verhaeltnisse = verhaeltnisse.sort_index()

    # Erstellen des Plots
    verhaeltnisse.plot(kind='bar', stacked=True)
    plt.ylabel('Prozent aller Unfälle')
    plt.xlabel('Tempolimit')
    plt.legend(title='Unfallkategorie')
    plt.savefig('Tempolimit_Unfallschwere.png')

    print("Plot hierzu erstellt")

    # Hypothesen für geringe Korrelation: bei geringen Geschwindigkeiten sind zwar Leute im Auto geschützt, die geringen Geschwindigkeiten herrschen
    # aber dort, wo Fußgänger und Radfahrer unterwegs sind als etwa auf Autobahnen, und diese verunglücken bereits bei geringen Geschwindigkeiten

# Definieren der X- und Y-Werte (und weights) für Clustering-Algorithmus
koordinaten = dfKA[['XGCSWGS84', 'YGCSWGS84']]
weights = 4 - dfKA['UKATEGORIE']    # tödlicher Unfall = 3, leichter Unfall = 1 

# Ermitteln der optimalen Anzahl Cluster
if optimiereKMeans == 1:
    # Vorbereiten des Silhouette-Score-Tests
    silhouetteScores = []
    optimalesK = 0
    bislangBesterScore = 0

    # Ermitteln des Silhouette-Scores für jedes k
    # (Auswahl schon eingeschränkt, nachdem wir gesehen haben, dass 308 Cluster angeblich viel zu wenige sind)
    for k in range(850, 1050):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(koordinaten, sample_weight=weights)
        score = silhouette_score(koordinaten, labels)
        silhouetteScores.append(score)
        print(f"Anzahl Cluster: {k}, Silhouette-Score: {score}")
        if score > bislangBesterScore:
            optimalesK = k
            bislangBesterScore = score

    print(f"Optimale Anzahl Cluster: {optimalesK}")
    
    # Erstellen eines Plots für Silhouette-Scores
    plt.plot(range(850, 1050), silhouetteScores)
    plt.xlabel('Anzahl Cluster')
    plt.ylabel('Silhouette-Score')
    plt.savefig('Silhouette-Score_Plot.png')

    print("Plot hierzu erstellt")

    kmeans = KMeans(n_clusters=optimalesK)

if optimiereKMeans == 0 or erzwingeAnderesK == 1:
    kmeans = KMeans(n_clusters=308)
    print("Arbeite mit 308 Clustern...")

dfKA['cluster'] = kmeans.fit_predict(koordinaten, sample_weight=weights)

# Erzeugen der interaktiven Karte
print("Erstelle Karte mit Unfällen und Cluster-Mittelpunkten...")
karte = folium.Map(location=[49.005, 8.405], zoom_start=13)    # Mittelpunkt Karlsruhe

# Erstellen von FeatureGroups für verschiedene Layer
unfaelle2023 = folium.FeatureGroup(name="Unfallorte 2023 (zur Tempolimit-Analyse betrachtet)")
unfaelleFrueher = folium.FeatureGroup(name="Unfallorte vor 2023 (zusätzlich zu 2023-Unfällen für Clustering gefährlicher Kreuzungen betrachtet)")
clusterMittelpunkte = folium.FeatureGroup(name="Gefährliche Kreuzungen (Cluster-Mittelpunkte)")

# Hinzufügen von Unfallpunkten zu deren Layern
for _, row in dfKA.iterrows():
    ukategorie = ""
    ulichtverh = ""
    istrad = "nein"
    istpkw = "nein"
    istfuss = "nein"
    istkrad = "nein"
    istgkfz = "nein"
    istsonstige = "nein"

    if row['UKATEGORIE'] == 1:
        ukategorie = "Unfall mit Toten"
    if row['UKATEGORIE'] == 2:
        ukategorie = "Unfall mit Schwerverletzten"
    if row['UKATEGORIE'] == 3:
        ukategorie = "Unfall mit Leichtverletzten"
    
    if row['ULICHTVERH'] == 0:
        ulichtverh = "bei Tag"
    if row['ULICHTVERH'] == 1:
        ulichtverh = "bei Nacht"

    farbe = 'yellow'    # Unfälle mit sonstigen Verkehrsmitteln
    if row['IstPKW'] == 1:
        farbe = 'green'
        istpkw = "ja"
    if row['IstRad'] == 1:
        farbe = 'red'
        istrad = "ja"
    if row['IstFuss'] == 1:
        farbe = 'blue'    # ob ein Radfahrer erfasst wird, spielt farblich keine Rolle, sobald ein Fußgänger betroffen war
        istfuss = "ja"

    if row['IstKrad'] == 1:
        istkrad = "ja"
    if row['IstGkfz'] == 1:
        istgkfz = "ja"
    if row['IstSonstige'] == 1:
        istsonstige = "ja"

    unfallpunkt = folium.CircleMarker(
        location = [row['YGCSWGS84'], row['XGCSWGS84']],
        radius = 2,
        color = farbe,
        popup = f"{ukategorie} {ulichtverh},\nFahrrad beteiligt: {istrad},\nPKW beteiligt: {istpkw},\nFußgänger beteiligt: {istfuss},\nMotorrad beteiligt: {istkrad},\nLKW beteiligt: {istgkfz},\nSonstige beteiligt: {istsonstige}"
    )

    if row['UJAHR'] == 2023:
        unfallpunkt.add_to(unfaelle2023)
    else:
        unfallpunkt.add_to(unfaelleFrueher)

# Hinzufügen von Cluster-Mittelpunkten (gefährliche Kreuzungen) zum entsprechenden Layer
for mittelpunkt in kmeans.cluster_centers_:
    folium.CircleMarker(
        location = [mittelpunkt[1], mittelpunkt[0]],    # Erst Breitengrad
        radius = 5,
        color = 'black'
    ).add_to(clusterMittelpunkte)

# Hinzufügen aller Layer zur Karte
unfaelle2023.add_to(karte)
unfaelleFrueher.add_to(karte)
clusterMittelpunkte.add_to(karte)
folium.LayerControl(collapsed=False).add_to(karte)    # Button

# Exportieren der Karte
karte.save('KA_Unfallkarte.html')

print("Fertig -- Karte nun im Ordner des Codes als HTML gespeichert")
