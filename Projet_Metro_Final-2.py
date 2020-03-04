#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:11:38 2019

@author: Nadia
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
import seaborn as sns
import numpy as np
from collections import defaultdict
from itertools import islice
import warnings
warnings.filterwarnings("ignore", category = UserWarning)


# using relative path

fichier_conf = '/users/mmath/ghernaou/PROJET_FINAL/fichier_conf.txt'
#fichier_conf = '/Users/Nadia/Documents/Maths/PROJET_METRO_PYTHON/fichier_conf.txt'



class Reseau: # lecture des 2 fichiers

    def __init__(self):
        pass

    def reading_metros(self,filename1):
        self.metros = pd.read_csv(path1, sep = ";" )
        return self.metros

    def reading_ordres(self,filename2):
        self.ordre_station = pd.read_csv(path2, sep = ";" )
        return self.ordre_station

    def station_names(self,filename2):
        self.ordre = Reseau.reading_ordres(self,filename2)
        self.names = list(self.ordre.Station)
        return self.names

    def station_names2(self,filename1):
        self.metros = Reseau.reading_metros(self,filename1)
        self.names2 = list(self.metros.station)
        return self.names2


class Graphique_Reseau:
    def __init__(self,filename1,filename2):
        self.ordre_station = Reseau.reading_ordres(self,filename2)
        self.metros = Reseau.reading_metros(self,filename1)
        self.names = Reseau.station_names(self,filename2)
        self.names2 = Reseau.station_names2(self,filename1)
        self.lignes = list(self.ordre_station['res_com'].unique())
        self.y1 = list(self.ordre_station['latitude station'])
        self.x1 = list(self.ordre_station['longitude station'])
        self.station = list(self.ordre_station.Station)
        self.station_suivant = list(self.ordre_station['Station suivante'])
        self.pos_all = {}


    def lignes_stations(self):      #donne une liste de liste des correspondances
        liste_cor = []
        for k in [1,2,3,4,5]:
            X = list(self.metros[f'C_{k}'])
            liste_cor.append(X)
        return(liste_cor)

    def lignes_stations2(self):              # rend un dico: en clé une station et en valeurs une liste des lignes qui traversent cette station
        lignes_stations = {}
        liste = Graphique_Reseau.lignes_stations(self)
        for i in range (0,len(self.names2)):
            nom = self.names2[i]
            l = (str(liste[0][i]),str(liste[1][i]),str(liste[2][i]),str(liste[3][i]),str(liste[4][i]))
            cleanedList = [x for x in l if str(x) != 'nan']
            lignes_stations[nom] = cleanedList
        return (lignes_stations)


    # Positionne tous les noeuds (stations) en utilisant les coordonnées
    def station_location(self):
        self.X = list(self.ordre_station['longitude station'])
        self.Y = list(self.ordre_station['latitude station'])
        for i in range(0, len(self.names)):
            self.pos_all[self.names[i]] = (self.X[i], self.Y[i])
        return self.pos_all


    # Permet de dessiner toutes les stations

    def drawing_correspondance(self):

        self.g = nx.Graph()
        cor = list(self.metros[pd.isna(self.metros['C_2']) == False].station)
        lon_cor, lat_cor = Shortest_Route.coordinates_liste_transfers(self)
        pos_cor = {}

        for k in range(len(cor)):
            pos_cor[cor[k]] = (lat_cor[k],lon_cor[k])
        self.g.add_node(cor[0])
        self.g.add_nodes_from(cor[1:])

        correspondance1 = nx.draw_networkx_nodes(self.g, pos=pos_cor, node_size=50, node_color='grey', alpha=0.25)
        correspondance2 = nx.draw_networkx_labels(self.g,pos=pos_cor,font_size=4.0,font_color='grey')

        return correspondance1,correspondance2


    def drawing_first_stations(self):    #destinations
            colors = ['#FFCD00','#C9910D','#704B1C','#007852','#6EC4E8','#62259D','#003CA6','#837902','#6EC4E8','#CF009E','#FF7E2E','#6ECA97','#FA9ABA','#6ECA97','#E19BDF','#B6BD00']
            lines=['M1', 'M10', 'M11', 'M12', 'M13', 'M14', 'M2', 'M3', 'M3 bis', 'M4', 'M5', 'M6', 'M7', 'M7 bis', 'M8', 'M9']
            first_=[]
            lat = []
            lon = []
            for i in range(len(lines)):
                if self.ordre_station[self.ordre_station['res_com']==self.lignes[i]].res_com.values.any()==self.lignes[i]:
                    m=list(self.ordre_station[self.ordre_station['res_com']==lines[i]].res_com)
                    y=list(self.ordre_station.loc[self.ordre_station['res_com']==self.lignes[i],'latitude station'])
                    x=list(self.ordre_station.loc[self.ordre_station['res_com']==self.lignes[i],'longitude station'])
                    first_.append(m[0])
                    lon.append(y[0])
                    lat.append(x[0])

                pos_first_={}
                for k in range(len(first_)):
                    pos_first_[first_[k]]=(lat[k],lon[k])
            self.g = nx.Graph()
            self.g.add_node(first_[0])
            self.g.add_nodes_from(first_[1:])

            first=nx.draw_networkx_nodes(self.g, pos=pos_first_, nodelist=first_,node_size=10, node_color='black',alpha=0.10)
            label=nx.draw_networkx_labels(self.g,pos=pos_first_,font_size=5,font_color='blue',bbox=dict(facecolor='grey', alpha=0.5))
                
            return first,label


    def drawing_edges(self):
        colors = ['#FFCD00','#C9910D','#704B1C','#007852','#6EC4E8','#62259D','#003CA6','#837902','#6EC4E8','#CF009E','#FF7E2E','#6ECA97','#FA9ABA','#6ECA97','#E19BDF','#B6BD00']

        for i in range(len(self.lignes)):
            n = len(self.ordre_station[self.ordre_station['res_com']==self.lignes[i]].res_com)

            if self.ordre_station[self.ordre_station['res_com']==self.lignes[i]].res_com.values.any()==self.lignes[i]:
                y = list(self.ordre_station.loc[self.ordre_station['res_com']==self.lignes[i],'latitude station'])
                x = list(self.ordre_station.loc[self.ordre_station['res_com']==self.lignes[i],'longitude station'])
                m = list(self.ordre_station[self.ordre_station['res_com']==self.lignes[i]].Station)
                m_s = list(self.ordre_station.loc[self.ordre_station['res_com']==self.lignes[i],'Station suivante'])
                pos = {}
                for k in range(n):
                    pos[m[k]] = (x[k],y[k])
                    self.g = nx.Graph()
                    self.g.add_node = m[k]

                for j in range(n):
                    self.g.add_edge(m[0],m_s[0])
                    self.g.add_edge(m[j],m_s[j])
                    nodes = nx.draw_networkx_nodes(self.g, pos=pos, node_size=7, node_color=colors[i])
                    edges = nx.draw_networkx_edges(self.g, pos=pos, edge_color=colors[i], width=0.5, alpha=0.25)

        return nodes, edges


    def shortest_route_drawing(self,source_x,source_y,target_x,target_y):

        station_list = Shortest_Route.shortest_route_weighted(self,source_x,source_y,target_x,target_y,temps_source,temps_target)

        y1 = list(self.metros['Latitude'])
        x1 = list(self.metros['Longitude'])
        m = list(self.metros.station)
        # creating positions of shortest route
        lat = []
        lon = []
        for i in range(1,len(station_list)-1):
            for j in range(len(y1)):
                if station_list[i]==m[j]:
                    lat.append(x1[j])
                    lon.append(y1[j])
        lat = [source_x,*lat,target_x]
        lon = [source_y,*lon,target_y]

        pos_short = {}
        for k in range(len(station_list)):
            pos_short[station_list[k]] = (lat[k],lon[k])

        self.g = nx.Graph()
        for i in range(len(station_list)):
            self.g.add_path(station_list)

        n2 = nx.draw_networkx_nodes(self.g, pos=pos_short,node_size=35,node_color='blue')
        n3 = nx.draw_networkx_labels(self.g, pos=pos_short,font_size=5,font_color='k')
        n4 = nx.draw_networkx_edges(self.g, pos=pos_short,width=2.5,edge_color='black',style='dashed',alpha=1)

        return n2,n3,n4


    def walking_route(self,source_x,source_y,target_x,target_y):
        extremes = ["Point de départ","Point d'arrivée"]
        lat = [source_x,target_x]
        lon = [source_y,target_y]
        pos_walk = {}
        for k in range(len(extremes)):
            pos_walk[extremes[k]]=(lat[k],lon[k])
        h = nx.Graph()
        for i in range(len(extremes)):
            h.add_path(extremes)

            nx.draw_networkx_nodes(h,pos=pos_walk,node_size=7,node_shape='s')
            nx.draw_networkx_edges(h,pos=pos_walk,width=1.0,edge_color='g',style='dotted')
            nx.draw_networkx_labels(h,pos=pos_walk,font_size=7,font_color='k')
#            plt.show()


    def combining_graphs(self,source_x,source_y,target_x,target_y,temps_source,temps_target):
        ds = min(Closest_Stations.distance_source_stations(self,source_x,source_y,temps_source))
        dt = min(Closest_Stations.distance_target_stations(self,target_x,target_y,temps_target))
        dist = Closest_Stations.distance_start_end(self,source_x,source_y,target_x,target_y)
        if dist < ds+dt:
            return Graphique_Reseau.drawing_correspondance(self),Graphique_Reseau.drawing_edges(self),Graphique_Reseau.walking_route(self,source_x,source_y,target_x,target_y), Closest_Stations.temps_total_marche(self,source_x,source_y,target_x,target_y)
        else:
            return Graphique_Reseau.drawing_correspondance(self),Graphique_Reseau.drawing_edges(self),Graphique_Reseau.shortest_route_drawing(self,source_x,source_y,target_x,target_y), Closest_Stations.temps_total2(self,source_x,source_y,target_x,target_y,temps_source,temps_target),Graphique_Reseau.drawing_first_stations(self)




class Shortest_Route(Graphique_Reseau):

    def __init__(self,filename1,filename2):
        Graphique_Reseau.__init__(self,filename1,filename2)



    def generating_initial_edge_list(self):
        edgelist = []
        for i in range(len(self.station)):
            edgelist.append((self.station[i],self.station_suivant[i]))
        return edgelist


    def shortest_route_weighted(self,source_x,source_y,target_x,target_y, temps_source,temps_target):    ################################################
        transfers = Shortest_Route.liste_transfers(self)
        edges_sc = Shortest_Route.edgelist_NOtransfers(self)
        edge_c = Shortest_Route.edgelist_transfers(self)
        stations_source = Closest_Stations.stations_source(self,source_x,source_y,temps_source)
        stations_target = Closest_Stations.stations_target(self,target_x,target_y,temps_target)

        walk = ["Point de départ","Point d'arrivée"]
        distances_source = Closest_Stations.distance_source_stations(self,source_x,source_y,temps_source)
        distances_target = Closest_Stations.distance_target_stations(self,target_x,target_y,temps_target)
        weight_source = (min(distances_source)/vitesse_marcheur) / temps_entre_stations
        weight_target = (min(distances_target)/vitesse_marcheur) / temps_entre_stations

        # Sélectionner la station la plus proche en se basant sur la distance
        index_source = np.argmin(distances_source)
        closest_source = stations_source[index_source]

        index_target = np.argmin(distances_target)
        closest_target = stations_target[index_target]

        g = nx.Graph(edges_sc)
        # Arete avec poids lié à la distance entre le début de la marche et la station la proche
        g.add_edge(walk[0],closest_source, weight = weight_source)   
        # Arete avec poids lié à la distance entre la station d'arrivée et le point d'arrivée
        g.add_edge(closest_target,walk[1],weight = weight_target)    

        metro_edgelist=Shortest_Route.generating_initial_edge_list(self)

        # liste de toutes les aretes
        g.add_path(metro_edgelist)

        # Trouver les arêtes avec correspondances pour ajuster le poids
        # On commence par sortir les 3 premiers chemins les plus courts
        for path in Shortest_Route.k_shortest_paths(self,g, "Point de départ", "Point d'arrivée", 5):    # 5 meilleurs shortest path: on en fait une liste , un poids a ete mis sur les aretes de correspondance
            p = path

        transfer_list = []
        avant = []
        suivant = []
        for j in range(len(p)-1):
            for i in range(len(transfers)):
                if p[j]==transfers[i]:
                    transfer_list.append(p[j])
                    avant.append(p[j-1])
                    suivant.append(p[j+1])

        c_edge = []
        for i in range(len(suivant)):
            ligne_station_suivante = list(self.ordre_station.loc[self.ordre_station['Station suivante'] == suivant[i],'res_com'])
            transfer_linked = list(self.ordre_station.loc[self.ordre_station['Station suivante'] == suivant[i],'Station'])
            ligne_station_avant = list(self.ordre_station.loc[self.ordre_station['Station'] == avant[i],'res_com'])
            station_suivant = list(self.ordre_station.loc[self.ordre_station['Station suivante'] == suivant[i],'Station suivante'])

            for k in range(len(ligne_station_suivante)):
                if ligne_station_suivante[k-1]!=ligne_station_avant[k-1]:
                        c_edge.append((transfer_linked[k],station_suivant[k]))
        # ajout des edges avec correspondance, s'il n'y a pas de transfer, le poids reste par défaut à 1, lorsqu'il y a correspondances, le poids est mis à 3.
        g.add_edges_from(edge_c)
        g.add_edges_from(c_edge, weight = temps_de_correspondance/temps_entre_stations )        ###################################################################################################@

        new_path = nx.shortest_path(g, source="Point de départ",target = "Point d'arrivée",weight='weight')

        return new_path



    def k_shortest_paths(self,G, source, target, k, weight=None):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


    def liste_transfers(self):
        return  list(self.metros[pd.isna(self.metros['C_2']) == False].station)  #stations avec correspondances

    def coordinates_liste_transfers(self):
        corr = Shortest_Route.liste_transfers(self)
        d_int = Graphique_Reseau.station_location(self)
        lat = []
        lon = []
        for items in corr:
            lat.append(d_int.get(items)[1])
            lon.append(d_int.get(items)[0])
        return lat,lon


    def liste_NOtransfers(self):
        return list(self.metros[pd.isna(self.metros['C_2']) == True].station) #stations sans correspondance


    def edgelist_transfers(self):
        liste_correspondance2 = Shortest_Route.liste_transfers(self)
        lignes_correspondance = list(tuple(self.ordre_station.loc[self.ordre_station['Station'] == liste_correspondance2[i],'res_com'])  for i in range(len(liste_correspondance2)))
        edges_c = []
        liste_station_suivante = list(tuple(self.ordre_station.loc[self.ordre_station['Station'] == liste_correspondance2[i],'Station suivante'])  for i in range(len(liste_correspondance2)))
        lignes_suivante = list(tuple(self.ordre_station.loc[self.ordre_station['Station suivante'] == liste_station_suivante[i],'res_com'])  for i in range(len(liste_station_suivante)))
        for i in range(len(liste_correspondance2)):
            for k in range(len(liste_station_suivante[i])):
                if lignes_correspondance[i]!=lignes_suivante[i]: # on ajoute la condition pour prendre en compte les changements de lignes
                    edges_c.append((liste_correspondance2[i],liste_station_suivante[i][k]))
        edges_c1 = []
        for i in range(len(liste_correspondance2)):
            for k in range(len(liste_station_suivante[i])):
                    edges_c1.append((liste_correspondance2[i],liste_station_suivante[i][k]))
        return edges_c


    def edgelist_NOtransfers(self):
        liste_sans_correspondance2 = Shortest_Route.liste_NOtransfers(self)
        liste_station_suivante_sc = list(tuple(self.ordre_station.loc[self.ordre_station['Station'] == liste_sans_correspondance2[i],'Station suivante']) for i in range (len(liste_sans_correspondance2)))
        edges_sc = []
        for i in range(len(liste_sans_correspondance2)):
            for k in range(len(liste_station_suivante_sc[i])):
                edges_sc.append((liste_sans_correspondance2[i],liste_station_suivante_sc[i][k]))
        return edges_sc

    def edgelist_source_target(self,source_x,source_y,target_x,target_y,temps_source,temps_target):
        source = list(set(Closest_Stations.stations_source(self,source_x,source_y,temps_source)))
        target = list(set(Closest_Stations.stations_target(self,target_x,target_y,temps_target)))
        new_edges = []
        extremes = ['start','end']

        for i in range(len(source)):
            new_edges.append([extremes[0],source[i]])
        for j in range(len(target)):
            new_edges.append([extremes[1],target[j]])
        return new_edges


class Closest_Stations(Shortest_Route):
    def __init__(self,filename1,filename2):
        super().__init__(filename1,filename2)

    def stations_source(self,source_x,source_y,temps_source):                       
        closest_stations_source = []
        s = temps_source * 60 * vitesse_marcheur
        for j in range(len(self.y1)):
            dist = Closest_Stations.distance_entre_points(self, source_x,source_y, self.x1[j], self.y1[j])
            if dist <= s:
                closest_stations_source.append(self.station[j])
        return closest_stations_source


    def stations_target(self,target_x,target_y,temps_target):                        
        closest_stations_target=[]
        s = temps_target * 60 * vitesse_marcheur
        for j in range(len(self.y1)):
            dist = Closest_Stations.distance_entre_points(self, target_x,target_y, self.x1[j], self.y1[j])
            if dist <= s:
                closest_stations_target.append(self.station[j])
        return closest_stations_target


    def coordinates_closest_station_source(self,source_x,source_y,temps_source):
        S = Closest_Stations.stations_source(self,source_x,source_y,temps_source)
        d_int = Graphique_Reseau.station_location(self)
        lat = []
        lon = []
        for items in S:
            lat.append(d_int.get(items)[0])
            lon.append(d_int.get(items)[1])
        return lat,lon


    def coordinates_closest_station_target(self,target_x,target_y,temps_target):
        T = Closest_Stations.stations_target(self,target_x,target_y,temps_target)
        d_int = Graphique_Reseau.station_location(self)
        lat = []
        lon = []
        for items in T:
            lat.append(d_int.get(items)[0])
            lon.append(d_int.get(items)[1])
        return lat,lon



    def distance_target_stations(self,target_x,target_y,temps_target):
        dist_target = []
        lat, lon = Closest_Stations.coordinates_closest_station_target(self,target_x,target_y,temps_target)
        for j in range(len(lat)):
            dist_target.append(Closest_Stations.distance_entre_points(self, target_x,target_y,lat[j],lon[j]))
        return dist_target



    def distance_source_stations(self,source_x,source_y,temps_source):
        lat, lon = Closest_Stations.coordinates_closest_station_source(self,source_x,source_y,temps_source)
        dist_source = []
        for j in range(len(lat)):
            dist_source.append(Closest_Stations.distance_entre_points(self,source_x,source_y,lat[j],lon[j]))
        return dist_source


    def distance_start_end(self,source_x,source_y,target_x,target_y):
        return (Closest_Stations.distance_entre_points(self,source_x,source_y,target_x,target_y))


    def temps_en_secondes(self, source_x,source_y,target_x,target_y):
        distance = Closest_Stations.distance_entre_points(self, source_x, source_y, target_x, target_y)
        temps = distance / vitesse_marcheur   #en secondes
        return temps   #en secondes

    def distance_entre_points(self,source_x, source_y, target_x,target_y):   #rend la distance calculée à partir de coordonnées gps
        latitude_point1 = source_x
        longitude_point1 = source_y
        latitude_point2 = target_x
        longitude_point2 = target_y
        R = 6372800                                      #en mètres
        phi1, phi2 = math.radians(latitude_point1), math.radians(latitude_point2)
        dphi  = math.radians(latitude_point2 - latitude_point1)
        dlambda    = math.radians(longitude_point1 - longitude_point2)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        distance = 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))  #en mètres
        return distance


    def temps_total_marche(self,source_x,source_y,target_x,target_y): #recuperer la liste des stations qui passent:
            distance = Closest_Stations.distance_entre_points(self,source_x,source_y,target_x,target_y)
            temps = Closest_Stations.temps_en_secondes(self,source_x,source_y,target_x,target_y) // 60
            print('Il vaut mieux marcher! Le trajet total a une durée de: ', temps,' minute(s)')

    def temps_total2(self,source_x,source_y,target_x,target_y,temps_source,temps_target): #recuperer la liste des stations qui passent:
        stations_visitees =  Shortest_Route.shortest_route_weighted(self,source_x,source_y,target_x,target_y,temps_source,temps_target)
        temps_total = 0
        dico = Graphique_Reseau.lignes_stations2(self)
        S_source = stations_visitees[1]  #station_source la plus proche du point de depart
        S_target = stations_visitees[-2]  #station_target la plus proche du point d'arrivée
        d_int = Graphique_Reseau.station_location(self)
        lat_source, lon_source = d_int.get(S_source)  #latitude et longitude station source
        lat_target, lon_target = d_int.get(S_target)   #latitude station target
        temps_de_marche_source = Closest_Stations.temps_en_secondes(self, source_x, source_y,lat_source, lon_source)
        print("Rejoindre la station", S_source," Ligne: ", str(set(dico.get(S_source)).intersection(set(dico.get(stations_visitees[2]))))," \n Temps de marche estimé: ", temps_de_marche_source//60, "minute(s)" )

        temps_de_marche_target = Closest_Stations.temps_en_secondes(self, target_x,target_y,lat_target,lon_target)
        temps_total = temps_de_marche_source + temps_de_marche_target
        for i in range (2,len(stations_visitees)-2):
            station_avant = stations_visitees[i-1]
            lignes_station_avant = dico.get(station_avant)
            station_apres = stations_visitees[i+1]
            lignes_station_apres = dico.get(station_apres)
            station = stations_visitees[i]
            lignes_station = dico.get(station)

            if len(set(lignes_station_avant).intersection(lignes_station_apres)) != 0 :
                temps_total += (temps_entre_stations)
            else:
                temps_total += temps_entre_stations + temps_de_correspondance
                print("Correspondance à la station", station, " \n Emprunter la ligne", str(set(lignes_station).intersection(set(lignes_station_apres))))    #afficher la ligne à prendre !!! en prenant la lignes de la station d'après  !
        
        print("S'arrêter à la station", S_target, ". Rejoindre votre point d'arrivée \n Temps de marche estimé = ", temps_de_marche_target//60 , 'minute(s)')
        temps_total_minutes = temps_total // 60
        temps_total_secondes = temps_total
        print('Le trajet a une durée totale de: ', temps_total_minutes,' minute(s)')
        return(temps_total_secondes)


if __name__=='__main__':

    """ Il faudra demander les coordonnees a 3d.p. dans le terminal"""

    #coordinates = [2.29,48.943,2.31,48.847]   #coordonnées simples avec temps de marche initial important, en vrai seulement 1 correspondance
   # coordinates = [2.451,48.812,2.341,48.852]   #coordonnées simples avec temps de marche initial de 11 min
  #  coordinates = [2.37721,48.7983,2.3847,48.8013]   #coordonnées proches affiche bien sur le graphe
    #coordinates = [2.2967,48.8609,2.300,48.8547]
    #coordinates = [2.2967,48.8609,2.34164,48.9238]   
   # coordinates = [2.2967,48.8609,2.341,48.852]
    #coordinates = [2.2967,48.8609,2.34164,48.9238]
    #coordinates = [2.2967,48.8609,2.34164,48.9238]

    dico = {}
    with open(fichier_conf,"r") as f:
        for line in f:
            line = line.strip()
            if line=="": continue
            if line[0]=="#": continue
            l = line.split('=')
            dico[l[0]] = l[1]
    path1 = dico.get("path1")
    path2 = dico.get("path2")
    vitesse_marcheur = int(dico.get("vitesse_marcheur")) / 3.6
    temps_entre_stations = int(dico.get("temps_entre_stations"))
    temps_de_correspondance = int(dico.get('temps_de_correspondance'))
    stat = Closest_Stations(path1,path2)
    dessin = Graphique_Reseau(path1,path2)
    short = Shortest_Route(path1,path2)
#    transfers = dessin.drawing_correspondance()

    reseau = dessin.drawing_edges()

    try:

        lignes = dessin.lignes_stations2()
        lat1 = float(input("Latitude de votre point de départ: "))
        lon1 = float(input("Longitude de votre point de départ: "))
        lat2 = float(input("Latitude de votre point de départ: "))
        lon2 = float(input("Longitude de votre point de départ: "))
        temps_source = int(input("Combien de temps en minutes voulez vous marcher au maximum avant d'atteindre votre station ? "))
        temps_target = int(input("Combien de temps en minutes voulez vous marcher au maximum avant d'atteindre votre destination finale ? "))

        graph = dessin.combining_graphs(lat1,lon1,lat2,lon2,temps_source,temps_target)
        #graph = dessin.combining_graphs(coordinates[0],coordinates[1],coordinates[2],coordinates[3],temps_source,temps_target)

    except ValueError as e:
        print("Désolées, il n'y a pas de station de métro à proximité, veuillez entrer d'autres coordonnées ou augmenter votre temps de marche")

    plt.show()
