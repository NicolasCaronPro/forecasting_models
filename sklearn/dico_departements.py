import datetime as dt
import random

# Liste des départements avec leurs codes
departements_list = [
    (1, 'Ain'), (2, 'Aisne'), (3, 'Allier'), (4, 'Alpes-de-Haute-Provence'),
    (5, 'Hautes-Alpes'), (6, 'Alpes-Maritimes'), (7, 'Ardeche'), (8, 'Ardennes'),
    (9, 'Ariege'), (10, 'Aube'), (11, 'Aude'), (12, 'Aveyron'),
    (13, 'Bouches-du-Rhone'), (14, 'Calvados'), (15, 'Cantal'), (16, 'Charente'),
    (17, 'Charente-Maritime'), (18, 'Cher'), (19, 'Correze'), (21, 'Cote-d-Or'),
    (22, 'Cotes-d-Armor'), (23, 'Creuse'), (24, 'Dordogne'), (25, 'Doubs'),
    (26, 'Drome'), (27, 'Eure'), (28, 'Eure-et-Loir'), (29, 'Finistere'),
    ('2A', 'Corse-du-Sud'), ('2B', 'Haute-Corse'), (30, 'Gard'), (31, 'Haute-Garonne'),
    (32, 'Gers'), (33, 'Gironde'), (34, 'Herault'), (35, 'Ille-et-Vilaine'),
    (36, 'Indre'), (37, 'Indre-et-Loire'), (38, 'Isere'), (39, 'Jura'),
    (40, 'Landes'), (41, 'Loir-et-Cher'), (42, 'Loire'), (43, 'Haute-Loire'),
    (44, 'Loire-Atlantique'), (45, 'Loiret'), (46, 'Lot'), (47, 'Lot-et-Garonne'),
    (48, 'Lozere'), (49, 'Maine-et-Loire'), (50, 'Manche'), (51, 'Marne'),
    (52, 'Haute-Marne'), (53, 'Mayenne'), (54, 'Meurthe-et-Moselle'), (55, 'Meuse'),
    (56, 'Morbihan'), (57, 'Moselle'), (58, 'Nievre'), (59, 'Nord'),
    (60, 'Oise'), (61, 'Orne'), (62, 'Pas-de-Calais'), (63, 'Puy-de-Dome'),
    (64, 'Pyrenees-Atlantiques'), (65, 'Hautes-Pyrenees'), (66, 'Pyrenees-Orientales'),
    (67, 'Bas-Rhin'), (68, 'Haut-Rhin'), (69, 'Rhone'), (70, 'Haute-Saone'),
    (71, 'Saone-et-Loire'), (72, 'Sarthe'), (73, 'Savoie'), (74, 'Haute-Savoie'),
    (75, 'Paris'), (76, 'Seine-Maritime'), (77, 'Seine-et-Marne'), (78, 'Yvelines'),
    (79, 'Deux-Sevres'), (80, 'Somme'), (81, 'Tarn'), (82, 'Tarn-et-Garonne'),
    (83, 'Var'), (84, 'Vaucluse'), (85, 'Vendee'), (86, 'Vienne'),
    (87, 'Haute-Vienne'), (88, 'Vosges'), (89, 'Yonne'), (90, 'Territoire de Belfort'),
    (91, 'Essonne'), (92, 'Hauts-de-Seine'), (93, 'Seine-Saint-Denis'), (94, 'Val-de-Marne'),
    (95, 'Val-d-Oise'), (971, 'Guadeloupe'), (972, 'Martinique'), (973, 'Guyane'),
    (974, 'La Reunion'), (976, 'Mayotte')
]

# Laurent steack

int2str = {code: name.lower().replace("'", "-") for code, name in departements_list}
int2strMaj = {code: name for code, name in departements_list}
int2name = {code: f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}" for code, name in departements_list}

str2int = {name.lower().replace("'", "-"): code for code, name in departements_list}
str2intMaj = {name: code for code, name in departements_list}
str2name = {name: f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}" for code, name in departements_list}

name2str = {f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": name for code, name in departements_list}
name2int = {f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": code for code, name in departements_list}
name2strlow = {f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": name.lower() for code, name in departements_list}
name2intstr = {
    f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": (f'0{code}' if code not in ['2A', '2B'] and int(code) < 10 else str(code))
    for code, name in departements_list
}

# Départements déjà définis
SAISON_FEUX = {
    1: {'jour_debut': '01', 'mois_debut': '03', 'jour_fin': '01', 'mois_fin': '11'},
    25: {'jour_debut': '01', 'mois_debut': '03', 'jour_fin': '01', 'mois_fin': '10'},
    78: {'jour_debut': '01', 'mois_debut': '03', 'jour_fin': '01', 'mois_fin': '10'},
    69: {'jour_debut': '01', 'mois_debut': '03', 'jour_fin': '01', 'mois_fin': '10'},
}

# Ajouter ou mettre à jour les départements restants
for code, name in departements_list:
    dept_key = code
    if dept_key not in SAISON_FEUX:
        SAISON_FEUX[dept_key] = {'jour_debut': '01', 'mois_debut': '03', 'jour_fin': '01', 'mois_fin': '10'}

# Définir les périodes à ignorer pour les départements spécifiques
PERIODES_A_IGNORER = {
    1: {'interventions': [(dt.datetime(2017, 6, 12), dt.datetime(2017, 12, 31)),
                          (dt.datetime(2023, 7, 3), dt.datetime(2023, 7, 26)),
                          (dt.datetime(2024, 3, 10), dt.datetime(2024, 5, 29))],
        'appels': [(dt.datetime(2023, 3, 5), dt.datetime(2023, 7, 19)),
                   (dt.datetime(2024, 4, 1), dt.datetime(2024, 5, 24))]},
    25: {'interventions': [],
         'appels': []},
    78: {'interventions': [],
         'appels': []},
    69: {'interventions': [#(dt.datetime(2023, 1, 1), dt.datetime(2024, 6, 26)),
                           (dt.datetime(2017, 6, 12), dt.datetime(2017, 12, 31))],
         'appels': []}
}

# Ajouter les départements manquants avec des listes vides
for code, _ in departements_list:
    if code not in PERIODES_A_IGNORER:
        PERIODES_A_IGNORER[code] = {'interventions': [], 'appels': []}

####################################################################################################

def select_departments(database, sinister):

    if database == 'bdiff':
        
        departements = [
            'departement-01-ain',
            'departement-02-aisne',
            'departement-03-allier',
            'departement-04-alpes-de-haute-provence',
            'departement-05-hautes-alpes',
            'departement-06-alpes-maritimes',
            'departement-07-ardeche',
            'departement-08-ardennes',
            'departement-09-ariege',
            'departement-10-aube',
            'departement-11-aude',
            'departement-12-aveyron',
            'departement-13-bouches-du-rhone',
            'departement-14-calvados',
            'departement-15-cantal',
            'departement-16-charente',
            'departement-17-charente-maritime',
            'departement-18-cher',
            'departement-19-correze',
            'departement-21-cote-d-or',
            'departement-22-cotes-d-armor',
            'departement-23-creuse',
            'departement-24-dordogne',
            'departement-25-doubs',
            'departement-26-drome',
            'departement-27-eure',
            'departement-28-eure-et-loir',
            'departement-29-finistere',
            #'departement-2A-corse-du-sud', 'departement-2B-haute-corse',
            'departement-30-gard',
            'departement-31-haute-garonne',
            'departement-32-gers',
            'departement-33-gironde',
            'departement-34-herault',
            'departement-35-ille-et-vilaine',
            'departement-36-indre', 'departement-37-indre-et-loire', 'departement-38-isere', 'departement-39-jura',
            'departement-40-landes',
            'departement-41-loir-et-cher', 'departement-42-loire', 'departement-43-haute-loire',
            'departement-44-loire-atlantique', 'departement-45-loiret', 'departement-46-lot', 'departement-47-lot-et-garonne',
            'departement-48-lozere', 'departement-49-maine-et-loire', 'departement-50-manche', 'departement-51-marne',
            'departement-52-haute-marne', 'departement-53-mayenne', 'departement-54-meurthe-et-moselle', 'departement-55-meuse',
            'departement-56-morbihan', 'departement-57-moselle', 'departement-58-nievre', 'departement-59-nord',
            'departement-60-oise', 'departement-61-orne', 'departement-62-pas-de-calais', 'departement-63-puy-de-dome',
            'departement-64-pyrenees-atlantiques','departement-65-hautes-pyrenees',
            'departement-66-pyrenees-orientales',
            'departement-67-bas-rhin', 'departement-68-haut-rhin', 'departement-69-rhone', 'departement-70-haute-saone',
            'departement-71-saone-et-loire', 'departement-72-sarthe', 'departement-73-savoie', 'departement-74-haute-savoie',
            'departement-75-paris', 'departement-76-seine-maritime', 'departement-77-seine-et-marne', 'departement-78-yvelines',
            'departement-79-deux-sevres', 'departement-80-somme', 'departement-81-tarn', 'departement-82-tarn-et-garonne',
            'departement-83-var',
            'departement-84-vaucluse',
            'departement-85-vendee', 'departement-86-vienne',
            'departement-87-haute-vienne', 'departement-88-vosges', 'departement-89-yonne', 'departement-90-territoire-de-belfort',
            'departement-91-essonne', 'departement-92-hauts-de-seine', 'departement-93-seine-saint-denis', 'departement-94-val-de-marne',
            'departement-95-val-d-oise',
        ]

        #random.seed(42)

        # Calculer 10 % de la liste
        #sample_size = int(len(departements) * 0.10)

        # Sélectionner aléatoirement 10 % des départements
        #train_departements = random.sample(departements, sample_size)
        train_departements = departements
        print(f' Train departement selected : {train_departements}')

    elif database == 'bdiff_small':
        departements = ['departement-13-bouches-du-rhone', 'departement-34-herault']
        train_departements = ['departement-13-bouches-du-rhone', 'departement-34-herault']
        
    elif database == 'georisques':
        departements = [
            'departement-01-ain',
            #'departement-02-aisne',
            #'departement-03-allier',
            #'departement-04-alpes-de-haute-provence',
            #'departement-05-hautes-alpes',
            'departement-06-alpes-maritimes',
            #'departement-07-ardeche',
            #'departement-08-ardennes',
            #'departement-09-ariege',
            #'departement-10-aube',
            #'departement-11-aude',
            #'departement-12-aveyron',
            #'departement-13-bouches-du-rhone',
            #'departement-14-calvados',
            #'departement-15-cantal',
            #'departement-16-charente',
            #'departement-17-charente-maritime',
            #'departement-18-cher',
            #'departement-19-correze',
            #'departement-21-cote-d-or',
            #'departement-22-cotes-d-armor',
            #'departement-23-creuse',
            #'departement-24-dordogne',
            'departement-25-doubs',
            #'departement-26-drome',
            #'departement-27-eure',
            #'departement-28-eure-et-loir',
            #'departement-29-finistere',
            #'departement-2A-corse-du-sud', 'departement-2B-haute-corse',
            #'departement-30-gard', 'departement-31-haute-garonne',
            #'departement-32-gers', 'departement-33-gironde', 'departement-34-herault', 'departement-35-ille-et-vilaine',
            #'departement-36-indre', 'departement-37-indre-et-loire', 'departement-38-isere', 'departement-39-jura',
            #'departement-40-landes',
            #'departement-41-loir-et-cher', 'departement-42-loire', 'departement-43-haute-loire',
            #'departement-44-loire-atlantique', 'departement-45-loiret', 'departement-46-lot', 'departement-47-lot-et-garonne',
            #'departement-48-lozere', 'departement-49-maine-et-loire', 'departement-50-manche', 'departement-51-marne',
            #'departement-52-haute-marne', 'departement-53-mayenne', 'departement-54-meurthe-et-moselle', 'departement-55-meuse',
            #'departement-56-morbihan', 'departement-57-moselle', 'departement-58-nievre', 'departement-59-nord',
            #'departement-60-oise', 'departement-61-orne', 'departement-62-pas-de-calais', 'departement-63-puy-de-dome',
            #'departement-64-pyrenees-atlantiques','departement-65-hautes-pyrenees',
            #'departement-66-pyrenees-orientales',
            #'departement-67-bas-rhin', 'departement-68-haut-rhin', 'departement-69-rhone', 'departement-70-haute-saone',
            #'departement-71-saone-et-loire', 'departement-72-sarthe', 'departement-73-savoie', 'departement-74-haute-savoie',
            #'departement-75-paris', 'departement-76-seine-maritime', 'departement-77-seine-et-marne', 'departement-78-yvelines',
            #'departement-79-deux-sevres', 'departement-80-somme', 'departement-81-tarn', 'departement-82-tarn-et-garonne',
            #'departement-83-var', 'departement-84-vaucluse', 'departement-85-vendee', 'departement-86-vienne',
            #'departement-87-haute-vienne', 'departement-88-vosges', 'departement-89-yonne', 'departement-90-territoire-de-belfort',
            #'departement-91-essonne', 'departement-92-hauts-de-seine', 'departement-93-seine-saint-denis', 'departement-94-val-de-marne',
            #'departement-95-val-d-oise',
            #'departement-971-guadeloupe', 'departement-972-martinique', 'departement-973-guyane', 'departement-974-la-reunion', 'departement-976-mayotte'
        ]

        train_departements = [
            'departement-01-ain',
            #'departement-02-aisne',
            #'departement-03-allier',
            #'departement-04-alpes-de-haute-provence',
            #'departement-05-hautes-alpes',
            'departement-06-alpes-maritimes',
            #'departement-07-ardeche',
            #'departement-08-ardennes',
            #'departement-09-ariege',
            #'departement-10-aube',
            #'departement-11-aude',
            #'departement-12-aveyron',
            #'departement-13-bouches-du-rhone',
            #'departement-14-calvados',
            #'departement-15-cantal',
            #'departement-16-charente',
            #'departement-17-charente-maritime',
            #'departement-18-cher',
            #'departement-19-correze',
            #'departement-21-cote-d-or',
            #'departement-22-cotes-d-armor',
            #'departement-23-creuse',
            #'departement-24-dordogne',
            'departement-25-doubs',
            #'departement-26-drome',
            #'departement-27-eure',
            #'departement-28-eure-et-loir',
            #'departement-29-finistere',
            #'departement-2A-corse-du-sud', 'departement-2B-haute-corse',
            #'departement-30-gard', 'departement-31-haute-garonne',
            #'departement-32-gers', 'departement-33-gironde', 'departement-34-herault', 'departement-35-ille-et-vilaine',
            #'departement-36-indre', 'departement-37-indre-et-loire', 'departement-38-isere', 'departement-39-jura',
            #'departement-40-landes',
            #'departement-41-loir-et-cher', 'departement-42-loire', 'departement-43-haute-loire',
            #'departement-44-loire-atlantique', 'departement-45-loiret', 'departement-46-lot', 'departement-47-lot-et-garonne',
            #'departement-48-lozere', 'departement-49-maine-et-loire', 'departement-50-manche', 'departement-51-marne',
            #'departement-52-haute-marne', 'departement-53-mayenne', 'departement-54-meurthe-et-moselle', 'departement-55-meuse',
            #'departement-56-morbihan', 'departement-57-moselle', 'departement-58-nievre', 'departement-59-nord',
            #'departement-60-oise', 'departement-61-orne', 'departement-62-pas-de-calais', 'departement-63-puy-de-dome',
            #'departement-64-pyrenees-atlantiques','departement-65-hautes-pyrenees',
            #'departement-66-pyrenees-orientales',
            #'departement-67-bas-rhin', 'departement-68-haut-rhin', 'departement-69-rhone', 'departement-70-haute-saone',
            #'departement-71-saone-et-loire', 'departement-72-sarthe', 'departement-73-savoie', 'departement-74-haute-savoie',
            #'departement-75-paris', 'departement-76-seine-maritime', 'departement-77-seine-et-marne', 'departement-78-yvelines',
            #'departement-79-deux-sevres', 'departement-80-somme', 'departement-81-tarn', 'departement-82-tarn-et-garonne',
            #'departement-83-var', 'departement-84-vaucluse', 'departement-85-vendee', 'departement-86-vienne',
            #'departement-87-haute-vienne', 'departement-88-vosges', 'departement-89-yonne', 'departement-90-territoire-de-belfort',
            #'departement-91-essonne', 'departement-92-hauts-de-seine', 'departement-93-seine-saint-denis', 'departement-94-val-de-marne',
            #'departement-95-val-d-oise',
            #'departement-971-guadeloupe', 'departement-972-martinique', 'departement-973-guyane', 'departement-974-la-reunion', 'departement-976-mayotte'
        ]

    elif database == 'firemen2':
        if sinister == 'firepoint':
            departements = [
            'departement-01-ain',
            #'departement-02-aisne',
            #'departement-03-allier',
            #'departement-04-alpes-de-haute-provence',
            #'departement-05-hautes-alpes',
            #'departement-06-alpes-maritimes',
            #'departement-07-ardeche',
            #'departement-08-ardennes',
            #'departement-09-ariege',
            #'departement-10-aube',
            #'departement-11-aude',
            #'departement-12-aveyron',
            #'departement-13-bouches-du-rhone',
            #'departement-14-calvados',
            #'departement-15-cantal',
            #'departement-16-charente',
            #'departement-17-charente-maritime',
            #'departement-18-cher',
            #'departement-19-correze',
            #'departement-21-cote-d-or',
            #'departement-22-cotes-d-armor',
            #'departement-23-creuse',
            #'departement-24-dordogne',
            'departement-25-doubs',
            #'departement-26-drome',
            #'departement-27-eure',
            #'departement-28-eure-et-loir',
            #'departement-29-finistere',
            #'departement-2A-corse-du-sud', 'departement-2B-haute-corse',
            #'departement-30-gard',
            #'departement-31-haute-garonne',
            #'departement-32-gers',
            #'departement-33-gironde',
            #'departement-34-herault',
            #'departement-35-ille-et-vilaine',
            #'departement-36-indre', 'departement-37-indre-et-loire', 'departement-38-isere', 'departement-39-jura',
            #'departement-40-landes',
            #'departement-41-loir-et-cher', 'departement-42-loire', 'departement-43-haute-loire',
            #'departement-44-loire-atlantique', 'departement-45-loiret', 'departement-46-lot', 'departement-47-lot-et-garonne',
            #'departement-48-lozere', 'departement-49-maine-et-loire', 'departement-50-manche', 'departement-51-marne',
            #'departement-52-haute-marne', 'departement-53-mayenne', 'departement-54-meurthe-et-moselle', 'departement-55-meuse',
            #'departement-56-morbihan', 'departement-57-moselle', 'departement-58-nievre', 'departement-59-nord',
            #'departement-60-oise', 'departement-61-orne', 'departement-62-pas-de-calais', 'departement-63-puy-de-dome',
            #'departement-64-pyrenees-atlantiques','departement-65-hautes-pyrenees',
            #'departement-66-pyrenees-orientales',
            #'departement-67-bas-rhin', 'departement-68-haut-rhin',
            'departement-69-rhone',
            #'departement-70-haute-saone',
            #'departement-71-saone-et-loire', 'departement-72-sarthe', 'departement-73-savoie', 'departement-74-haute-savoie',
            #'departement-75-paris', 'departement-76-seine-maritime', 'departement-77-seine-et-marne',
            'departement-78-yvelines',
            #'departement-79-deux-sevres', 'departement-80-somme', 'departement-81-tarn', 'departement-82-tarn-et-garonne',
            #'departement-83-var',
            #'departement-84-vaucluse',
            #'departement-85-vendee', 'departement-86-vienne',
            #'departement-87-haute-vienne', 'departement-88-vosges', 'departement-89-yonne', 'departement-90-territoire-de-belfort',
            #'departement-91-essonne', 'departement-92-hauts-de-seine', 'departement-93-seine-saint-denis', 'departement-94-val-de-marne',
            #'departement-95-val-d-oise',
            ]
            
            train_departements = [
            'departement-01-ain',
            #'departement-02-aisne',
            #'departement-03-allier',
            #'departement-04-alpes-de-haute-provence',
            #'departement-05-hautes-alpes',
            #'departement-06-alpes-maritimes',
            #'departement-07-ardeche',
            #'departement-08-ardennes',
            #'departement-09-ariege',
            #'departement-10-aube',
            #'departement-11-aude',
            #'departement-12-aveyron',
            #'departement-13-bouches-du-rhone',
            #'departement-14-calvados',
            #'departement-15-cantal',
            #'departement-16-charente',
            #'departement-17-charente-maritime',
            #'departement-18-cher',
            #'departement-19-correze',
            #'departement-21-cote-d-or',
            #'departement-22-cotes-d-armor',
            #'departement-23-creuse',
            #'departement-24-dordogne',
            'departement-25-doubs',
            #'departement-26-drome',
            #'departement-27-eure',
            #'departement-28-eure-et-loir',
            #'departement-29-finistere',
            #'departement-2A-corse-du-sud', 'departement-2B-haute-corse',
            #'departement-30-gard',
            #'departement-31-haute-garonne',
            #'departement-32-gers',
            #'departement-33-gironde',
            #'departement-34-herault',
            #'departement-35-ille-et-vilaine',
            #'departement-36-indre', 'departement-37-indre-et-loire', 'departement-38-isere', 'departement-39-jura',
            #'departement-40-landes',
            #'departement-41-loir-et-cher', 'departement-42-loire', 'departement-43-haute-loire',
            #'departement-44-loire-atlantique', 'departement-45-loiret', 'departement-46-lot', 'departement-47-lot-et-garonne',
            #'departement-48-lozere', 'departement-49-maine-et-loire', 'departement-50-manche', 'departement-51-marne',
            #'departement-52-haute-marne', 'departement-53-mayenne', 'departement-54-meurthe-et-moselle', 'departement-55-meuse',
            #'departement-56-morbihan', 'departement-57-moselle', 'departement-58-nievre', 'departement-59-nord',
            #'departement-60-oise', 'departement-61-orne', 'departement-62-pas-de-calais', 'departement-63-puy-de-dome',
            #'departement-64-pyrenees-atlantiques','departement-65-hautes-pyrenees',
            #'departement-66-pyrenees-orientales',
            #'departement-67-bas-rhin', 'departement-68-haut-rhin',
            'departement-69-rhone',
            #'departement-70-haute-saone',
            #'departement-71-saone-et-loire', 'departement-72-sarthe', 'departement-73-savoie', 'departement-74-haute-savoie',
            #'departement-75-paris', 'departement-76-seine-maritime', 'departement-77-seine-et-marne',
            'departement-78-yvelines',
            #'departement-79-deux-sevres', 'departement-80-somme', 'departement-81-tarn', 'departement-82-tarn-et-garonne',
            #'departement-83-var',
            #'departement-84-vaucluse',
            #'departement-85-vendee', 'departement-86-vienne',
            #'departement-87-haute-vienne', 'departement-88-vosges', 'departement-89-yonne', 'departement-90-territoire-de-belfort',
            #'departement-91-essonne', 'departement-92-hauts-de-seine', 'departement-93-seine-saint-denis', 'departement-94-val-de-marne',
            #'departement-95-val-d-oise',
            ]

        elif sinister == 'inondation':
            departements = ['departement-25-doubs']
            train_departements = ['departement-25-doubs']
        else:
            pass

    elif database == 'firemen':
        if sinister == 'firepoint':
            departements = ['departement-01-ain',
                            'departement-25-doubs', 'departement-69-rhone', 'departement-78-yvelines']
            train_departements = ['departement-01-ain', 'departement-25-doubs',
                                  'departement-69-rhone',
                                  'departement-78-yvelines'
                                  ]
            
        elif sinister == 'inondation':
            departements = ['departement-25-doubs']
            train_departements = ['departement-25-doubs']
        else:
            pass
    elif database == 'vigicrues':
        departements = ['departement-01-ain']
        train_departements = ['departement-01-ain']
    elif database == 'bouches_du_rhone':
        departements = ['departement-13-bouches-du-rhone']
        train_departements = ['departement-13-bouches-du-rhone']
    else:
        pass

    return departements, train_departements