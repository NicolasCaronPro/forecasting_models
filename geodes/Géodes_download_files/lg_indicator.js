(function(root) {
    var lang = 'fr',
        bundleName = 'indicator'  ;

    var lang_o = {
        pilotage: {
            H1_TITLE:           "Indicateurs : cartes, données et graphiques",
            H2_SELECT_INDICS:   "CHOISIR DES INDICATEURS",
            H2_CHANGE_GEO:      "CHANGER LE DÉCOUPAGE GÉOGRAPHIQUE",

            SEARCH_TERRITORY:   "Rechercher un territoire",
            SEARCH_PLACEHOLDER: "Chercher un nom ou une ville...",

            TYPES_IND:      "Types de représentations",
            TYPIND_ALL:     "Tous types de représentations",
            TYPIND_PIE:     "Camemberts",
            TYPIND_CHORO:   "Choroplèthes (ratios)",
            TYPIND_RC:      "Ronds coloriés",
            TYPIND_FLUX:    "Flux",
            TYPIND_OURSINS: "Oursins",
            TYPIND_POINTS:  "Ponctuels",
            TYPIND_LINS:    "Linéaires",//NEW
            TYPIND_RONDS:   "Ronds proportionnels (additifs)",
            TYPIND_SOLDES:  "Soldes",
            TYPIND_TYPO:    "Typologies",
            TYPIND_ZON:     "Zonages",

            SHOW_INDICS_ON_MAP: "Voir les indicateurs sur la carte interactive",
            SHOW_TABLE:         "Voir le tableau avec tous les indicateurs chargés",
            SHOW_SYNTH:         "Voir une fiche pour chaque indicateur",

            ACC1_TITLE: "Partager, imprimer, exporter",
            ACC1_TXT:   "Pour partager, imprimer, exporter, sauvegarder, ajouter des couches, voir le menu ACTIONS, en haut à droite de chaque restitution, CARTE, TABLEAU ou SYNTHÈSE",
            ACC2_TITLE: "Jouer avec les zonages",
            ACC2_TXT:   "Pour afficher des zonages, de nouvelles couches d'habillage, des plans ou des photos aériennes, voir le menu ACTIONS, en haut à droite de la carte.",
            ACC2_TXT2:  "Vous pouvez aussi, dans un espace dédié, visualiser et comparer un large éventail de découpages, ou lister les zonages englobant un territoire donné.",
            ACC2_LINK:  "ESPACE ZONAGES",
            ACC3_TITLE: "Editer des rapports",
            ACC3_TXT:   "Délimiter un territoire d'étude et produire un rapport détaillé",
            ACC3_LINK:  "ESPACE RAPPORTS",
            ACC4_TITLE: "Charger des données externes",
            ACC4_TXT:   "Intégrer des données par copier-coller, dans un espace dédié, depuis un tableur ou par connexion à un serveur externe",
            ACC4_LINK:  "ESPACE DONNÉES EXTERNES",

            TABLE_PREPARING:    "Composition du tableau en cours...",
            CHARTS_COMPARISONS: "Graphiques et comparaisons",
            DETAILED_DOC:       "Documentation détaillée",
            CHANGE_COMP_ZONE:   "Changer la zone de comparaison",

            INDIC_VALREF:       "Valeur(s) de référence",
            INDIC_STDEV:        "écart-type",
            INDIC_NB_VALID_OBS: "nb. d'observations valides",
            INDIC_LIM_UTIL:     "Limites",
            INDIC_LIM_UTIL2:    "Conditions d'utilisation",
            INDIC_DESC:         "Description",
            INDIC_PREC:         "Précisions",
            INDIC_STATS:        "Stats",
            INDIC_KNOW_MORE:    "Pour en savoir +",
            INDIC_PRES:         "Présentation",
            INDIC_DISTRIB1:     "Analyse de la distribution par",
            INDIC_NODISPLAYDATA:"données non diffusibles",
            INDIC_STAT:         "Statistique",
            INDIC_VALID_OBS:    "observations valides",
            INDIC_SSSTAT_OBS:   "observations sous secret stat.",
            INDIC_MEAN_VAL:     "moyenne",
            INDIC_NB_VALS:      "nb. de valeurs",
            INDIC_MEDIANE:      "médiane",
            INDIC_STAT_DISTRIB: "Distribution statistique",
            INDIC_XTABLE:       "Tableau comparatif multi-indicateurs",

            SEARCH_BY_KEY_WORD: "par mot-clé",
            BACK_TO_TAG_CLOUD:  "Retour au nuage de mots-clés",

            INDIC_CLEAR_FILTERS: "Effacer tous les filtres",
            INDIC_FILTERS_HINT:  "Filtrer par type de représentation ou afficher seulement les indicateurs visibles sur cette carte",
            INDIC_OTHER_FILTERS: "Autres filtres",

            INDIC_ESSENTIALS:               "indicateurs essentiels",
            INDIC_ESSENTIALS_HINT:          "Réduire aux indicateurs les plus essentiels",
            INDIC_VISIBLE_ON_THIS_MAP:      "visibles sur cette carte",
            INDIC_VISIBLE_ON_THIS_MAP_HINT: "Réduire aux indicateurs compatibles avec l'étendue et le niveau de détail de la carte",

            INDIC_SORT_BY_TYPE:             "classer par type",

            INDIC_MY_INDICS:                "Mes indicateurs",
            INDIC_AVAIL_OTHER_MAP:          "indicateur disponible sur une autre carte",
            LEVEL_NOCOMPAT_INDIC:           "Non compatible avec cet indicateur",

            TREE_OPEN:              "Cliquer pour dérouler",
            TREE_LIST_INDICS:       "Cliquer pour consulter les indicateurs de ce thème",
            TREE_BACK_TO:           "Cliquer pour revenir à l'arborescence",
            TREE_ESSENTIALS_ONLY:   "indicateurs essentiels seulement",
            TREE_SORT_BY_TYPE_HINT: "Distinguer par type de représentation cartographique",

            INDIC_VISIBLE_ON_ANOTHER_MAP:   "visible sur une autre carte",
            TREE_SORT_BY_TYPE:              "classer par type",
            TREE_TYPE_LEGEND:               "Légende des types de représentation",
            INDIC_TYPO:                     "typologies",
            INDIC_RATIOS:                   "taux",
            INDIC_ADDITIVES:                "additifs",
            INDIC_POINTS:                   "ponctuels",
            INDIC_FLUX:                     "flux",
            MAP_AT_THIS_LEVEL:              "Cartographier à ce niveau",
            LOCATE_IN_TREE:                 "Localiser dans l'arborescence",
            LINK_BOTH_MAPS:                 "Lier les 2 cartes (même cadrage géographique)",

            ADD_INDIC_TREE:                 "Ajoutez un indicateur en parcourant l'arborescence ci-dessus...",

            PREVIOUS: "precédent",
            NEXT: "suivant",

            TYPES_REPRESENTATIONS: "Types de représentations"
        },

        restit: {
            TABS_1MAP:      "CARTE SEULE",
            TABS_MAP_TABLE: "CARTE + TABLEAU",
            TABS_MAP_SYNTH: "CARTE + SYNTHÈSE",
            TABS_2MAPS:     "2 CARTES",

            DRAG_TO_OTHER_VIEW  :  "Faire glisser pour transférer vers l'autre carte" ,

            EXPORT_FICINDIC: "Exporter la fiche de synthèse",
            HELP_FICINDIC:   "Aide fiche de synthèse",
            PRINT_FICINDIC:  "Imprimer la fiche de synthèse",

            COMPAR_TABLE:    "Tableau comparatif",
            COMPAR_SEL_REF:  "Comparaison Sélection / Zone de référence",
            NO_COMPAR_DATA:  "pas de données comparatives"


        }
    } ;

    window.GCO5.core.GcLangManager.getInstance().addBundle( lang_o, lang, bundleName );

}(this));