require.config( {
    baseUrl: 'js/libs',

    paths: {
        lgmain:'gco5/' + 'lang/' + GCO5._lang + '/lg_main',
        lgreport:'gco5/' + 'lang/' + GCO5._lang + '/lg_report',
        lgind:'gco5/' + 'lang/' + GCO5._lang + '/lg_indicator',
        lgzonage:'gco5/' + 'lang/' + GCO5._lang + '/lg_zonage',
        lgexternaldata:'gco5/' + 'lang/' + GCO5._lang + '/lg_externaldata',
        jquery: ['https://code.jquery.com/jquery-3.5.1.min', 'jquery/jquery-3.5.1.min'],  // fallback en l'absence de connexion internet
        text: 'require/'+'text',
        async: 'require/'+'async',
        gc_core: 'gco5/' + 'gc_core',
        proj4:'gco5/'+'modules/' + 'proj4',
        xlsx:'xlsx/' + 'xlsx',
        mapthmsettings:'gco5/' + 'modules/'+ 'gc_mapthmsettings',
        mapgeosettings:'gco5/' + 'modules/'+ 'gc_mapgeosettings',
        tablesettings:'gco5/' + 'modules/'+ 'gc_tablesettings',
        "datatables.net": [ 'datatables/' + 'jquery.dataTables'], // 'https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min'
        select:['datatables/'+'dataTables.select'], // 'https://cdn.datatables.net/select/1.2.7/js/dataTables.select.min'
        jszip:['zip/' + 'jszip','https://cdnjs.cloudflare.com/ajax/libs/jszip/2.5.0/jszip.min'],
        dtResponsive:['datatables/Responsive-2.1.0/js/dataTables.responsive.min'], // 'https://cdn.datatables.net/responsive/2.1.0/js/dataTables.responsive.min',
        templates: 'gco5/' +'templates'
    },

    waitSeconds:0,

    shim: {  //décrit les dépendances, et détermine donc l'ordre de chargement
        gc_core: {
            deps: ['jquery'],
            exports: 'gc_core'
        },
        proj4: {
            deps: ['gc_core'],
            exports: 'proj4'
        },
        xlsx: {
            deps: ['jszip'],
            exports: 'xlsx'
        },

        jszip: {
            deps: ['gc_core'],
            exports: 'jszip'
        },
        gc_mobile: {
            deps: ['gc_core'],
            exports: 'gc_mobile'
        },
        mapsettings: {
            deps: ['gc_core'],
            exports: 'proj4'
        },
        lgmain: {
            deps: ['gc_core'],
            exports: 'lgmain'
        },
        select: {
            deps: ['datatables.net']
        },
        dtButtonsHtml5: {
            deps: ['datatables.net']
        },
        dtButtons: {
            deps: ['datatables.net']
        }
    }
  }
);

require(   ['jquery',
            'gc_core',
            'text!templates/pageApp_tmpl.html',
            //'text!templates/pageHome_tmpl.html',
            'lgmain'],

    function ( $, Gco5, tmpl1, tmpl2 ) {
        $( function() {    //dom ready, ne s'exécute qu'une fois
            var gclg = GCO5.core.GcLangManager.getInstance() ;
            gclg.setLang(GCO5._lang) ;
            var $tmpls_o = {} ;

            $( tmpl1 + tmpl2 ).each( function () {
                if ( this.id )  //remplacement des chaines de langue par leur déclinaison en langue locale
                    $tmpls_o[ this.id.split("_")[0] ] = gclg.injectLocaleStrings( $(this).html() ) ;
            });

            $.templates( $tmpls_o );  //jsrender, mise en mémoire des templates chargés

            var facade = GCO5.ApplicationFacade.getInstance( GCO5.ApplicationFacade.NAME ) ;  //PUREMVC lancement

            facade.startup() ;
        } );
     }
);



