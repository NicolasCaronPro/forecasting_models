define( [
    'text!templates/compIndicator_tmpl.html',
    'text!templates/compMapIndicatorCommon_tmpl.html',
    'lgind'

],function( tmpl1, tmpl2 ) {
    var initialize = function( that, callback, def_o ) {
        $tmpls_o = {} ;

        $(tmpl1).each(function () {
            if (this.id)
                $tmpls_o[ this.id.split("_")[0] ] = GCO5.core.GcLangManager.getInstance().injectLocaleStrings( $(this).html() ) ;
        });

        $(tmpl2).each(function () {
            if (this.id)
                $tmpls_o[ this.id.split("_")[0] ] = GCO5.core.GcLangManager.getInstance().injectLocaleStrings( $(this).html() ) ;
        });

        $.templates( $tmpls_o ) ;

        if (callback)
            callback.apply( that, [def_o] ) ;
    } ;

    return {
        initialize: initialize
    };
});
