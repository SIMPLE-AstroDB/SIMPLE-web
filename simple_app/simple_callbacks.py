"""
Dump for all the javascript callbacks to be used in website
"""


class JSCallbacks:
    dropdownx_js = """
    for (let i = 0; i < this.options.length; i++) {
        if (this.value == this.options[i][0]) {
            this.label = this.options[i][1];
            break;
        }
    }
    var newx = fulldata[this.value];
    var newnewx = [];
    for (let i = 0; i < newx.length; i++) {
        var val = newx[i];
        if (isNaN(val)) {
            continue;
        }
        newnewx.push(val);
    }
    newx = newnewx;
    var minx = Math.min(...newx);
    var maxx = Math.max(...newx);
    if (xbut.active) {
        xrange.start = maxx;
        xrange.end = minx;
    }
    else {
        xrange.start = minx;
        xrange.end = maxx;
    }
    try {
        thisplot.glyph.x.field = this.value;
        thisplot.glyph.change.emit();
    } catch ( error ) {}
    fullplot.glyph.x.field = this.value;
    xaxis.axis_label = this.label;
    fullplot.glyph.change.emit();
    """

    dropdowny_js = """
    for (let i = 0; i < this.options.length; i++) {
        if (this.value == this.options[i][0]) {
            this.label = this.options[i][1];
            break;
        }
    }
    var newy = fulldata[this.value];
    var newnewy = [];
    for (let i = 0; i < newy.length; i++) {
        var val = newy[i];
        if (isNaN(val)) {
            continue;
        }
        newnewy.push(val);
    }
    newy = newnewy;
    var miny = Math.min(...newy);
    var maxy = Math.max(...newy);
    if (ybut.active) {
        yrange.start = maxy;
        yrange.end = miny;
    }
    else {
        yrange.start = miny;
        yrange.end = maxy;
    }
    try {
        thisplot.glyph.y.field = this.value;
        thisplot.glyph.change.emit();
    } catch ( error ) {}
    fullplot.glyph.y.field = this.value;
    yaxis.axis_label = this.label;
    fullplot.glyph.change.emit();
    """

    button_flip = """
    var newstart = axrange.end;
    var newend = axrange.start;
    axrange.start = newstart;
    axrange.end = newend;
    axrange.change.emit()
    """
