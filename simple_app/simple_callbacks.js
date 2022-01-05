function dropdownx_js(fulldata, fullplot, xrange, xbut, xaxis, thisplot) {
    for (let i = 0; i < this.options.length; i++) {
        if (this.value === this.options[i][0]) {
            this.label = this.options[i][1];
            break;
        }
    }
    let fullvalue = this.value;
    if (fullvalue.includes('(')) {
        fullvalue = fullvalue.substr(0, fullvalue.indexOf('('));
    }
    let newx = fulldata[fullvalue];
    let newnewx = [];
    for (let i = 0; i < newx.length; i++) {
        let val = newx[i];
        if (isNaN(val)) {
            continue;
        }
        newnewx.push(val);
    }
    newx = newnewx;
    let minx = Math.min(...newx);
    let maxx = Math.max(...newx);
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
    fullplot.glyph.x.field = fullvalue;
    xaxis.axis_label = this.label;
    fullplot.glyph.change.emit();
}

function dropdowny_js(fulldata, fullplot, yrange, ybut, yaxis, thisplot) {
    for (let i = 0; i < this.options.length; i++) {
        if (this.value === this.options[i][0]) {
            this.label = this.options[i][1];
            break;
        }
    }
    let fullvalue = this.value;
    if (fullvalue.includes('(')) {
        fullvalue = fullvalue.substr(0, fullvalue.indexOf('('));
    }
    let newy = fulldata[fullvalue];
    let newnewy = [];
    for (let i = 0; i < newy.length; i++) {
        let val = newy[i];
        if (isNaN(val)) {
            continue;
        }
        newnewy.push(val);
    }
    newy = newnewy;
    let miny = Math.min(...newy);
    let maxy = Math.max(...newy);
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
    fullplot.glyph.y.field = fullvalue;
    yaxis.axis_label = this.label;
    fullplot.glyph.change.emit();
}

function button_flip(axrange) {
    let newstart = axrange.end;
    let newend = axrange.start;
    axrange.start = newstart;
    axrange.end = newend;
    axrange.change.emit()
}

function normslider(spmin, spmax, cdslist) {
    let minwave = this.value[0];
    let maxwave = this.value[1];
    console.log(minwave, maxwave, cdslist);
    spmin.location = minwave;
    spmax.location = maxwave;
}