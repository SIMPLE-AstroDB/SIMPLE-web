function dropdown_x_js(full_data, full_plot, x_range, x_button, x_axis, this_plot) {

    // find the selected dropdown option
    for (let i = 0; i < this.options.length; i++) {

        if (this.value === this.options[i][0]) {

            this.label = this.options[i][1];
            break;

        }
    }

    // work out what mag/colour is being represented by the label
    let full_value = this.value;

    if (full_value.includes('(')) {

        full_value = full_value.substr(0, full_value.indexOf('('));

    }

    // change what is being shown as the x data
    let new_x = full_data[full_value];
    let updated_x = [];

    for (let i = 0; i < new_x.length; i++) {

        let val = new_x[i];

        if (isNaN(val)) {

            continue;

        }
        updated_x.push(val);
    }

    new_x = updated_x;

    // rescale the x axis to match the data
    let min_x = Math.min(...new_x);
    let max_x = Math.max(...new_x);

    if (x_button.active) {

        x_range.start = max_x;
        x_range.end = min_x;

    }

    else {

        x_range.start = min_x;
        x_range.end = max_x;

    }

    try {

        this_plot.glyph.x.field = this.value;
        this_plot.glyph.change.emit();

    } catch ( error ) {}

    full_plot.glyph.x.field = full_value;
    x_axis.axis_label = this.label;
    full_plot.glyph.change.emit();

}

function dropdown_y_js(full_data, full_plot, y_range, y_button, y_axis, this_plot) {

    // find the selected dropdown option
    for (let i = 0; i < this.options.length; i++) {

        if (this.value === this.options[i][0]) {

            this.label = this.options[i][1];
            break;

        }
    }

    // work out what mag/colour is being represented by the label
    let full_value = this.value;

    if (full_value.includes('(')) {
        full_value = full_value.substr(0, full_value.indexOf('('));

    }

    // change what is being shown as the y data
    let new_y = full_data[full_value];
    let updated_y = [];

    for (let i = 0; i < new_y.length; i++) {

        let val = new_y[i];

        if (isNaN(val)) {

            continue;

        }
        updated_y.push(val);
    }

    new_y = updated_y;

    // rescale the y axis to match the data
    let miny = Math.min(...new_y);
    let maxy = Math.max(...new_y);

    if (y_button.active) {

        y_range.start = maxy;
        y_range.end = miny;

    }

    else {

        y_range.start = miny;
        y_range.end = maxy;

    }

    try {

        this_plot.glyph.y.field = this.value;
        this_plot.glyph.change.emit();

    } catch ( error ) {}

    full_plot.glyph.y.field = full_value;
    y_axis.axis_label = this.label;
    full_plot.glyph.change.emit();

}

function button_flip(ax_range) {

    // flip the start and end of the axis
    let newstart = ax_range.end;
    let newend = ax_range.start;
    ax_range.start = newstart;
    ax_range.end = newend;
    ax_range.change.emit()

}

function reset_dropdown(dropdown_x, dropdown_y, x_value, y_value) {

    // resetting the dropdown columns to their initial values when reset button pressed
    dropdown_x.value = x_value;
    dropdown_y.value = y_value;

    dropdown_x.change.emit();
    dropdown_y.change.emit();

}