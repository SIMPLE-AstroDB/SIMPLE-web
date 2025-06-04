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

function normalisation_slider(spectra_min, spectra_max, cds_list) {

    function min_check (number) {

        // check if a value is greater than a minimum value
        return number >= this.min_point;

    }

    function max_check (number) {

        // check if a value is greater than a minimum value
        return number >= this.max_point;

    }

    function median(numbers) {

        // find the median of a set of numbers
        const sorted = numbers.slice().sort((a, b) => a - b);
        const middle = Math.floor(sorted.length / 2);

        if (sorted.length % 2 === 0) {

            return (sorted[middle - 1] + sorted[middle]) / 2;

        }

        return sorted[middle];

    }

    // pick minimum and maximum wavelengths to normalise by
    let min_wave = this.value[0];
    let max_wave = this.value[1];

    if (max_wave - min_wave < 0.01) {

        max_wave = min_wave + 0.01;

    }


    // for every spectrum being plotted, perform normalisation by the median
    let cds_len = cds_list.length;

    for (let i = 0; i < cds_len; i++) {

        // select initial values
        let source = cds_list[i];
        let data = source.data;
        let wave = data.wave;
        let wave_start = wave[0];
        let wave_end = wave[wave.length - 1];
        let object_min_wave = min_wave;
        let object_max_wave = max_wave;

        // logic block of finding the minimum/maximum values to be checked, in case normalisation is outside regime
        if (wave_start > min_wave && wave_end > min_wave && wave_start > max_wave && wave_end > max_wave) {

            object_min_wave = wave_start;
            object_max_wave = wave_start + 0.01;

        }

        else if (wave_start > min_wave && wave_end > min_wave && wave_start < max_wave && wave_end >= max_wave) {

            object_min_wave = wave_start;
            object_max_wave = max_wave;

        }

        else if (wave_start <= min_wave && wave_end > min_wave && wave_start < max_wave && wave_end >= max_wave) {

            object_min_wave = min_wave;
            object_max_wave = max_wave;

        }

        else if (wave_start <= min_wave && wave_end > min_wave && wave_start < max_wave && wave_end < max_wave) {

            object_min_wave = min_wave;
            object_max_wave = wave_end;

        }

        else if (wave_start < min_wave && wave_end < min_wave && wave_start < max_wave && wave_end < max_wave) {

            object_min_wave = wave_end - 0.01;
            object_max_wave = wave_end;

        }

        if (object_max_wave - object_min_wave < 0.01) {

            object_max_wave = object_min_wave + 0.01

        }

        const comparisons = {

            min_point: object_min_wave,
            max_point: object_max_wave

        }

        // perform the normalisation of the median for given region
        let index_min = wave.findIndex(min_check, comparisons);
        let index_max = wave.findIndex(max_check, comparisons);
        let flux_region = data.flux.slice(index_min, index_max);
        let med = median(data.flux);

        if (flux_region.length > 0) {

            med = median(flux_region);

        }

        for (let j = 0; j < wave.length; j++) {

            data.normalised_flux[j] = data.flux[j] / med;

        }

        source.change.emit();
    }

    // update values on slider itself
    spectra_min.location = min_wave;
    spectra_max.location = max_wave;

}

function reset_slider (spectra_slide) {

    // resetting the spectral normalisation slider to default values 0.81--0.82um
    spectra_slide.value = [0.81, 0.82];
    spectra_slide.change.emit();

}

function reset_dropdown(dropdown_x, dropdown_y) {

    // resetting the dropdown columns when reset button pressed
    dropdown_x.value = dropdown_x.options[0][0];
    dropdown_y.value = dropdown_y.options[1][0];

    dropdown_x.change.emit();
    dropdown_y.change.emit();

}