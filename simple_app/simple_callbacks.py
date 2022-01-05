"""
Dump for all the javascript callbacks to be used in website
"""


class JSCallbacks:
    dropdownx_js = ''
    dropdowny_js = ''
    button_flip = ''

    def __init__(self):
        jsfuncnames = ('dropdownx_js', 'dropdowny_js', 'button_flip')
        with open('simple_app/simple_callbacks.js', 'r') as fcall:
            whichvar = ''
            outstr = """"""
            for line in fcall:
                for funcname in jsfuncnames:
                    if funcname in line:
                        whichvar = funcname
                        outstr = """"""
                        break
                else:
                    if line.startswith('}'):
                        setattr(self, whichvar, outstr)
                        whichvar = ''
                        outstr = """"""
                        continue
                    outstr = '\n'.join([outstr, line.strip('\n')])
