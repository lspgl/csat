import inspect
import sys
import os
# sys.path.append(os.path.abspath('../'))
# sys.path.append(os.path.abspath('./'))
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(__location__ + '/../')


def getDocFromClass(cl):
    docDict = {}

    for member in inspect.getmembers(cl):
        if member[0][:2] != '__' or member[0] == '__init__':
            fname = str(member[0])
            sig = str(inspect.signature(member[1]))
            doc = str(member[1].__doc__).replace('        ', '')
            docDict[fname] = (sig, doc)
    return docDict


def parseTemplate(fn, fn_css='css/style.css'):
    with open(os.path.join(__location__, fn)) as f:
        read_data = f.read()
    with open(os.path.join(__location__, fn_css)) as f:
        read_css = f.read()
    style_external = '<link href="css/style.css" rel="stylesheet" type="text/css">'
    read_data = read_data.replace(style_external, '<style>' + read_css + '</style>')
    read_data_ln = read_data.split('\n')

    proc = []
    marker = {}
    block = ''
    for ln in read_data_ln:
        if '@_' not in ln:
            block += ln + '\n'
        else:
            proc.append(block)
            start = ln.index('@_')
            trimm = ln[start:]
            if ' ' in trimm:
                end = trimm.index(' ')
                key = trimm[:end]
            else:
                key = trimm
            idx = len(proc)
            key = key[2:]
            marker[key] = idx
            proc.append(key)
            block = ''
    if block != '':
        proc.append(block)
    return proc, marker


def formatDoc(docDict):
    param_delim = '\nParameters\n----------\n'
    ret_delim = '\nReturns\n-------\n'
    maincontent = ''
    for key in docDict:
        params = False
        ret = False
        sig, doc = docDict[key]

        func_string = ('<hr><span class=mod_name>Walker.</span>' +
                       '<span class=func_name>' +
                       key +
                       '</span>' + sig + '\n')
        if param_delim in doc:
            params = True
            param_str = doc.split(param_delim)[-1]
        if ret_delim in doc:
            ret = True
            ret_str = doc.split(ret_delim)[-1]

        if params and ret:
            param_str = param_str.split(ret_delim)[0]

        if params:
            desc_str = doc.split(param_delim)[0]
        elif ret:
            desc_str = doc.split(ret_delim)[0]
        else:
            doc_str = doc

        if params and not ret:
            doc_str = desc_str + paramStyle(param_str, title='Parameters')
        elif not params and ret:
            doc_str = desc_str + paramStyle(ret_str, title='Returns')
        elif params and ret:
            doc_str = desc_str + paramStyle(param_str, title='Parameters') + paramStyle(ret_str, title='Returns')

        outstr = func_string + '<div class=docstring>' + doc_str + '</div>' + '\n' * 2
        maincontent += outstr
    return maincontent


def paramStyle(params, title):
    retstr = '\n<span class=paramTitle>' + title + ':</span><div class=params>'
    param_dict = {}
    p_nl = params.split('\n')[:-1]

    current_desc = ''
    # print(p_nl)
    key = p_nl[0]
    for p in p_nl[1:]:
        if '    ' in p:
            current_desc += p[4:] + '\n'
        else:
            param_dict[key] = current_desc[:]
            current_desc = ''
            key = p[:]
    param_dict[key] = current_desc

    for key in param_dict:
        try:
            var, typ = key.split(': ')
            var = '<span class=variable>' + var + '</span>: '
            typ = '<span class=type>' + typ + '</span>'
        except ValueError:
            var = ''
            typ = '<span class=type>' + key[:] + '</span>'
        desc = param_dict[key]
        retstr += var + typ + '\n<div class=variable_desc>' + desc + '</div>'

    retstr += '</div>'
    return retstr


def makeHtml(proc, fn='api.html'):
    html = ''.join(s for s in proc)
    with open(os.path.join(__location__, fn), 'w+') as f:
        f.write(html)
    return html


if __name__ == '__main__':
    fn = 'api.template.html'
    proc, marker = parseTemplate(fn)
    from core.midpoints import Walker
    docDict = getDocFromClass(Walker)
    maincontent = formatDoc(docDict)

    proc[marker['maincontent']] = maincontent

    html = makeHtml(proc)
