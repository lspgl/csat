import sys


class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[35m'
    YEL = '\033[33m'
    RED = '\033[36m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def configread(cfg_path):
    cfg = {}
    f = open(cfg_path, 'r')

    int_keys = ['nex', 'tsteps', 'n_stat', 'rseed']
    str_keys = ['name']

    for line in f:
        entry = line.strip().split('#')[0].replace(' ', '').split(':')
        if len(entry) > 1:
            if entry[1] == '' or entry[1] == 'None':
                entry[1] = None
            elif entry[1] == 'True':
                entry[1] = True
            elif entry[1] == 'False':
                entry[1] = False
            elif entry[0] in int_keys:
                entry[1] = int(entry[1])
            elif entry[0] in str_keys:
                entry[1] = str(entry[1])
            else:
                entry[1] = float(entry[1])

            cfg[entry[0]] = entry[1]

    if cfg['stat_FLAG']:
        cfg['mp_FLAG'] = False

    if cfg['nex'] is not None:
        cfg['density'] = False
        cfg['laser'] = False
        cfg['rho'] = 0
    elif cfg['rho'] is not None:
        cfg['density'] = True
        if cfg['fwhm'] is not None:
            cfg['laser'] = True
        else:
            cfg['laser'] = False
    else:
        print(colors.WARNING + 'Set either RHO or NEX in config' + colors.ENDC)
        sys.exit()
    if cfg['cw_FLAG'] and (cfg['fwhm'] is None):
        print(colors.WARNING + 'Continuous Wave operation is only configured for gaussian beams' + colors.ENDC)
        sys.exit()
    if cfg['entropy_FLAG'] and (cfg['fwhm'] is None):
        print(colors.WARNING + 'Entropic mapping only configured for gaussian beams' + colors.ENDC)
        sys.exit()

    return cfg
