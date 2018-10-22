#!/usr/bin/env python

import sys
import os
import matplotlib.pyplot as plt
plt.switch_backend('qt5Agg')

path = os.path.dirname(os.path.realpath(__file__))  # directory of eqw.py
# location of pyEW functions:
sys.path.append("%s/lib/" % path)

from IO_functions import *
from Prepare_spectrum import *
from Line_analysis import *
from Plotting import *


######################################################
######################################################
def get_equvalent_widths(llist, file_name, cfg):
    # Create output files
    log = create_log(file_name)
    out_file = create_out(file_name, cfg)

    # Here calculations start
    # Load observational spectrum
    spec_obs = np.loadtxt(file_name)

    ###################################################
    # Get rejt and snr
    rejt, snr = get_snr(spec_obs, cfg, log)
    out_file.write(file_name + ' snr = ' + '%4.0f' % snr + '\n')

    #####################################################
    # Lets analyze every single line
    for a_line in llist:
        print_and_log(log, ['\n#####\n', a_line[0], a_line[1]])

        d = spec_obs[np.where((spec_obs[:, 0] > a_line[0] - cfg.getfloat('Spectrum', 'off')) &
                              (spec_obs[:, 0] < a_line[0] + cfg.getfloat('Spectrum', 'off')))]

        if d.shape[0] == 0 or d[:, 0].max() < a_line[0] or d[:, 0].min() > a_line[0]:
            print_and_log(log, ['Nothing to do for line', a_line[0]])
            continue

        # Make spectrum linear
        # (default assumption - spectrum is not linear)
        lin, dx = do_linear(d)

        if cfg.getboolean('Spectrum', 'local_cont'):
        # Correct continuum around chosen line
            try:
                lin = local_continuum(lin, rejt, cfg.getfloat('Spectrum', 't_sig'), log)
            except:
                print_and_log(log, ['Unable to correct continuum'])
                continue

        # Find flex point
        gggf_infm, gggf_infp, gggf = find_flex_points(lin, dx, cfg)

        # If there are no inflection points, go to next line on list
        if gggf_infm.size == 0 or gggf_infp.size == 0:
            continue

        # Identify strong lines
        strong_lines = find_strong_lines(lin[:, 0], a_line[0], gggf, gggf_infm, cfg, log)

        if len(strong_lines) == 0:
            continue

        print_and_log(log, ['I see', len(strong_lines), 'line(s) in this range'])

        ################################################################################
        show_lines = np.array(cfg.get('Lines', 'show_lines').split(' '), dtype=float)

        if a_line[0] in show_lines and not cfg.getboolean('Lines', 'plot_flag'):
            plot_line = True
        elif a_line[0] not in show_lines and not cfg.getboolean('Lines', 'plot_flag'):
            plot_line = False
        else:
            plot_line = True

        if not plot_line:
            r_tab, lr = EW_analysis(a_line[0], lin, strong_lines, cfg, log)
            moog_output(out_file, log, a_line, lr)

        # Ploting module -
        else:
            fig, ax = plt.subplots(2, sharex=True)

            p = Plot_Module(lin, ax, cfg, a_line, gggf, gggf_infm, strong_lines, out_file, log)
            p.run()


def main():
    # INPUT : CONFIG FILE
    config_file_name = sys.argv[1]  # Get config file name
    config = read_config_file(config_file_name)  # Read config file

    # INPUT :  LINE LIST
    line_list = get_lab_data(config)  # tab with laboratory data for lines to analyze

    # INPUT : file list
    flist = get_flist(config)

    for fname in flist:
        if fname == 'end_iter':
            exit()
        get_equvalent_widths(line_list, fname, config)
        exit()


if __name__ == "__main__":

    main()
