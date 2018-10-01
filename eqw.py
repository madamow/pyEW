import matplotlib.pyplot as plt
from IO_functions import *
from Prepare_spectrum import *
from Line_analysis import *
from Plotting import *
import sys
plt.switch_backend('qt5Agg')

######################################################
######################################################

# INPUT : CONFIG FILE
cfg_name = sys.argv[1]  # Get config file name
cfg = read_config_file(cfg_name)  # Read config file

# INPUT :  LINE LIST
llist = get_lab_data(cfg)  # tab with laboratory data for lines to analyze

# INPUT : file list
flist = get_flist(cfg)

for file_name in flist:
    if file_name == 'end_iter':
        exit()
    # Create output files
    log = create_log(file_name)
    out_file = create_out(file_name, cfg)

    mgtab = np.empty((0, 5), dtype=float)
    
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
        lin1, dx = do_linear(d)

        # Correct continuum around chosen line
        try:
            lin = local_continuum(lin1, rejt, cfg.getfloat('Spectrum', 't_sig'), log)
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
