import time

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from Prepare_spectrum import get_thold
from Line_analysis import EW_analysis
from IO_functions import *
plt.ion()
# #####################################################
# Ploting module
# #####################################################
class Plot_Module(object):

    def __init__(self, spec, ax,  cfile, a_line, gggf, gggf_infm, strong_lines, out_file, logfile):
        self.killme = False
        self.ax = ax
        self.strong_lines = strong_lines
        self.a_line = a_line
        self.spec = spec
        self.cfile = cfile
        self.gggf = gggf
        self.log = logfile
        self.out = out_file
        self.rtab = [0]
        self.gggf_infm = gggf_infm
        self.rtab = []
        self.lr = []

        thold = get_thold(gggf, self.cfile.getfloat('Lines', 'r_lvl'))

        x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        self.ax[0].set_title('%s %s' % (a_line[0], a_line[1]))
        self.ax[0].xaxis.set_major_formatter(x_formatter)
        self.ax[0].axhline(1.0, color='g', label='continuum')
        self.ax[0].set_xlabel('Wavelenght')
        self.ax[0].plot(self.spec[:, 0], self.spec[:, 1], 'o', color='k', label='spectrum')
        self.ax[0].set_ylim(min(self.spec[:, 1]) - 0.1, 1. + 0.4 * (1. + 0.1 - min(self.spec[:, 1])))

        self.ax[1].axhline(0)
        self.ax[1].plot(self.spec[:, 0], self.gggf, 'b', label='3rd derivative')
        self.ax[1].set_ylim(np.min(self.gggf), np.max(self.gggf))
        self.ax[1].set_xlim(a_line[0] - cfile.getfloat('Spectrum', 'off'),
                            a_line[0] + cfile.getfloat('Spectrum', 'off'))
        self.ax[1].plot(self.gggf_infm, np.zeros_like(self.gggf_infm),
                        'o', color='b', label='flex points + -> -')
        self.ax[1].axhline(-thold, c='r')
        self.ax[1].axhline(thold, c='r')

        self.lstyle = np.array([['mg', 'c', '-', 'multi Gauss', 4],
                                ['g', 'y', '-', 'Gauss', 2],
                                ['v', 'm', '-', 'Voigt', 1]])

        plt.gcf().canvas.mpl_connect('key_press_event', self.ontype)
        plt.gcf().canvas.mpl_connect('button_press_event', self.onclick)

    def make_plot(self):
        for lbl in self.r_tab:
            fit_style = np.squeeze(self.lstyle[np.where(self.lstyle[:, 0] == lbl)])
            self.ax[0].plot(self.spec[:, 0], 1.0 - self.r_tab[lbl][0],
                            color=fit_style[1],
                            ls=fit_style[2],
                            label=fit_style[3],
                            zorder=fit_style[4])

        x01 = self.r_tab['g'][1][1]
        s1 = self.r_tab['g'][1][2]

        self.ax[0].axvspan(x01 - self.cfile.getfloat('Lines', 'w_factor') * s1,
                           x01 + self.cfile.getfloat('Lines', 'w_factor') * s1, color='g',
                           alpha=0.25, label='eval. area')
        self.ax[0].legend(loc=2, numpoints=1, fontsize='10', ncol=4)

        self.ax[0].axvline(x01, color='r', lw=1.5, label='line')
        for oline in self.r_tab['mg'][1][:, 0]:
            self.ax[0].axvline(oline, c='r', zorder=1, label='strong lines')

        self.ax[1].plot(self.strong_lines, np.zeros_like(self.strong_lines),
                        'o', color='r', label='strong lines')
        self.ax[1].legend(loc=3, numpoints=1, fontsize='10', ncol=3)

    def clean_plot(self):
        plt.sca(self.ax[0])
        for artist in plt.gca().get_children():
            if hasattr(artist, 'get_label') and (
                    artist.get_label() in self.lstyle[:, 3] or
                    artist.get_label() == 'line' or
                    artist.get_label() == 'eval. area' or
                    artist.get_label() == 'strong lines' or
                    artist.get_label() == 'line_pnt'):
                artist.remove()

        plt.sca(self.ax[1])
        for artist in plt.gca().get_children():
            if hasattr(artist, 'get_label') and (
                    artist.get_label() == 'strong lines' or
                    artist.get_label() == 'line_pnt'):
                artist.remove()

    def onclick(self, event):
        toolbar = plt.get_current_fig_manager().toolbar
        if event.button == 1 and toolbar.mode == '':
            ind = np.abs((self.gggf_infm - event.xdata)).argmin()
            self.ax[1].plot(self.gggf_infm[ind], 1.0, 'o', color='r', mec='b', picker=5,
                            label='line_pnt')
            self.ax[0].axvline(self.gggf_infm[ind], c='r', ls=':', zorder=1,
                               label='line_pnt')
            self.strong_lines.append(self.gggf_infm[ind])
            print_and_log(self.log, ['Adding ', self.gggf_infm[ind], 'to fit'])

        elif event.button == 3 and toolbar.mode == '':
            ind = np.squeeze(np.where(
                np.abs(self.strong_lines - event.xdata) < 0.05))

            if ind:
                self.ax[1].plot(self.strong_lines[ind], 1.0, 'o',
                                color='b', mec='r',
                                picker=5, label='line_pnt')
                self.ax[0].axvline(self.strong_lines[ind], c='b', ls=':', zorder=1,
                                   label='line_pnt')
                self.strong_lines.pop(ind)
                print_and_log(self.log, ['Removing ', self.strong_lines[ind], 'from fit'])
            else:
                print 'Click closer to chosen point'

        plt.draw()

    def ontype(self, event):
        if event.key == 'enter':
            self.clean_plot()
            self.update_plot()
        elif event.key == 'q':
            exit()
        else:
            self.killme = True

    def update_plot(self):
        self.strong_lines = sorted(list(set(self.strong_lines)))
        self.r_tab, self.lr = EW_analysis(self.a_line[0], self.spec, self.strong_lines,
                                          self.cfile, self.log)

        self.strong_lines = list(self.r_tab['mg'][1][:, 0])  # fitted strong lines
        self.make_plot()
        plt.draw()
        moog = moog_entry(self.a_line, self.lr[3], self.lr[4])
        print_and_log(self.log, ['\nCurrent moog entry:\n', moog])

    def run(self):
        plt.show()
        self.update_plot()
        print '########################################'
        print 'Click on plot to edit strong lines list:'
        print '(left button - add lines, right - remove lines).'
        print 'Then hit enter to redo the plot'
        print 'q - to quit, any other key to move to the next line'
        print '#######################################'
        while not self.killme:
            keyboard_click = False
            while not keyboard_click:
                keyboard_click = plt.waitforbuttonpress()
            time.sleep(0.1)

        plt.close()
        print 'Writing to output file...'
        moog_output(self.out, self.log, self.a_line, self.lr)
