import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.patheffects as path_effects


class FixedKeysDict(dict):
    def __init__(self, *args, **kwargs):
        super(FixedKeysDict, self).__init__(*args, **kwargs)
        self._frozen_keys = set(self.keys())  # Capture initial keys

    def __setitem__(self, key, value):
        if key not in self._frozen_keys:
            raise KeyError(f"Key '{key}' cannot be added.")
        super(FixedKeysDict, self).__setitem__(key, value)

    def __delitem__(self, key):
        raise KeyError(f"Key '{key}' cannot be deleted.")

    def pop(self, key, default=None):
        raise KeyError(f"Key '{key}' cannot be popped.")

    def popitem(self):
        raise KeyError("Cannot pop item from FixedKeysDict.")

    def clear(self):
        raise KeyError("Cannot clear FixedKeysDict.")

    def update(self, *args, **kwargs):
        for key in dict(*args, **kwargs):
            if key not in self._frozen_keys:
                raise KeyError(f"Key '{key}' cannot be added.")
        super(FixedKeysDict, self).update(*args, **kwargs)


class SetDict(object):
    # set_dict(self, dTarget, dSource, bPreserve)
    # Takes a target dictionary, and enters values from the source dictionary, overwriting or not, as asked.
    # For example,
    #    dT={'a':1, 'b':2}
    #    dS={'a':0, 'c':0}
    #    Set(dT, dS, True)
    #    dT is {'a':1, 'b':2, 'c':0}
    #
    #    dT={'a':1, 'b':2}
    #    dS={'a':0, 'c':0}
    #    Set(dT, dS, False)
    #    dT is {'a':0, 'b':2, 'c':0}
    #
    def set_dict(self, dTarget, dSource, bPreserve):
        for k, v in dSource.items():
            bKeyExists = (k in dTarget)
            if (not bKeyExists) and type(v) == type({}):
                dTarget[k] = {}
            if ((not bKeyExists) or not bPreserve) and (type(v) != type({})):
                dTarget[k] = v

            if type(v) == type({}):
                self.set_dict(dTarget[k], v, bPreserve)

    def set_dict_with_preserve(self, dTarget, dSource):
        self.set_dict(dTarget, dSource, True)

    def set_dict_with_overwrite(self, dTarget, dSource):
        self.set_dict(dTarget, dSource, False)


class LatexStates:
    """ Holds LaTex format corresponding to set symbolic variables.
    """

    def __init__(self, dict=None):
        self.dict = {'v_para': r'$v_{\parallel}$',
                     'v_perp': r'$v_{\perp}$',
                     'phi': r'$\phi$',
                     'phidot': r'$\dot{\phi}$',
                     'phi_dot': r'$\dot{\phi}$',
                     'phiddot': r'$\ddot{\phi}$',
                     'w': r'$w$',
                     'zeta': r'$\zeta$',
                     'I': r'$I$',
                     'm': r'$m$',
                     'C_para': r'$C_{\parallel}$',
                     'C_perp': r'$C_{\perp}$',
                     'C_phi': r'$C_{\phi}$',
                     'km1': r'$k_{m_1}$',
                     'km2': r'$k_{m_2}$',
                     'km3': r'$k_{m_3}$',
                     'km4': r'$k_{m_4}$',
                     'd': r'$d$',
                     'psi': r'$\psi$',
                     'gamma': r'$\gamma$',
                     'alpha': r'$\alpha$',
                     'of': r'$\frac{g}{d}$',
                     'gdot': r'$\dot{g}$',
                     'v_para_dot': r'$\dot{v_{\parallel}}$',
                     'v_perp_dot': r'$\dot{v_{\perp}}$',
                     'v_para_dot_ratio': r'$\frac{\Delta v_{\parallel}}{v_{\parallel}}$',
                     'x':  r'$x$',
                     'y':  r'$y$',
                     'v_x': r'$v_{x}$',
                     'v_y': r'$v_{y}$',
                     'v_z': r'$v_{z}$',
                     'w_x': r'$w_{x}$',
                     'w_y': r'$w_{y}$',
                     'w_z': r'$w_{z}$',
                     'a_x': r'$a_{x}$',
                     'a_y': r'$a_{y}$',
                     'vx': r'$v_x$',
                     'vy': r'$v_y$',
                     'vz': r'$v_z$',
                     'wx': r'$w_x$',
                     'wy': r'$w_y$',
                     'wz': r'$w_z$',
                     'ax': r'$ax$',
                     'ay': r'$ay$',
                     'beta': r'$\beta',
                     'thetadot': r'$\dot{\theta}$',
                     'theta_dot': r'$\dot{\theta}$',
                     'psidot': r'$\dot{\psi}$',
                     'psi_dot': r'$\dot{\psi}$',
                     'theta': r'$\theta$',
                     'Yaw': r'$\psi$',
                     'R': r'$\phi$',
                     'P': r'$\theta$',
                     'dYaw': r'$\dot{\psi}$',
                     'dP': r'$\dot{\theta}$',
                     'dR': r'$\dot{\phi}$',
                     'acc_x': r'$\dot{v}x$',
                     'acc_y': r'$\dot{v}y$',
                     'acc_z': r'$\dot{v}z$',
                     'Psi': r'$\Psi$',
                     'Ix': r'$I_x$',
                     'Iy': r'$I_y$',
                     'Iz': r'$I_z$',
                     'Jr': r'$J_r$',
                     'Dl': r'$D_l$',
                     'Dr': r'$D_r$',
                     }

        if dict is not None:
            SetDict().set_dict_with_overwrite(self.dict, dict)

    def convert_to_latex(self, list_of_strings, remove_dollar_signs=False):
        """ Loop through list of strings and if any match the dict, then swap in LaTex symbol.
        """

        if isinstance(list_of_strings, str):  # if single string is given instead of list
            list_of_strings = [list_of_strings]
            string_flag = True
        else:
            string_flag = False

        list_of_strings = list_of_strings.copy()
        for n, s in enumerate(list_of_strings):  # each string in list
            for k in self.dict.keys():  # check each key in Latex dict
                if s == k:  # string contains key
                    # print(s, ',', self.dict[k])
                    list_of_strings[n] = self.dict[k]  # replace string with LaTex
                    if remove_dollar_signs:
                        list_of_strings[n] = list_of_strings[n].replace('$', '')

        if string_flag:
            list_of_strings = list_of_strings[0]

        return list_of_strings


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline(x, y, z, ax=None, cmap=plt.get_cmap('copper'), norm=None, linewidth=1.5, alpha=1.0):
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    # Set normalization
    if norm is None:
        norm = plt.Normalize(np.min(z), np.max(z))

    print(norm)

    # Make segments
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha,
                              path_effects=[path_effects.Stroke(capstyle="round")])

    # Plot
    if ax is None:
        ax = plt.gca()

    ax.add_collection(lc)

    return lc

def plot_heatmap_log_timeseries(data, ax=None, log_ticks=None, data_labels=None,
                                cmap='inferno_r', y_label=None,
                                aspect=0.25, interpolation=False):
    """ Plot log-scale time-series as heatmap.
    """

    n_label = data.shape[1]

    # Set ticks
    if log_ticks is None:
        log_tick_low = int(np.floor(np.log10(np.min(data))))
        log_tick_high = int(np.ceil(np.log10(np.max(data))))
    else:
        log_tick_low = log_ticks[0]
        log_tick_high = log_ticks[1]

    log_ticks = np.logspace(log_tick_low, log_tick_high, log_tick_high - log_tick_low + 1)

    # Set color normalization
    cnorm = mpl.colors.LogNorm(10 ** log_tick_low, 10 ** log_tick_high)

    # Set labels
    if data_labels is None:
        data_labels = np.arange(0, n_label).tolist()
        data_labels = [str(x) for x in data_labels]

    # Make figure/axis
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5 * 1, 4 * 1), dpi=150)
    else:
        # ax = plt.gca()
        fig = plt.gcf()

    # Plot heatmap
    if interpolation:
        data = 10**scipy.ndimage.zoom(np.log10(data), (interpolation, 1), order=1)
        aspect = aspect / interpolation

    ax.imshow(data, norm=cnorm, aspect=aspect, cmap=cmap, interpolation='none')

    # Set axis properties
    ax.grid(True, axis='x')
    ax.tick_params(axis='both', which='both', labelsize=6, top=False, labeltop=True, bottom=False, labelbottom=False,
                   color='gray')

    # Set x-ticks
    LatexConverter = LatexStates()
    data_labels_latex = LatexConverter.convert_to_latex(data_labels)
    ax.set_xticks(np.arange(0, len(data_labels)) - 0.5)
    ax.set_xticklabels(data_labels_latex)

    # Set labels
    ax.set_ylabel('time steps', fontsize=7, fontweight='bold')
    ax.set_xlabel('states', fontsize=7, fontweight='bold')
    ax.xaxis.set_label_position('top')

    # Set x-ticks
    xticks = ax.get_xticklabels()
    for tick in xticks:
        tick.set_ha('left')
        tick.set_va('center')
    #     tick.set_rotation(0)
    #     tick.set_transform(tick.get_transform() + transforms.ScaledTranslation(6 / 72, 0, ax.figure.dpi_scale_trans))

    # Colorbar
    if y_label is None:
        y_label = 'values'

    cax = ax.inset_axes((1.03, 0.0, 0.04, 1.0))
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap), cax=cax, ticks=log_ticks)
    cbar.set_label(y_label, rotation=270, fontsize=7, labelpad=8)
    cbar.ax.tick_params(labelsize=6)

    ax.spines[['bottom', 'top', 'left', 'right']].set_color('gray')

    return cnorm, cmap, log_ticks