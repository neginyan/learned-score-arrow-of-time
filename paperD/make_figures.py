"""Paper D figures 1-6 from the stored results only (paperD/results).  Run from anywhere:  python paperD/make_figures.py
Output: paperD/figures/fig{1..6}_*.pdf and .png.  Colours: Okabe-Ito (colour-blind safe); every series also differs in marker
or line style, so the figures read in grey scale as well.  Symbols as in papers B and C."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import paperD_data as D

OI = {'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2', 'red': '#D55E00', 'purple': '#CC79A7'}
SIGC = {0.03: OI['green'], 0.1: OI['blue'], 0.3: OI['red']}             # one colour per sigma_0 throughout the paper
SIGM = {0.03: '^', 0.1: 'o', 0.3: 's'}
plt.rcParams.update({'font.size': 8.5, 'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.linewidth': 0.7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 7, 'xtick.direction': 'out', 'ytick.direction': 'out',
                     'savefig.bbox': 'tight', 'savefig.dpi': 200, 'lines.linewidth': 1.2, 'lines.markersize': 4})
FIG = os.path.join(D.ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
MARGIN = r'$(\kappa-R)/\sigma$'


def save(fig, name):
    for ext in ['pdf', 'png']: fig.savefig(os.path.join(FIG, f'{name}.{ext}'))
    plt.close(fig)


def pending(a, what):
    a.text(0.5, 0.5, f'{what}\n(result not yet available)', ha='center', va='center', transform=a.transAxes, color='0.4')


# ---------------------------------------------------------------- Fig. 1: estimator form and required accuracy
def fig1():
    """smooth random errors (correlation length sigma, 95th percentile of 20 draws): plain form (dashed, open) vs divergence /
    integration-by-parts form (solid, filled); the scale error (1 + eta) s (dotted grey) is the same for both forms."""
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    panels = [(ax[0], [o for o in D.required_2d() if o['state'] == 'gauss' and o['sig0'] in (0.1, 0.3)], '2-d pendulum, best Gaussian'),
              (ax[1], [o for o in D.required_4d() if o['eps'] == 0.3], r'4-d coupled pendulums, $\varepsilon=0.3$')]
    for a, L, title in panels:
        if not L: pending(a, title); continue
        for o in L:
            c = SIGC[o['sig0']]; mk = SIGM[o['sig0']]
            e, y = zip(*o['smooth_plain']); a.loglog(e, y, '--', marker=mk, mfc='white', color=c, label=fr"$\sigma_0={o['sig0']:g}$, plain")
            e, y = zip(*o['smooth_div']); a.loglog(e, y, '-', marker=mk, ms=6.5 if o['sig0'] == 0.1 else 3.5, color=c, label=fr"$\sigma_0={o['sig0']:g}$, divergence")
            e, y = zip(*o['scale']); a.loglog(e, y, ':', color='0.45', lw=1.0, label='scale error, both forms' if o is L[0] else None)
        a.axhline(0.1, color='0.6', lw=0.5, ls='-.'); a.axhline(0.5, color='0.6', lw=0.5, ls='-.'); a.set_xlabel(r'relative score error $\eta$'); a.legend(loc='lower right' if a is ax[0] else 'upper left', fontsize=6.2)
        a.set_title(title, fontsize=8.5)
    ax[0].set_ylabel(r'$|\Delta R|\,/\,(\kappa-R)$')
    save(fig, 'fig1_required_accuracy')


# ---------------------------------------------------------------- Fig. 2: 2-d learned vs exact
def fig2():
    T = D.table_2d(); states = ['gauss', 'fan', 'needle']; titles = {'gauss': 'best Gaussian', 'fan': 'fan', 'needle': 'needle'}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for a, st in zip(ax, states):
        for s0 in [0.03, 0.1, 0.3]:
            rows = [r for r in T if r['state'] == st and r['sig0'] == s0]
            if not rows: continue
            t = np.array([r['t'] for r in rows]); c = SIGC[s0]
            a.plot(t, [r['margin_exact']['v'] for r in rows], '-', color=c, lw=1.0, label=fr'$\sigma_0={s0:g}$ exact')
            a.errorbar(t + 0.006, [r['learned_margin']['v'] for r in rows], yerr=[r['learned_band']['v'] for r in rows], fmt=SIGM[s0], mfc='white', color=c, capsize=1.5, elinewidth=0.7,
                       label=fr'$\sigma_0={s0:g}$ learned')
        a.set_title(titles[st], fontsize=8.5); a.set_xlabel(r'$t$'); a.set_xlim(-0.03, 0.55)
    ax[0].set_ylabel(MARGIN); ax[0].legend(loc='upper left', fontsize=6.3)
    save(fig, 'fig2_learned_2d')


# ---------------------------------------------------------------- Fig. 3: 4-d learned vs exact, Gamma marks, seed band
def fig3():
    T = D.table_4d(); EX = D.exact_4d(); S = D.table_seeds()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for a, eps in zip(ax, [0.1, 0.3]):
        for s0 in [0.1, 0.3]:
            c = SIGC[s0]; ex = EX[(eps, s0)]; cv = ex['curve']
            a.plot([x['t'] for x in cv], [x['margin']['v'] for x in cv], '-', color=c, lw=1.0, label=fr'$\sigma_0={s0:g}$ exact')
            rows = [r for r in T if r['eps'] == eps and r['sig0'] == s0]
            a.errorbar(np.array([r['t'] for r in rows]) + 0.006, [r['learned_margin']['v'] for r in rows], yerr=[r['learned_band']['v'] for r in rows], fmt=SIGM[s0], mfc='white',
                       color=c, capsize=1.5, elinewidth=0.7, label=fr'$\sigma_0={s0:g}$ learned')
            a.plot([-0.045], [ex['Gamma_minus_c1_sigma']['v']], marker='_', ms=12, mew=2, color=c, clip_on=False)
            if eps == 0.3 and s0 == 0.1:
                a.fill_between([r['t'] for r in S], [r['min'] for r in S], [r['max'] for r in S], color=c, alpha=0.18, lw=0, label='3 seeds (range)')
        a.plot([-0.045], [EX[(eps, 0.1)]['Gamma']['v']], marker='*', ms=9, color='k', clip_on=False, label=r'$\Gamma$ (paper C)')
        a.set_title(fr'coupled pendulums, $\varepsilon={eps:g}$  ($\kappa={EX[(eps, 0.1)]["kappa"]["v"]:.4f}$)', fontsize=8.5); a.set_xlabel(r'$t$'); a.set_xlim(-0.06, 0.55)
    ax[0].set_ylabel(MARGIN); ax[1].legend(loc='lower left', fontsize=6.3, ncol=2)
    ax[0].text(0.03, 0.95, r'bars at $t<0$: $\Gamma-c_1\sigma_0$', transform=ax[0].transAxes, fontsize=6.5, color='0.3')
    save(fig, 'fig3_learned_4d')


# ---------------------------------------------------------------- Fig. 4: the band kappa - c sigma < r < kappa
def fig4():
    b = D.band(); fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7)); a = ax[0]
    sty = {'0.9541': (OI['red'], 'o', '-'), '1': (OI['blue'], 's', '--'), '1.1': (OI['green'], '^', ':')}
    for r, v in b['sde'].items():
        c, m, ls = sty[r]; a.plot([x['t'] for x in v], [100 * x['varV_rel'] for x in v], ls, marker=m, color=c, label=fr'$r={float(r):g}$' + (r' $<\kappa$ (in the band)' if float(r) < 1 else ''))
    a.axhline(0, color='0.5', lw=0.6); a.set_xlabel(r'$t$'); a.set_ylabel(r'error of $\mathrm{Var}_V$ (%)'); a.legend(loc='upper left')
    a.set_title('(I) forward SDE with the physical drift', fontsize=8.5)
    a = ax[1]; E = b['entropy']
    a.plot([x['t'] for x in E], [x['dS_exact'] for x in E], '-', color=OI['red'], label=r'exact $S(t)-S(0)$')
    a.errorbar([x['t'] for x in E], [x['dS_knn_mean'] for x in E], yerr=[2 * x['dS_knn_se'] for x in E], fmt='o', mfc='white', color='k', capsize=1.5, label='k-NN (2 s.e.)')
    a.set_xlabel(r'$t$'); a.set_ylabel(r'$S(t)-S(0)$'); a.legend(loc='upper left')
    a2 = a.twinx(); a2.plot([x['t'] for x in E], [x['R'] for x in E], '--', color=OI['blue'], lw=1); a2.axhline(float(b['r_band']), color=OI['red'], ls=':', lw=0.8)
    a2.axhline(1.0, color='k', ls=':', lw=0.8); a2.set_ylim(0.0, 1.1); a2.set_ylabel(r'$R(t)$ (dashed), $r$, $\kappa$ (dotted)', fontsize=7); a2.spines['right'].set_visible(True)
    a.set_title(fr"(II) coarse-grained entropy at $r={b['r_band']:g}<\kappa$", fontsize=8.5)
    save(fig, 'fig4_band')


# ---------------------------------------------------------------- Fig. 5: knee of (I)
def fig5():
    """top: forward SDE with the physical drift (excess of the Var_V error over its plateau at r >= kappa);
    bottom: reverse SDE minus probability-flow ODE (Var_V error), exact (filled, solid) and learned (open, dashed) score."""
    K = D.table_knee(); FW = D.table_forward(); fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    for j, dim in enumerate([2, 4]):
        a = ax[0, j]; rows = [r for r in FW if r['dim'] == dim]
        if not rows: pending(a, f'{dim}-d forward SDE')
        for r in sorted(rows, key=lambda r: r['sig0']):
            c = SIGC[r['sig0']]; a.plot(r['r_over_kappa'], np.array(r['varV_rel']) - r['plateau'], '-', marker=SIGM[r['sig0']], color=c,
                                        label=fr"$\sigma_0={r['sig0']:g}$ (knee {r['knee_excess']:.3f}; plateau {100 * r['plateau']:+.1f}%)")
        a.set_title(('2-d pendulum' if dim == 2 else r'4-d coupled pendulums, $\varepsilon=0.3$') + ': forward SDE', fontsize=8.5)
        a = ax[1, j]; rows = [r for r in K if r['dim'] == dim and r['main']]
        if not rows: pending(a, f'{dim}-d reverse samplers')
        for r in sorted(rows, key=lambda r: (r['mode'], r['sig0'])):
            c = SIGC[r['sig0']]; ex = r['mode'] == 'exact'
            a.plot(r['r_over_kappa'], r['sde_minus_ode'], '-' if ex else '--', marker=SIGM[r['sig0']], mfc=c if ex else 'white', color=c,
                   label=fr"$\sigma_0={r['sig0']:g}$, {r['mode']} (knee {r['knee']['v']:.3f})")
        a.set_title('reverse SDE $-$ ODE', fontsize=8.5); a.set_xlabel(r'$r/\kappa$')
    for a in ax.ravel():
        a.axvline(1, color='k', lw=0.7); a.axhline(0, color='0.5', lw=0.6); a.legend(fontsize=5.8, loc='upper right' if a in ax[0] else 'lower right')
    ax[0, 0].set_ylabel(r'$\mathrm{Var}_V$ error $-$ plateau'); ax[1, 0].set_ylabel(r'$\mathrm{Var}_V$ error: SDE $-$ ODE')
    save(fig, 'fig5_knee')


# ---------------------------------------------------------------- Fig. 6: entropy measured directly
def fig6():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7)); a = ax[0]
    E2 = D.entropy_2d(); k = 0
    for st, mk in [('gauss', 'o'), ('fan', 'D')]:
        for s0 in [0.1, 0.3]:
            for kappa_run in [True, False]:
                rows = [r for r in E2 if r['state'] == st and r['sig0'] == s0 and ((r['r'] == 1.0) == kappa_run)]
                if not rows: continue
                x = np.array([r['t0'] + 0.05 for r in rows]) + 0.006 * (k % 6 - 2.5); k += 1
                a.errorbar(x, [r['dS_knn'] / r['dS_exact'] for r in rows], yerr=[2 * r['dS_knn_se'] / r['dS_exact'] for r in rows], fmt=mk, color=SIGC[s0], mfc=SIGC[s0] if kappa_run else 'white',
                           capsize=1.2, elinewidth=0.6, label=fr"{st}, $\sigma_0={s0:g}$, " + (r'$r=\kappa$' if kappa_run else fr"$r={rows[0]['r']:g}$"))
    a.axhline(1, color='k', lw=0.7); a.set_xlabel(r'$t$ (interval midpoint)'); a.set_ylabel(r'$\Delta S_{\rm k\text{-}NN}/\Delta S_{\rm exact}$'); a.legend(fontsize=5.6, ncol=2, loc='lower center')
    a.set_title('2-d pendulum (k-NN, 2 s.e.)', fontsize=8.5); a.set_ylim(0.8, 1.15)
    a = ax[1]; E4 = D.entropy_4d(); x = np.array([r['t0'] + 0.05 for r in E4])
    a.errorbar(x - 0.01, [r['dS_knn'] for r in E4], yerr=[2 * r['dS_knn_se'] for r in E4], fmt='o', mfc='white', color='k', capsize=1.2, label='k-NN, raw')
    a.errorbar(x, [r['dS_knn_white'] for r in E4], yerr=[2 * r['dS_knn_white_se'] for r in E4], fmt='^', color=OI['blue'], capsize=1.2, label='k-NN, whitened')
    a.plot(x + 0.01, [r['dS_exact_pred'] for r in E4], 's', mfc='none', ms=6, color=OI['red'], label=r'exact $\int (r\sigma^2{\rm tr}F+\Sigma)\,dt$')
    a.plot(x + 0.02, [np.mean(list(r['dS_learned'].values())) for r in E4], 'D', ms=3, color=OI['orange'], label='learned score (4 nets, mean)')
    a.set_xlabel(r'$t$ (interval midpoint)'); a.set_ylabel(r'$\Delta S$ over $\Delta t=0.1$'); a.legend(fontsize=6)
    a.set_title(r'4-d, $\varepsilon=0.3$, $\sigma_0=0.1$, $r=\kappa$', fontsize=8.5)
    save(fig, 'fig6_entropy')


if __name__ == '__main__':
    for f in [fig1, fig2, fig3, fig4, fig5, fig6]: f(); print('done', f.__name__)
