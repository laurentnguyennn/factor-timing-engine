"""
Visualization module — standardised plotting utilities.
All plots saved at 300 DPI with consistent colour scheme.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, COLORS


def setup_style():
    """Apply consistent matplotlib style across all notebooks."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 100,
    })
    sns.set_palette('Set2')


def save_fig(fig, name, dpi=300):
    """Standardised figure saving."""
    path = FIGURES_DIR / f'{name}.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')
    return path


def plot_cumulative_returns(returns_df, title='Cumulative Returns', figsize=(14, 6)):
    """Plot cumulative return lines for multiple series."""
    fig, ax = plt.subplots(figsize=figsize)
    cum = (1 + returns_df).cumprod()
    cum.plot(ax=ax, linewidth=1.2)
    ax.set_ylabel('Cumulative Return (base=1)')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper left')
    return fig


def plot_regime_overlay(prices, regime_labels, title=''):
    """Price series with colour-coded regime background shading."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(prices.index, prices, color='black', linewidth=0.8)

    regime_colors = {
        'Expansion': COLORS['expansion'],
        'Slowdown': COLORS['slowdown'],
        'Crisis': COLORS['crisis'],
        'QE Era': COLORS['expansion'],
        'Rate Hike': COLORS['slowdown'],
        'Late Cycle': COLORS['slowdown'],
        'COVID Crash': COLORS['crisis'],
        'Zero-Rate Recovery': COLORS['expansion'],
        'Inflation/Rate Shock': COLORS['crisis'],
        'AI Boom': COLORS['expansion'],
        'Tariff/Geopolitical': COLORS['slowdown'],
    }

    for i in range(len(prices) - 1):
        label = regime_labels.iloc[i] if hasattr(regime_labels, 'iloc') else regime_labels[i]
        color = regime_colors.get(str(label), '#cccccc')
        ax.axvspan(prices.index[i], prices.index[i + 1],
                   alpha=0.15, color=color, linewidth=0)

    ax.set_ylabel('Price')
    ax.set_title(title, fontweight='bold')
    return fig


def plot_correlation_heatmap(corr_matrix, title='Correlation Matrix', figsize=(8, 8)):
    """Plot annotated correlation heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, ax=ax)
    ax.set_title(title, fontweight='bold')
    return fig


def plot_drawdown(nav_series, title='Drawdown', figsize=(14, 5)):
    """Plot drawdown area chart from NAV series."""
    fig, ax = plt.subplots(figsize=figsize)
    dd = (nav_series / nav_series.cummax()) - 1
    ax.fill_between(dd.index, dd, alpha=0.4, color=COLORS['crisis'])
    ax.plot(dd.index, dd, color=COLORS['crisis'], linewidth=0.8)
    ax.set_ylabel('Drawdown')
    ax.set_title(title, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    return fig


def plot_regime_with_probabilities(prices, regime_probs, regime_labels,
                                   title='', figsize=(14, 8)):
    """
    Dual-panel signature plot: price + regime shading (top), stacked probabilities (bottom).
    This is the publication-quality regime overlay from blueprint §G.1.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)

    # Top panel: price with regime background shading
    ax1.plot(prices.index, prices, color='black', linewidth=0.8)

    regime_colors = {
        'Expansion': COLORS['expansion'],
        'Slowdown': COLORS['slowdown'],
        'Crisis': COLORS['crisis'],
    }

    for i in range(len(prices) - 1):
        label = regime_labels.iloc[i] if hasattr(regime_labels, 'iloc') else regime_labels[i]
        color = regime_colors.get(str(label), '#cccccc')
        ax1.axvspan(prices.index[i], prices.index[i + 1],
                    alpha=0.15, color=color, linewidth=0)

    ax1.set_ylabel('Price / Index Level')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: regime probabilities stacked area
    prob_cols = [c for c in regime_probs.columns if c.startswith('p_')]
    if not prob_cols:
        prob_cols = regime_probs.select_dtypes(include=[np.number]).columns.tolist()

    colors_list = [COLORS.get('expansion', '#2ecc71'),
                   COLORS.get('slowdown', '#f39c12'),
                   COLORS.get('crisis', '#e74c3c')][:len(prob_cols)]

    ax2.stackplot(regime_probs.index,
                  *[regime_probs[c] for c in prob_cols],
                  colors=colors_list, alpha=0.7, labels=prob_cols)
    ax2.set_ylabel('Regime Probability')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_regime_heatmap(regime_stats, metric='ann_return', figsize=(8, 5)):
    """
    Heatmap showing factor performance by regime — the visual core of the thesis.
    """
    fig, ax = plt.subplots(figsize=figsize)
    data = regime_stats.pivot(index='factor', columns='regime', values=metric)

    col_order = ['Expansion', 'Slowdown', 'Crisis']
    data = data[[c for c in col_order if c in data.columns]]

    im = ax.imshow(data.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f'{data.values[i, j]:.2%}'
            color = 'white' if abs(data.values[i, j]) > 0.15 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=11)

    ax.set_title(f'Factor {metric.replace("_", " ").title()} by Regime', fontsize=13)
    fig.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    return fig


def plot_weight_evolution(weights_df, title='Portfolio Weight Evolution',
                          figsize=(14, 6)):
    """Stacked area chart of portfolio weights over time."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(weights_df.index,
                 *[weights_df[c] for c in weights_df.columns],
                 labels=weights_df.columns, alpha=0.8)
    ax.set_ylabel('Portfolio Weight')
    ax.set_ylim(0, 1)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    return fig
