"""shared figure helpers for the CIFAR leg."""


def setup_mpl():
    """
    matplotlib defaults for the paper figures: sans-serif with a Helvetica-first
    fallback and PDF/PS text kept editable (fonttype 42). sets the global rcParams
    and returns the pyplot module.
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'Helvetica', 'Nimbus Sans', 'Arial', 'Liberation Sans', 'DejaVu Sans'
    ]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    return plt
