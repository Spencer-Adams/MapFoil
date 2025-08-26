import matplotlib.pyplot as plt # type: ignore
from matplotlib.patches import Polygon # type: ignore

def read_su2(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    nodes = []
    elements = []
    boundaries = {} # Dictionary to hold boundary edges with tags
    reading_nodes = False
    reading_elements = False
    n_nodes = n_elements = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('NPOIN='):
            n_nodes = int(line.split('=')[1])
            reading_nodes = True
            i += 1
            for _ in range(n_nodes):
                parts = lines[i].strip().split()
                x, y = float(parts[0]), float(parts[1])
                nodes.append((x, y))
                i += 1
            continue
        elif line.startswith('NELEM='):
            n_elements = int(line.split('=')[1])
            reading_elements = True
            i += 1
            for _ in range(n_elements):
                parts = list(map(int, lines[i].strip().split()))
                etype = parts[0]
                if etype == 5:  # triangle
                    elements.append(parts[1:4])
                elif etype == 9:  # quad
                    elements.append(parts[1:5])
                i += 1
            continue
        elif line.startswith('NMARK='):
            n_mark = int(line.split('=')[1])
            i += 1
            for _ in range(n_mark):
                tag_line = lines[i].strip()
                tag = tag_line.split('=')[1].strip()
                i += 1
                n_edges = int(lines[i].strip().split('=')[1])
                i += 1
                edges = []
                for _ in range(n_edges):
                    parts = list(map(int, lines[i].strip().split()))
                    edges.append((parts[1], parts[2]))
                    i += 1
                boundaries[tag] = edges
            continue
        i += 1
    return nodes, elements, boundaries

def plot_su2_mesh(nodes, elements, boundaries):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#F5F5DC')  # light earth background
    for elem in elements:
        poly = Polygon([nodes[i] for i in elem], closed=True, edgecolor='#8B5C2D', facecolor='#A3C585', alpha=0.7)
        ax.add_patch(poly)

    xs, ys = zip(*nodes)
    ax.plot(xs, ys, 'o', markersize=1, color='#4F97A3', alpha=0.5)

    # Plot boundaries in different colors
    boundary_colors = ['#D7263D', '#1CA9C9', '#FFD700', '#6B8E23', '#FF8C00']
    for idx, (tag, edges) in enumerate(boundaries.items()):
        color = boundary_colors[idx % len(boundary_colors)]
        for n0, n1 in edges:
            x_vals = [nodes[n0][0], nodes[n1][0]]
            y_vals = [nodes[n0][1], nodes[n1][1]]
            ax.plot(x_vals, y_vals, color=color, linewidth=2, label=tag if n0 == edges[0][0] and n1 == edges[0][1] else "")

    ax.set_aspect('equal')
    ax.set_title('SU2 Mesh Visualization')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.grid(True, color='#FFD700', alpha=0.2)  # soft yellow grid
    plt.show()

