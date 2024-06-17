import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = nx.dorogovtsev_goltsev_mendes_graph(2)
adj = nx.adjacency_matrix(G).todense()
adj -= np.tril(adj)
coords = np.array(list(nx.kamada_kawai_layout(G).values()))

plt.figure()
ax = plt.axes()
for out_ in range(len(adj)):
    for in_ in range(len(adj)):
        if adj[out_, in_]:
            x = [coords[out_][0], coords[in_][0]]
            y = [coords[out_][1], coords[in_][1]]
            ax.plot(x, y, 'k')
ax.scatter(coords[:,0], coords[:,1], color='k')
for ix in range(len(coords)):
    ax.annotate(str(ix), xy=(coords[ix,0], coords[ix, 1]), xycoords='data', xytext=(20, 10), textcoords='offset points')
plt.show()
