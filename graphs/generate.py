import csv
import networkx as nx
import numpy as np

def process(adj):
    degrees = np.sum(adj, axis=0)
    print("max degree:", max(degrees))
    print("average degree:", np.mean(degrees))
    adj = np.triu(adj, k=1)
    print("number of edges:", sum(sum(adj)))
    return adj

def save(adj, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(adj)

def barabasi_albert():
    Ns = [100, 500, 1000, 2000, 5000]
    m = 2
    for N in Ns:
        print("barabasi_albert", N)
        G = nx.barabasi_albert_graph(N, m)
        adj = process(nx.adjacency_matrix(G).todense())
        save(adj, f"barabasi_albert_{N}.txt")

def dorogovtsev_goltsev_mendes():
    Ns = [5, 6, 7, 8, 9]
    for N in Ns:
        print("dorogovtsev_goltsev_mendes", N)
        G = nx.dorogovtsev_goltsev_mendes_graph(N)
        adj = process(nx.adjacency_matrix(G).todense())
        save(adj, f"dorogovtsev_goltsev_mendes_{N}.txt")

def scale_free():
    Ns = [100, 500, 1000, 2000, 5000]
    for N in Ns:
        print("scale_free", N)
        G = nx.scale_free_graph(N)
        adj = process(nx.adjacency_matrix(G).todense())
        save(adj, f"scale_free_{N}.txt")

#barabasi_albert()
dorogovtsev_goltsev_mendes()
#scale_free()
