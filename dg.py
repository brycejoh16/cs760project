

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

import  ns


def filter_out_None(nk):
    Nk = []
    for elem in nk:
        if elem != None:
            Nk.append(elem)

    return Nk
def make_graph(out,k,show=False):
    """
    Makes DG from NS output.
    :param out: here out is the output from the NS output
    :return: an undirected graph (nx.Graph)
    """
    out=sorted(out, reverse=True)
    print(out)
    G=nx.Graph()

    for i in range(len(out)):
        nk=[None]*k
        dk = np.empty((k,))
        dk[:] = np.inf
        ni=out[i]
        for j in np.arange(i+1,len(out),1):
            if i == j:
                raise Exception("wait what ?")
            nj=out[j]
            d=ni.distance(nj)
            if np.any(dk>d):
                nk[np.argmax(dk)]=(i,j)
                dk[np.argmax(dk)]=d


        # if i ==19 :
        #     print("hello")

        nk=filter_out_None(nk)
        G.add_edges_from(nk)


    if show:
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

    return G

def unit_test_make_graph():
    # out=np.loadtxt('./single_guassian_3walkers/out.txt')
    # out=[ns.Point(x) for x in out]

    out=np.array([-3,2,1,0])#-0.5,-.4,3.5,-4,3,0])
    out = [ns.Point(x) for x in out]
    # print(out)
    G=make_graph(out,2)

    fun_with_graphs(G)



def graph_func(p0,pN,s:nx.Graph):
    N=s.number_of_nodes()
    if N==1:
        y=s.nodes[0] #todo: this is wrong
        x=p0+(p0+pN)/2
        return [(x,y)]


    node2remove=sorted(s.nodes,reverse=True)[0]

    s.remove_node(node2remove)
    m=[]
    for nds in s.nodes:
        if s.degree[nds] ==0: #todo:  check this !
            m.append(nds)
            s.remove_node(nds)

    components=[c for c in nx.connected_components(s)]
    length_list=[len(c) for c in components]



    # randomly put the new basin on either side if the graph
    if np.random.randint(0,2):
        length_list+=[len(m)]
        components+=[{min(m)}]
    else:
        length_list=[len(m)]+length_list
        components=[min(m)]+ components

    # length of the given section
    xypoints=[]
    length_of_rod=(pN*(N-1)+ p0*(N-1))/N
    for l,c,i in zip(length_list,components,np.arange(len(components))):
        if i ==0 :
            # get proper starting point for p0
            p0+=p0 + (pN-p0)/(2*N)
        else:
            p0=pN_tilde
        ratio = l / (N - 1)
        pN_tilde = p0+ ratio*length_of_rod

        xypoint=graph_func(p0,pN_tilde,nx.subgraph(c))

        # add all the points that were found!!!!
        xypoints+=xypoint

    return xypoints



        # the first p0 has to have this specific value. Then after it is additive by a different amount







def fun_with_subgraphs():
    G=nx.Graph()

    G.add_edges_from([(0,1),(0,2),(2,3),(2,4),(1,5),(1,6)])
    print(nx.node_connected_component(G,0))
    for s in nx.connected_components(G):
        print(s)

    G.remove_node(0)
    G.remove_node(1)
    G.remove_node(5)
    print("my components,",nx.node_connected_component(G,2))
    # subgraphs=nx.connected_components(G)

    for c in nx.connected_components(G):
        H=G.subgraph(c)
        nx.draw(H, with_labels=True, font_weight='bold')
        plt.show()

    # for s in subgraphs:
    #     print(s)

def fun_with_graphs(G:nx.Graph):
    N=G.number_of_nodes()
    nodes2remove=np.arange(N-1,-1,-1)

    print(nodes2remove)
    for n2r in nodes2remove:
        # so remove a node
        G.remove_node(n2r)



    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()





    # this function

def make_a_graph():
    G = nx.petersen_graph()
    subax1 = plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    subax2 = plt.subplot(122)
    nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
    plt.show()
if __name__=="__main__":
    # unit_test_make_a_graph_from_list_of_numbers()
    # unit_test_make_graph()

    # fun_with_subgraphs()
    make_a_graph()