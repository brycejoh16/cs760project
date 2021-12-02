

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from matplotlib import cm
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
    print(np.arange(len(out)))
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

    return G,out

def unit_test_make_graph():
    # out=np.loadtxt('./single_guassian_3walkers/out.txt')
    # out=[ns.Point(x) for x in out]

    out=np.array([-3,2,1,0])#-0.5,-.4,3.5,-4,3,0])
    out = [ns.Point(x) for x in out]
    # print(out)
    G=make_graph(out,2,True)

    # fun_with_graphs(G)


def graph_func(p0,pN,s:nx.Graph,out):
    N=s.number_of_nodes()

    if N==1:
        y=out[list(s.nodes)[0]]() #todo: this is wrong, for now ...
        x=p0+(pN-p0)/2
        return [(x,y)]


    node2remove=sorted(s.nodes,reverse=True)[0]

    s.remove_node(node2remove)
    m=[]
    nodes=list(s.nodes)
    for nds in nodes:
        if s.degree[nds] ==0:
            m.append(nds)
            s.remove_node(nds)





    components=[c for c in nx.connected_components(s)]
    length_list=[len(c) for c in components]



    # todo: test this function with an ideal test set to make sure its working


    #randomly put the new basin on either side if the graph, we don't want new ,
    # basins always on same side of the graph.


    if np.random.randint(0,2) and len(m)>0:
        m_i=min(m)
        length_list+=[len(m)]
        components+=[{m_i}]
        s.add_edge(m_i,m_i)
    elif len(m)>0:
        m_i=min(m)
        length_list=[len(m)]+length_list
        components=[{m_i}]+ components
        s.add_edge(m_i,m_i)

    # length of the given section
    xypoints=[]
    length_of_rod=(pN-p0)-(pN-p0)/N
    for l,c,i in zip(length_list,components,np.arange(len(components))):
        # the first p0 has to have this specific value. Then after it is additive by a different amount
        if i ==0 :
            # get proper starting point for p0
            p0=p0 + (pN-p0)/(2*N)
        else:
            p0=pN_tilde
        ratio = l / (N - 1)
        pN_tilde = p0+ ratio*length_of_rod


        # something here is messed up with my p0 and pN_tilde's
        xypoints+=[(p0,out[node2remove]())]
        xypoints+=[(pN_tilde,out[node2remove]())]
        H=nx.Graph()
        H.add_edges_from(list(s.edges(c)))
        xypoint=graph_func(p0,pN_tilde,H,out)
        # add all the points that were found!!!!
        xypoints+=xypoint


    return xypoints

def unit_test_graph_func():

    # todo: before you get to excited you still don't know if the reduction in
    #  phase space is correct since you never tested it. So don't get to pumped yet.
    #  like their still definetly some errors in here.
    # also need to test the possiblity of a multi - split.

    G = nx.Graph()
    G.add_edges_from([(2, 1), (2, 4),(5, 6), (5, 7), (6, 7),(6,1),(4,1),(1,9)])

    subax1 = plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')


    xypoints=graph_func(0,1,G)
        # your not even including the intial pionts boy..
    xypoints+=[(1,0),(0,0)]

    xypoints=sorted(xypoints, key=lambda xy : xy[0])

    x=[xy[0] for xy in xypoints]
    y=[xy[1] for xy in xypoints]


    subax2 = plt.subplot(122)
    plt.plot(x,y,markersize=10,marker='o')
    plt.ylabel('node value')
    plt.xlabel('')
    plt.show()




def unit_Test_graph_ns(neighbors=2):
    # out = np.array([-3, 2, 1, 0, -0.5,-.4,3.5,-4,3,0])

    # out=np.loadtxt('single_guassian_3walkers/out.txt')
    # out = [ns.Point(x) for x in out]
    input={'point':ns.multiGuass1d, 'm' : 5, 'K' : 15, 'N' :100}
    out=ns.ns(**input)
    # print(out)

    # the ordering of out effects the results a lot,
    # so to remain consistent I've decided to just have out ,
    # be returned from the graph so its exact.
    G,out = make_graph(out,neighbors, False)

    # subax1 = plt.subplot(131)
    # nx.draw(G, with_labels=True, font_weight='bold')


    xypoints = graph_func(0, 1, G,out)
    # your not even including the intial pionts boy..
    xypoints += [(1, 0), (0, 0)]

    xypoints = sorted(xypoints, key=lambda xy: xy[0])

    x = [xy[0] for xy in xypoints]
    y = [xy[1] for xy in xypoints]
    plt.title(f"Neighbors{neighbors}")
    subax2 = plt.subplot(121)
    plt.plot(x, y, markersize=5, marker='o')

    plt.ylim([-.1,.5])



    subax3= plt.subplot(122)

    cmap = cm.get_cmap('autumn_r')
    ns.plot_gaussian_with_points(ns.helper_fitness_multi1D(),points=sorted(out), cmap=cmap)
    plt.ylim([-.1,.5])
    plt.savefig(f"./dg_unit_tests/{input},neighbors={neighbors}.png")



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

def peterson_make_a_graph():
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
    # make_a_graph()
    # unit_test_graph_func()
    unit_Test_graph_ns()