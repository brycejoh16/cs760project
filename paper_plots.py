

import dg


def neighbors_comparison_multi_modal_gaussian():

    for k in [1,3,10]:
        dg.unit_Test_graph_ns(neighbors=k)



if __name__=="__main__":
    neighbors_comparison_multi_modal_gaussian()