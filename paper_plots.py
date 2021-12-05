

import dg,fs
import numpy as np
import ns
import matplotlib.pyplot as plt
import pipeline as pl
def neighbors_comparison_multi_modal_gaussian():

    for k in [1,2,3,10]:
        dg.unit_Test_graph_ns(neighbors=k)

def visualize_landscape():
    input = {'point': pl.multiVariateNormal2D_multimodal, 'm': 20, 'K': 100, 'N': 100,'checkpoint':10}
    out = ns.ns(**input)

    # you have to do this b/c this is what is done in the DG.
    out=sorted(out, reverse=True)

    x, y = np.mgrid[ns.bounds[0]:ns.bounds[1]:.1, ns.bounds[0]:ns.bounds[1]:.1]
    pos = np.dstack((x, y))
    z = pl.helper_multiVariateNormal2D_multimodal()(pos)
    fig = plt.figure()

    # syntax for 3-D plotting
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.set_title('(B)')
    # for point in points:
    #     ax.scatter(point.x[0],point.x[1],point.find_fitness(),c='black')
    # syntax for plotting
    surf = ax.plot_surface(x, y, z, cmap='autumn_r')
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.set_label('fitness', rotation=270, labelpad=25)
    ax.set_zticks([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(1, 2, 1)
    x_s = np.array([p.x[0] for p in out])
    y_s = np.array([p.x[1] for p in out])
    f_s = np.array(sorted([p() for p in out]))
    ax.contour(x, y, z, colors='lightgrey', levels=f_s, zorder=-1, alpha=.3, linewidths=.5)

    cont = ax.scatter(x_s, y_s, c=np.arange(len(out) - 1, -1, -1), cmap='autumn_r', s=5, zorder=1)
    cbar = fig.colorbar(cont, shrink=0.5, aspect=10)
    cbar.set_label('Iteration', rotation=270, labelpad=25)
    ax.set_title('(A)')
    ax.set_ylabel("$x_2$")
    ax.set_xlabel("$x_1$")
    fig.savefig(fs.make_dir(input['point'])+f"/{input}.png")
def plot_gaussian_with_points(fitness_func,points=None,cmap=None):
    x=np.arange(ns.bounds[0],ns.bounds[1],.1)
    y=fitness_func(x)
    plt.plot(x,y,'--',label='true',c='k')
    x1=np.array([p.x for p in points])
    y1=np.array([p.find_fitness() for p in points])
    if cmap !=None:
        CS=plt.scatter(x1,y1 , c=np.arange(len(points)), cmap=cmap, edgecolor='None')
        cbar=plt.colorbar(CS, cmap=cmap)
        cbar.set_label('Iteration', rotation=270, labelpad=25)
    else:
        plt.scatter(x1,y1,label="samples")

if __name__=="__main__":
    # neighbors_comparison_multi_modal_gaussian()
    visualize_landscape()