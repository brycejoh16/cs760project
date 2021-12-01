
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib import cm
# make a multivariate gaussian.
def gaussian(x):
    return np.exp((-x**2)/2) / np.sqrt(2*np.pi) # sigma =1 , mu=0

bounds=[-10,10]

def plot_gaussian_with_points(points=None,cmap=None):
    x=np.arange(bounds[0],bounds[1],.1)
    y=gaussian(x)
    plt.plot(x,y,'--',label='true',c='k')
    x1=np.array([p.x for p in points])
    y1=np.array([p.find_fitness() for p in points])
    if cmap !=None:
        CS=plt.scatter(x1,y1 , c=np.arange(len(points)), cmap=cmap, edgecolor='None')
        cbar=plt.colorbar(CS, cmap=cmap)
        cbar.set_label('Iteration', rotation=270, labelpad=25)
    else:
        plt.scatter(x1,y1,label="samples")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")


class Point:
    def __init__(self,x=None):
        if x==None:
            self.x=np.random.uniform(*bounds)
        else:
            self.x=x
    def find_fitness(self):
        return gaussian(self.x)

    def mutate(self):
        self.x=np.random.uniform(*bounds)

    def __repr__(self):
        return f"({self.x:0.5f},{self.find_fitness()})"
    def __call__(self):
        return self.find_fitness()
    def distance(self,point):
        # returns distance from one point to another.
        return  np.linalg.norm(self.x-point.x)

    def __lt__(self, other):
        return self.find_fitness() < other.find_fitness()



def cp(point):
    return copy.deepcopy(point)


def ns(m=4,K=25,N=50):
    # sample m points
    np.random.seed(30)
    points=[Point() for _ in range(m)]

    # start nested sampling loop
    T=[]
    for k in range(K):
        points.sort(key=lambda a:a())
        threshold=cp(points[0])
        print("threshold",threshold)
        testpoint=cp(points[np.random.randint(1,m)])

        # the monte-carlo random walk
        for n in range(N):
            new_testpoint=cp(testpoint)
            new_testpoint.mutate()
            if new_testpoint() >= threshold():
                testpoint=new_testpoint


        points[0]=testpoint
        T.append(threshold)

        plot_gaussian_with_points(points)
        plt.title(f"NS k:{k}")
        plt.savefig(f"./single_guassian_3walkers/k_{k}.png")
        plt.clf()

    print("Threshold values",T)

    cmap=cm.get_cmap('autumn_r')
    plot_gaussian_with_points(T,cmap=cmap)
    plt.title(f"Single Guassian N($\mu$=0,$\sigma=1$) with threshold \npoints from each iterations of NS")
    plt.savefig(f"./single_guassian_3walkers/thresholds.png")
    plt.clf()


    out1=np.array([p.x for p in T]).T
    np.savetxt('./single_guassian_3walkers/out.txt',out1)
    return T

def disconnectivity_graph():
    pass
    # make sure that this algorithm is robust enough to interchange the distance function.
    # since that is what will be changing between calls
if __name__=="__main__":
    # plot_gaussian()
    ns()
    # print(plt.colormaps())


    #https://stackoverflow.com/questions/25741214/how-to-use-colormaps-to-color-plots-of-pandas-dataframes