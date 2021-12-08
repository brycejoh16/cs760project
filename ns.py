import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib import cm
import fs

from scipy.stats import multivariate_normal

# make a multivariate gaussian.

bounds=[-10,10]


class Point:
    def __init__(self,x=None):
        if x is None:
            raise Exception("must define self.x silly")
        self.x=x

    def find_fitness(self):
        raise Exception("Must be over-ridden in child class ")

    def mutate(self):
        raise Exception("Must be over-ridden in child class")

    def __repr__(self):
        if self.x.shape[0]<3:
            return f"({self.x},{self.find_fitness()})"
        return f"({self.find_fitness()})"

    def __call__(self):
        return self.find_fitness()

    def distance(self,point):
        # returns distance from one point to another.
        return  np.linalg.norm(self.x-point.x)

    def __lt__(self, other):
        return self.find_fitness() < other.find_fitness()

    def __hash__(self):
        return hash((self.x[0],self.x[1], self.find_fitness()))

    def __eq__(self, other):
        tol=1e-6
        xtol=np.ones_like(other)*tol
        if not isinstance(other, type(self)): return NotImplemented
        return np.all(abs(self.x - other.x) < xtol) and abs(self.find_fitness()- other.find_fitness())<tol

def cp(point):
    return copy.deepcopy(point)

def ns(point,m=10,K=50,N=50,checkpoint=None):
    # sample m points
    np.random.seed(30)
    points=[point() for _ in range(m)]

    # start nested sampling loop
    T=[]

    for k in range(K):
        points.sort(key=lambda a:a())
        threshold=cp(points[0])


        if k > 0:
            if threshold==T[-1]:
                print(f'stopping nested sampling at iteration: {k} of {K}')
                break

        print("threshold",threshold)

        # resample to a new point in the list
        testpoint=cp(points[np.random.randint(1,m)])

        # the monte-carlo random walk
        for n in range(N):
            new_testpoint=cp(testpoint)
            new_testpoint.mutate()
            if new_testpoint() >= threshold():
                testpoint=new_testpoint



        points[0]=testpoint


        T.append(threshold)


        if checkpoint!=None:
            if k % checkpoint==0:
                # save stuff.
                input={'point':point,'m':m,'K':K,'N':N}
                fname=fs.filename_ns(input)
                X=np.array([t.x for t in T])
                np.savetxt(fname,X)

    return T


def fun_with_3Dplots():
    # defining surface and axes
    x = np.outer(np.linspace(-2, 2, 100), np.ones(100))
    y = x.copy().T


    # i guess that will work for mutations.
    # mean = (1, 2)
    # cov = [[1, 0], [0, 1]]
    # x = np.random.multivariate_normal(mean, cov, (3, 3))
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])

    pos = np.dstack((x, y))

    z = rv.pdf(pos)

    fig = plt.figure()

    # syntax for 3-D plotting
    ax = plt.axes(projection='3d')

    # syntax for plotting
    ax.plot_surface(x, y, z, cmap='autumn_r', edgecolor='orange')
    ax.set_title('Surface plot geeks for geeks')
    plt.show()

if __name__=="__main__":
    pass
    # unit_test_ns_for_multivariate_gaussian()
    # plot_gaussian()
    # fun_with_3Dplots()
    # ns(multiGuass1d)
    # print(plt.colormaps())


    #https://stackoverflow.com/questions/25741214/how-to-use-colormaps-to-color-plots-of-pandas-dataframes


        # plot_gaussian_with_points(points)
        # plt.title(f"NS k:{k}")
        # plt.savefig(f"./single_guassian_3walkers/k_{k}.png")
        # plt.clf()

    # print("Threshold values",T)

    # cmap=cm.get_cmap('autumn_r')
    # plot_gaussian_with_points(helper_fitness_multi1D(),T,cmap=cmap)
    # plt.title(f"Guassian N($\mu$=0,$\sigma=1$) with threshold \npoints from each iterations of NS")
    # plt.title(f"Multi Modl Guassian")
    # plt.show()
    # plt.clf()


    # out1=np.array([p.x for p in T]).T
    # np.savetxt('./single_guassian_3walkers/out.txt',out1)
