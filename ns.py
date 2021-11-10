
import numpy as np
import matplotlib.pyplot as plt
import copy
# make a multivariate gaussian.
def gaussian(x):
    return np.exp((-x**2)/2) / np.sqrt(2*np.pi) # sigma =1 , mu=0

bounds=[-10,10]

def plot_gaussian_with_points(points=None):
    x=np.arange(bounds[0],bounds[1],.1)
    y=gaussian(x)
    plt.plot(x,y,label='true')
    x1=np.array([p.x for p in points])
    y1=np.array([p.find_fitness() for p in points])
    plt.scatter(x1,y1,label="samples")
    plt.legend()

class Point:
    def __init__(self):
        self.x=np.random.uniform(*bounds)

    def find_fitness(self):
        return gaussian(self.x)

    def mutate(self):
        self.x=np.random.uniform(*bounds)

    def __repr__(self):
        return f"({self.x:0.5f},{self.find_fitness()})"
    def __call__(self):
        return self.find_fitness()

def cp(point):
    return copy.deepcopy(point)



def ns(m=3,K=25,N=10):
    # sample m points
    np.random.seed(30)
    points=[Point() for _ in range(m)]

    # plot_gaussian_with_points(points)
    # plt.title("start")
    # plt.show()

    # start nested sampling loop
    T=[]
    for k in range(K):
        points.sort(key=lambda a:a())
        threshold=cp(points[0])
        print("threshold",threshold)
        testpoint=cp(points[np.random.randint(1,m)])
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

if __name__=="__main__":
    # plot_gaussian()
    ns()

