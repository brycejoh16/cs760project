
import ns,dg,fs
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x,b=0):
    return (np.exp((-(x-b)**2)/2) / np.sqrt(2*np.pi)) # sigma =1 , mu=0

def helper_fitness_multi1D():
    y = lambda x: gaussian(x, b=0) + gaussian(x, b=4)
    return y

class multiGuass1d(ns.Point):
    def __init__(self):
        super().__init__(np.random.uniform(*ns.bounds))
    def find_fitness(self):
        return helper_fitness_multi1D()(self.x)
    def mutate(self):
        self.x = np.random.uniform(*ns.bounds)

def helper_multiVariateNormal2D_unimodal():
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])

    return lambda x:rv.pdf(x)

class multiVariateNormal2D_unimodal(ns.Point):
    def __init__(self,x=None):
        if np.all(x !=None):
            super().__init__(x=x)
        else:
            x = np.random.uniform(*ns.bounds, size=2)
            super().__init__(x=x)


    def find_fitness(self):
        return helper_multiVariateNormal2D_unimodal()(self.x)

    def mutate(self):
        # for now just mutate one feature dimension b/c Idk how else one would do it.
        # self.x[np.random.randint(0,2)]=np.random.uniform(*bounds)
        self.x=np.random.uniform(*ns.bounds,size=2)

def helper_multiVariateNormal2D_multimodal():
    rv = multivariate_normal([-1, 2], [[1, -.5], [-.5, 1]])
    ra = multivariate_normal([2, 2], [[1, 0], [0, 1]])
    rb=multivariate_normal([-4,-4],[[1,0],[0,1]])
    return lambda x: rv.pdf(x)+ra.pdf(x)+rb.pdf(x)

class multiVariateNormal2D_multimodal(multiVariateNormal2D_unimodal):
    def find_fitness(self):
        return helper_multiVariateNormal2D_multimodal()(self.x)



# class labeling(ns.Point):
#     # parent class for all the labeling functions,
#     # since only thing that will change will be the
#     # calls to find_fitness
#     def __init__(self, x=None):
#         # i could just save the x vector in
#         super().__init__(x)
#         self.generator= # load the model here
#         if np.all(self.x==None):
#             pass
#             # want to generate a single image from the generator.
#
#     def mutate(self):
#         pass
#     def distance(self,point):
#         pass






def main_ns(input):
    T=ns.ns(**input)
    # # save the data what we just made
    X=np.array([p.x for p in T])
    np.savetxt(fs.filename_ns(input), X)

def main_dg(neighbors,input):
    fname=fs.filename_ns(input)

    X=np.loadtxt(fname)
    out=[input['point'](x=x) for x in X]
    xypoints, out = dg.get_xypoints(neighbors, out)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, (1, 2))
    x = [xy[0] for xy in xypoints]
    y = [xy[1] for xy in xypoints]
    ax.set_title(f"Neighbors {neighbors}")
    ax.plot(x, y, markersize=5, marker='o')
    ax.set_xticks([])
    ax.set_ylabel('fitness')
    fig.savefig(fs.filename_dg(input,neighbors))

        # want to run dg for however many neighbors i've specified and save figures

        # then like yeah nvmd that's all i want to do lol.


if __name__=="__main__":


    input= {'point': multiVariateNormal2D_multimodal, 'm': 20, 'K': 100, 'N': 100}
    # main_ns(input)

    main_dg(3,input)









# if sys.argv[1]=='ns':
# try:
#     point = eval(sys.argv[2])
#     print(point)
#     input = {'point': point, 'm': 20, 'K': 25, 'N': 100}
#     for i in np.arange(3, len(sys.argv), 1):
#         key = sys.argv[i][0]
#         print(key)
#         print(int(sys.argv[i][2:])
#         # input[key]=)
# except:
#     pass