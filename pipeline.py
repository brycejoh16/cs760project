
import ns,dg,fs
from scipy.stats import multivariate_normal
import numpy as np
import torch
import matplotlib.pyplot as plt
import generator as g
import labellingFunctions as lf
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

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


class labeling(ns.Point):
    # parent class for all the labeling functions,
    # since only thing that will change will be the
    # calls to find_fitness
    def __init__(self, x=None):
        # i could just save the x vector in
        self.generator=g.load_generator()
        if np.all(x==None):
            z = g.z()
            with torch.no_grad():
                pz = self.generator(z).view(28 * 28)
            x = g.torch2numpy(pz)
            # want to generate a single image from the generator.
            # then save it as a flattened numpy array.
        super().__init__(x)
    def mutate(self):
        ## need to get a new z.
        z=g.z()
        with torch.no_grad():
            pz = self.generator(z).view(28*28)
        self.x=g.torch2numpy(pz)


class lambda1(labeling):
    def find_fitness(self):
        return lf.fillSum(self.x)
class lambda2(labeling):
    def find_fitness(self):
        return lf.fillCount(self.x)

class lambda3(labeling):
    def find_fitness(self):
        return lf.euclideanDistance(self.x)

class lambda4(labeling):
    def find_fitness(self):
        return lf.pixelCount(self.x)
class lambda5(labeling):
    def find_fitness(self):
        return lf.euclideanBinarize(self.x)


class horizontalPeakCount(labeling):
    def find_fitness(self):
        return lf.horizontalPeakCount(self.x)
class horizontalPeakCount_params(labeling):
    def find_fitness(self):
        return lf.horizontalPeakCount_params(self.x)

class lambda6(labeling):
    def find_fitness(self):
        return lf.ratioPeakCount(self.x)

class lambda7(labeling):
    def find_fitness(self):
        return lf.edgeDetectRatio(self.x)
def main_ns(input):
    T=ns.ns(**input)
    # # save the data what we just made
    X=np.array([p.x for p in T])
    np.savetxt(fs.filename_ns(input), X)


def artists(ax,arr,xy,offset):

    im = OffsetImage(arr, zoom=.5)
    im.image.axes = ax
    ab = AnnotationBbox(im, xy,
                        xybox=(0., offset),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.3,
                        arrowprops=dict(arrowstyle="->")
                        )
    ax.add_artist(ab)

def main_dg(neighbors,input):
    fname=fs.filename_ns(input)

    X=np.loadtxt(fname)
    out=[input['point'](x=x) for x in X]
    xypoints, out = dg.get_xypoints(neighbors, out)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, (1, 2))


    x = [xy[0] for xy in xypoints]
    y = [xy[1] for xy in xypoints]

    offset=[25,50]
    j=0
    for xy in xypoints:
        if len(xy)>2 and len(xy[2])>2:
            artists(ax,xy[2].reshape(28,28),(xy[0],xy[1]),offset[j%2])
            j+=1


    # ax.set_title(f"Neighbors {neighbors}")
    ax.plot(x, y, markersize=5, marker='o')
    ax.set_xticks([])
    ax.set_ylabel('fitness')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # todo: still need to include the phase space on the other axis
    y=np.array(y)
    s=y.std()
    ax.set_ylim([y.min()-s/2,y.max()+s/2])

    # ax.set_xlabel(f"{input['point'].__name__}")
    fig.savefig(fs.filename_dg(input,neighbors))

        # want to run dg for however many neighbors i've specified and save figures

        # then like yeah nvmd that's all i want to do lol.



def unit_test_labeling_class():

    # could run mutations here and show that like most images that are generated
    # we expect to have low fitness. B/c they are a bit nonsense. But I'd say that's evidence its working. LOL>)
    lbs=[labeling() for _ in range(2)]
    print(lbs[0].x.min(),lbs[0].x.max())
    fig=plt.figure()
    # ax=fig.add_subplot(1,4,1)
    # lbs[0].mutate()
    # ax.imshow(lbs[0].x.reshape(28,28))
    # ax.set_title("%0.2f"%(lf.euclideanDistance(lbs[0].x)))
    # ax=fig.add_subplot(1,4,2)
    # ax.imshow(lbs[1].x.reshape(28, 28))
    #
    # ax.set_title("%0.2f"%(lf.euclideanDistance(lbs[1].x)))
    # lbs[0].mutate()
    #
    # ax=fig.add_subplot(1,4,3)
    # ax.imshow(lbs[0].x.reshape(28,28))
    # ax.set_title("%0.2f"%(lf.euclideanDistance(lbs[0].x)))
    for i in range(25):
        lbs[0].mutate()
    for i in range(25):

        ax=fig.add_subplot(5,5,i+1)
        ax.imshow(lbs[0].x.reshape(28, 28))
        ax.set_title("%0.2f" % (lf.euclideanDistance(lbs[0].x)))
        lbs[0].mutate()

    # oh wait in the normalized coordinate system it makes sense actually.

    ## ohhh lol i see. the more -1's their are the better. So it skews towards that.
    ### okay that makes sense.
    #
    # ax=fig.add_subplot(1,4,4)
    # ax.imshow(lbs[1].x.reshape(28,28))
    # ax.set_title("%0.2f"%(lf.euclideanDistance(lbs[1].x)))
    fig.suptitle('euclidean distance from -1 matrix')
    fig.show()



if __name__=="__main__":



    # make sure to have an N that properly dissociates

    input= {'point': horizontalPeakCount_params, 'm': 20, 'K': 250, 'N': 100}
    # i bet you lambda3 didn't actually converge lol. b/c like why it makes
    # no sense. some of them should have way higher norms.
    # main_ns(input)
    main_dg(2,input)
    # unit_test_labeling_class()
    # unit_test_labeling_class()






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