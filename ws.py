

import labellingFunctions as lf
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import generator as g
import matplotlib.pyplot as plt
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
])
train_data = datasets.MNIST(
    root='input/data',
    train=True,
    download=True,
    transform=transform
)

def testing_labelingfunc():

    zero=train_data.data[train_data.targets==0][0]
    x=g.torch2numpy(zero)
    fig=plt.figure()
    fig.suptitle('fill sum')
    ax=fig.add_subplot(1,2,1)
    ax.imshow(x)
    count,filled=lf.fillSum(x)
    ax=fig.add_subplot(1,2,2)
    ax.imshow(filled)


    fig.show()
if __name__ == '__main__':
    testing_labelingfunc()





# this is the file for doing the weak supervision.
## i'll have to think of someway to keep consistency between labeling functions.
### init a single point. and then like just return the call to find_fitness.
### very stupid. but at the same time. not stupid, if its not slow.

## use snorkel now to evaluate the long string of data that i have.

### will probably have to make helper functions to make sure to have the correct thresholds and stuff.
### but lol. like if I don't have helper functions. I will need helper functions. fuck. lol. Their is no way around it.
### well their is but helper functions is fast and quick and easy and won't have multiple versions of the functions themselves.




###### so now using that idea just need to use labeling functions to like do WS. see the accuracy...

##### lol how are we going to test the accuracy metric here. idk yet. well that is definetely still up for debate.
# I'm going to have to refresh myself on a lot of snorkel stuff.