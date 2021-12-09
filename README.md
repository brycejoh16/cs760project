# Visualization of Labeling Function Landscapes


To set up environment the following libraries are needed :

``scikit-image=0.18.*             
scikit-learn=1.0.*  numpy=1.19.* pytorch=1.7.1 python=3.8.* matplotlib=3.4.3 networkx=2.6.3``


Note snorkel requires that you use networkx 0.2.6 while these modules require 2.6.*. 
Will need to make two seperate environments to run the code, one for snorkel, 
one for the visualization. A bummer, but a consequence of dependencies I guess. 




To run the output of a labeling function go to  `pipeline.py` and specify the input via a choice of a labeling function in the input dictionary.

Then look at output in the corresponding folder ./data/<labeling_func>

To look at the plots made for the nested sampling figures, look at `paper_plots.py`
and just run this function.

Description of major modules:
1. `ns.py` - contains the code for the nested sampling run
2.  `dg.py` - contains the code to return the x and y points for the disconnectivity graph visualization.
3. `pipeline.py` - entry point to do runs over different labeling functions.
