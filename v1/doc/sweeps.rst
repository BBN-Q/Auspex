Sweep Documentation
===================

Sweeping in measurement software is often a very rigid afair. Typical parameter sweeps take the form ::

    for this in range(10):
        for that in np.linspace(30.3, 100.6, 27):
            set_this(this)
            set_that(that)
            measure_everything()

A few questions arise at this point:

1. Is there any reason to use a rectangular grid of points? 
2. Am I wasting time if features aren't uniformly distributed over this grid?
3. What if our range wasn't sufficient to capture the desired data?
4. What if we didn't get good enough statistics?

To tackle (1), there are some clear reasons to measure on a rectilinear grid. First of all, it is extremely convenient. Also, if you expect that regions of interest (ROI) are distributed evenly across your measurement domain then this seems like a reasonable choice. Finally there are simple aesthetic considerations: image plots look much better when they are fill a rectangular domain rather than leaving swaths of NaNs strewn the periphery of your image. 

Point (2) is really an extension of (1): if you are looking at data that follows ``sin(x)*sin(y)`` then the information density is practically constant across the domain. If you are looking at a crooked phase transition in a binary system, then the vast majority of your points will be wasted on regions with very low information content. Take the following phase diagrams for an MRAM cell's switching probability.

Structured Sweeps
*****************

Unstructured Sweeps
*******************
