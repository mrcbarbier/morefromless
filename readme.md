Simulation experiments for multi-scale ecological interactions. (v1.0 - July 14th 2018)
=======

## Requirements:
 Python 2.7

 Libraries: numpy, scipy, pandas, matplotlib, seaborn


## Simulation runs and figures for the article "Getting More by Asking for Less: Linking Species Interactions to Species Co-Distributions in Metacommunities" (https://doi.org/10.1101/2023.06.04.543606 )

    python gettingmore.py

## Where to edit:

   Default parameter values:  default_parameters.dat

   Plots: plots.py

## How to process images in /movie folder into a gif (Linux, with ImageMagick)

   convert -delay 10 -loop 0 'fig%d.png[0-50]' movie.gif
