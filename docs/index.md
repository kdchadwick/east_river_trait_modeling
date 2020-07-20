# Trait model application overview 


<p>This repository handles the management and merging of NEON AOP remote sensing data with field/lab-based trait data.
The code leverages a plsr ensembling routine (external to this repository) to create trait models and assess the
model performance. This code was created as part of an effort to generate foliar trait maps throughout the Department of Energy (DOE) Watershed Function Scientific Focus Area (WF-SFA) site in Crested Butte, CO in association with NEON's Assignable Asset program.</p><br>

A full description of the effort can be found at:

> K. Dana Chadwick, Philip Brodrick, Kathleen Grant, Tristan Goulden, Amanda Henderson, Nicola Falco, Haruko Wainwright, Kenneth H. Williams, Markus Bill, Ian Breckheimer, Eoin L. Brodie, Heidi Steltzer, C. F. Rick Williams, Benjamin Blonder, Jiancong Chen, Baptiste Dafflon, Joan Damerow, Matt Hancher, Aizah Khurram, Jack Lamb, Corey Lawrence, Maeve McCormick. John Musinsky, Samuel Pierce, Alexander Polussa, Maceo Hastings Porro, Andea Scott, Hans Wu Singh, Patrick O. Sorensen, Charuleka Varadharajan, Bizuayehu Whitney, Katharine Maher. Integrating airborne remote sensing and field campaigns for ecology and Earth system science. <i>In Review</i>, 2020.

and use of this code should cite that manuscript.

### Visualization code in GEE for all products in this project can be found here: 
https://code.earthengine.google.com/5c96bbc96ffd50e3c8b1433b34a0bb86
<br>

### Generated datasets are available as assets on GEE: 
https://code.earthengine.google.com/?asset=users/kdc/ER_NEON <br>
<br> 
and are part of the data package: 
> Chadwick K D ; Brodrick P ; Grant K ; Henderson A ; Bill M ; Breckheimer I ; Williams C F R ; Goulden T ; Falco N ; McCormick M ; Musinsky J ; Pierce S ; Hastings Porro M ; Scott A ; Brodie E ; Hancher M ; Steltzer H ; Wainwright H ; Maher K W; undefined K M (2020): NEON AOP foliar trait maps, maps of model uncertainty estimates, and conifer map. A Multiscale Approach to Modeling Carbon and Nitrogen Cycling within a High Elevation Watershed. DOI: 10.15485/1618133
<br>
<br>

## Additional relevant repositories:

### Atmospheric correction wrapper: 
https://github.com/pgbrodrick/acorn_atmospheric_correction

### Shade ray tracing: 
https://github.com/pgbrodrick/shade-ray-trace

### Conifer Modeling:
https://github.com/pgbrodrick/conifer_modeling

### Trait Model Generation:
https://github.com/kdchadwick/east_river_trait_modeling

### PLSR Ensembling:
https://github.com/pgbrodrick/ensemblePLSR

