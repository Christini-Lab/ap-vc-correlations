## Rapid ionic current phenotyping (RICP) identifies mechanistic underpinnings of iPSC-CM AP heterogeneity 


By: Alexander P. Clark, Siyu Wei, Kristin Fullerton, David J. Christini, Trine Krogh-Madsen


### Summary

All code required to complete the analyses and plot figures within the following manuscript:

TODO


### Directory structure

- All python scripts are in the root directory. They are used to generate figures, run simulations, and analyze data.
- **data/** – Contains all *in vitro* experimental, modeling, and literature data used in the analyses and figures. This data must be downloaded separately (see end of README). *in vitro* experimental data can be found in **cells/** and **cell_metadata/**. The **ap_features.csv** file contains AP feature data for each cell. The **literature_ap_morph.xlsx** file contains AP feature summary data. The **mod_populations/** directory contains information about the population of models.
- **mmt/** – Contains the .mmt files.
- **figure-pdfs** – All the figures, in pdf format, that are generated by the python scripts. This folder, at times, may have more than the number of figures in the manuscript, as I do not always clean up after changing names or archiving scripts. 


### Software requirements 



### Download data

1. `cd` into `/figures`. 
2. Download all raw experimental data from [HERE, TODO]() and extract the data inside the root. That's it!