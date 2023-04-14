# Elwha_woodmapping
Codes for carrying out Elwha river wood morphodynamics study. How did wood affect the geomorphic evolution of the Elwha river and delta during the world's largest dam removal?

Written by Daniel Buscombe, Marda Science LLC, for the USGS Coastal Hazards and Resources Program Landscape Response to Disturbance Project.

USGS collaborators:
* Amy East
* Amy Foxgrover
* Andy Ritchie
* Josh Logan
* Jon Warrick

## Overview
Large quantities of wood were released from Lake Mills (former upper reservoir) and Lake Aldwell (lower former reservoir) during dam removal on the Elwha river (2011-2014). he Elwha and the Glines Canyon dams were located approximately 7 km and 20 km upstream, respectively, from the Elwha River's mouth on the Strait of Juan de Fuca. A sediment pulse consisting of 20 million tons over 5 years made it to the coast, but we don't know how much wood made it to the delta and adjacent beaches, and at what rate.

What was the role of wood in the magnitude and timing of sediment redistributions, and channel evolutions such as river avulsions and braiding? How does this behavior change when associated with a large sediment pulse due to a disturbance, compared to a natural background rate and magnitude of wood?


## Model implementation

Sendrowski et al (2023) recently showed that deep-learning-based image-segmentation could effectively be made at large scales to map wood from remotely sensed imagery. In a similar vein, we use similar techniques on a time-series of SfM-derived orthomosaic imagery using a commodity camera mounted to an aircraft. Such imagery

We used a Segformer (Xie et al., 2021) model pre-trained on ImageNet, fine-tuned using labels made using Doodler (Buscombe et al., 2021), within the Segmentation Gym (Buscombe and Goldstein, 2022) software package. The SegFormer model architecture uses a hierarchical Transformer architecture, called "Mix Transformer", as an encoder, and a lightweight decoder for segmentation. It yields state-of-the-art performance on semantic segmentation while being more efficient than existing models. Previous attempts to train a Residual UNet on this training dataset resulted in worse performance.

Imagery were split into 768x768 pixel tiles, with 50% overlap. Each model was applied to each tile. Separate models were trained to segment wood, sediment, vegetation, development, and water. Those per-tile probabilities are mosaiced into a large geotiff file. Those basic maps serve as the inputs to the current workflow, which uses GDAL, Dask, and Xarray to combine, clean, and analyze the pixel-level wood, water, sediment, development probability maps.

The initial focus is on the so-called "middle reach" (MR) below the former upper reservoir, and the "lower reach" (LR) below the former lower reservoir.

## Data

* Ritchie, A.C., Curran, C.A., Magirl, C.S., Bountry, J.A., Hilldale, R.C., Randle, T.J., and Duda, J.J., 2018, Data in support of 5-year sediment budget and morphodynamic analysis of Elwha River following dam removals: U.S. Geological Survey data release, https://doi.org/10.5066/F7PG1QWC.
	* Digital elevation models (DEMs) of the lower Elwha River, Washington, water year 2013 to 2016
	* Orthomosaic images of the middle and lower Elwha River, Washington, 2012 to 2017


## References and background reading

* Buscombe, D., & Goldstein, E. B. (2022). A reproducible and reusable pipeline for segmentation of geoscientific imagery. Earth and Space Science, 9, e2022EA002332. https://doi.org/10.1029/2022EA002332 
*  Buscombe, D., Goldstein, E. B., Sherwood, C. R., Bodine, C., Brown, J. A., Favela, J., et al. (2022). Human-in-the-loop segmentation of Earth surface imagery. Earth and Space Science, 9, e2021EA002085. https://doi.org/10.1029/2021EA002085 
* Sendrowski, A., Wohl, E., Hilton, R., Kramer, N., & Ascough, P. (2023). Wood-based carbon storage in the Mackenzie River Delta: The world's largest mapped riverine wood deposit. Geophysical Research Letters, 50, e2022GL100913. https://doi.org/10.1029/2022GL100913 
* Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M. and Luo, P., 2021. SegFormer: Simple and efficient design for semantic segmentation with transformers. Advances in Neural Information Processing Systems, 34, pp.12077-12090.

## Elwha project folder structure

* raw_data
	* dig_wood
	* GIS
	* time_series
	* wood_animation
	* DEMS
	* MR
		* MR_veg
		* MR_dev
		* MR_wood
		* MR_orthos_orig
	* LR
		* LR_veg
		* LR_dev
		* LR_wood
		* LR_orthos_orig
* regrid_data
	* MR
		* MR_veg
		* MR_dev
		* MR_wood
		* MR_orthos_orig
	* LR
		* LR_veg
		* LR_dev
		* LR_wood
		* LR_orthos_orig
* results
* Elwha_woodmapping (this folder)
	* gdal_scripts 
	
`gdal_scripts` should be copied into the `raw_data>>MR`, `raw_data>>LR` `results>>MR` and `results>>LR` nested folder structures
