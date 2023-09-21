# pbrt-specular-microstructure-rendering

This repository provides an implementation in pbrt (v3) of the algorithm proposed in the paper "Position-normal distributions for efficient rendering of specular microstructure" by Yan et al. (2016) . It provides a class that converts normal maps to Gaussian mixtures and a Glitter Material class that overrides the normal distribution function **D(h)**, with **h** a normal vector, that plays a central role in the specular microfacet BRDF.

[An explanatory video and demo can be found here.](https://youtu.be/ojLtIwjEaLs?si=yVJqBnytwZdXaUxR)
