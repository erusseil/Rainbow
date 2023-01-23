The RAINBOW method is an astrophysical transient feature extraction method. 
The usual way to extract features from a transient object is to fit it's lightcurves in each passband with a function. 
The fitted parameter value sare then used as features, ensuring the same number of parameters for all objects.


A function commonly used to describe transient events is the Bazin function, which requires 4 free parameters : https://arxiv.org/pdf/1701.05689.pdf.
It needs to be applied to each telescope filter, which in the case of LSST (https://www.lsst.org/) will be 6. 
That feature extraction method results in too many features, with too many correlations.

The RAINBOW method aims at providing a single fit in all passband at the same time for a given physical object, rather than fitting each passband individually.
In this case we will be fitting a big physically motivated 7 parameters function that should describe the luminosity at any time at any wavelength.
ADD MATH DETAILS

In order to evaluate and compare the standard method to the RAINBOW we will be using the PLAsTiCC data set (https://plasticc.org/).
It is a simulation of future LSST data and should be suited for the test. The data used can be found here : https://zenodo.org/record/2539456

In order to run a feature extraction you should first specify the path of your data in the kernel.py file.

To start a full the feature extraction run :

```python
  python data_processing.py --target --nmax --cores --database --field
```
