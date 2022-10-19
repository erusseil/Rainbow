The MultiBand Fit (MfB) method is an astrophysical transient feature extraction method. 
The usual way to extract features from a transient object is to fit it's lightcurves in each passband with a function. 
The fitted parameter value sare then used as features, ensuring the same number of parameters for all objects.


A function commonly used to describe transient events is the Bazin function, which requieres 5 free parameters : https://arxiv.org/pdf/1701.05689.pdf.
It needs to be applied to each telescope filter, which in the case of LSST (https://www.lsst.org/) will be 6. 
That feature extraction method results in too many features, with too many correlations.

The MultiBand Fit (MfB) method aims at providing a single fit in all passband at the same time for a given physical object, rather than fitting each passband individually.
In this case we will be fitting a big physically motivated 8 parameters function that should describe the luminosity at any time at any wavelength.
ADD MATH DETAILS

In order to evaluate and compare the standard method to the MfB we will be using the ELASTiCC data set (https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/).
It is a simulation of future LSST data and should be suited for the test. The data used can be found here : https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/TRAINING_SAMPLES/.

In order to run a feature extraction using either the "bazin" method or the "mfb" method you should first specify the path of your data in the kernel.py file.

To start the feature extraction run :

```python
  python data_processing.py 'target' max_number_of_objects 'method' number_of_core_to_use 'has_data_been_preprocessed_already'
```
  
For example if I want to feature extract for the first time at most 10000 objects of the class "PISN" using 10 cores with the mfb method :
```python
  python data_processing.py 'PISN' 10000 'mfb' 10 'False'
```
