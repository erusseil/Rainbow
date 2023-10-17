Code associated to the Rainbow paper : https://arxiv.org/abs/2310.02916

![3D_YSE(1)|50%](https://github.com/erusseil/Rainbow/assets/79919110/050acdfa-2087-4d46-9a1d-5316c60f3338)

The PLAsTiCC data is available at : https://zenodo.org/record/2539456.

The YSE data is available here : https://zenodo.org/records/7317476.

In order to run a feature extraction you should first adapt the path of your data in the kernel.py file.
To start a full the feature extraction run :

```python
  python data_processing.py --target --nmax --cores --database --field --band_wavelength
```
For example the feature extraction command used to process the PLAsTiCC's SNIa as described within the Rainbow paper is :

```python
  python data_processing.py --target='YSE_SNIa' --nmax=1000 --cores=3  --database='YSE' --field='wfd' --band_wavelength='integrate' 
```
