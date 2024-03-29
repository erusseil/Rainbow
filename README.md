Code associated to the Rainbow paper : https://arxiv.org/abs/2310.02916

![3D_YSE(1)](https://github.com/erusseil/Rainbow/assets/79919110/b6fdb9a4-3089-4982-8c8d-e8ce5a2601d8)

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
