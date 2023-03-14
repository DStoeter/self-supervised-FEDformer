# self-supervised-FEDformer
Self-supervised transformer for LTSF based on the [FEDformer](https://github.com/MAZiqing/FEDformer) .

Copy the datasets you want to work with into the dataset folder.
Datasets: [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing).

Change the masking parameters manually in the FEDformer model and choose between fix and random masking.

## Get Started:

Alternative A:
Use the notebook [masking](https://github.com/DJFKO/self-supervised-FEDformer/blob/main/masking.ipynb) and change the parsed arguments manually. Default dataset is traffic with univariate prediction.

Alternatvie B:
Use the scripts for univariate or multivariate forecasting.

bash ./scripts/run_M.sh

bash ./scripts/run_S.sh
