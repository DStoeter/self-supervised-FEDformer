# self-supervised-FEDformer
Self-supervised transformer for LTSF based on the [FEDformer](https://github.com/MAZiqing/FEDformer) .

Copy the datasets you want to work with into the dataset folder.
Datasets: [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing).

Change the masking parameters manually in the FEDformer.py and choose between fix and random masking of the encoder and the decoder input.

## Get Started:

Alternative A:
Use the notebook [masking](https://github.com/DJFKO/self-supervised-FEDformer/blob/main/masking.ipynb) and change the parsed arguments manually.

Alternatvie B:
Execute "run.py" in a terminal and adjust the parsed agruments like sequence length and prediction length manually beforehand. The default dataset is Illness with a sequence length of 36 and a prediction length of 60.

Alternatvie C:
Use the scripts for univariate or multivariate forecasting.

bash ./scripts/run_M.sh

bash ./scripts/run_S.sh
