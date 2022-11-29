# self-supervised-FEDformer
Self-supervised transformer for LTSF based on the [FEDformer](https://github.com/MAZiqing/FEDformer) .

Copy the datasets you want to work with into the dataset folder.
Datasets: [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing).

First masking approach is in the exp/exp_main.py

## Get Started:

Alternative A:
Use the notebook masking and change the parsed arguments manually. Default dataset is traffic with univariate prediction.

Alternatvie B:
Use the scripts for univariate or multivariate forecasting.

bash ./scripts/run_M.sh

bash ./scripts/run_S.sh
