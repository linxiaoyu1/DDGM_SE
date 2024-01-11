# DDGM_SE

To train the model in the noise-dependent way, run

`python train_nd.py`

To train the model in the noise-agnostic way, run first

`python ./src/data_preprocessing/_split_vp.py`

to generate the json files for the dataset, then run
`python train_na.py --json_file JSON_FILE_PATH`
