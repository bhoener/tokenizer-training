# Tokenization and Training Improvements

## [[Read the Blogpost Here!]](https://bhoener.github.io/posts/Tokenization-and-Training-Improvements/)


A repo containing tokenizer training code in java and a simple pytorch LM training setup.


## Usage
First clone the repository, `cd env` to enter the environment folder.

run `Scripts/activate.ps1` in the console (if using powershell) to activate the virtual environment.
To install dependencies, run `pip install -r requirements.txt`

To train a tokenizer, first download and prepare the data.
```shell
python3 src/training/download_data.py
python3 src/training/download_data_sft.py
python3 src/training/generate_tokenizer_train_samples.py
```

Then run the java script to train the tokenizer:
```shell
java ./src/tokenizer/TokenizerTrain.java
```

Finally, encode all the files with the trained tokenizer:

```shell
java ./src/tokenizer/EncodeFiles.java
```

To train the model:

```
python3 src/training/train.py --config_file <your_config_here>
```

(config sample for training can be found at `env/src/training/configs/sample_train.yaml`)

And for inference:

```
python3 src/training/inference.py --config_file <your_config_here>
```