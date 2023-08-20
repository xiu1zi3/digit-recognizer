kaggle digit-recognizer CNN implementation

[toc]

## Environment
python 3.9.12
Aug 20, 2023

## Usage
make sure your structure is as follows:



```
digit-recognizer.
├─LICENSE
├─README.md
├─data
│  ├─test.csv
│  └─train.csv
├─model
│  
├─output
│  
└─main.py
```
- dataset from [Kaggle Digit Recognizer challenge](https://www.kaggle.com/c/digit-recognizer) is stored in `data` directory.

Now run the `main.py`
Then 

```
digit-recognizer.
├─LICENSE
├─README.md
├─data
│  ├─test.csv
│  └─train.csv
├─model
│  └─CNN.pt
├─output
│  └─submission.csv
└─main.py
```

- The prediction is stored in `output` directory.
- The trianed model is stored in `output` directory.

You can ajust the hyper parameters. Or you can choose other loss function and optimizer.