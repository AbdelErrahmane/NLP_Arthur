
import pandas as pd
from utils import *

model_params={
    "MODEL":"t5-base",             # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE":8,          # training batch size
    "VALID_BATCH_SIZE":8,          # validation batch size
    "TRAIN_EPOCHS":3,              # number of training epochs
    "VAL_EPOCHS":1,                # number of validation epochs
    "LEARNING_RATE":1e-4,          # learning rate
    "MAX_SOURCE_TEXT_LENGTH":512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH":50,   # max length of target text
    "SEED": 42                     # set seed for reproducibility

}

my_df= pd.read_csv('test_corpus.csv')

my_df["s2t"] = "summarize: "+my_df["s2t"]

def main():
    print("Execution of th main function ...................")
    T5Trainer(dataframe=my_df[:500], source_text="s2t", target_text="original", model_params=model_params,
              output_dir="outputs")


if __name__ == "__main__":
    main()




