class DataSplitter:

    def split(self, df):

        train_df = df[df["train_test"] == "train"].copy()

        prediction_df = df[df["train_test"] == "prediction"].copy()

        return train_df, prediction_df