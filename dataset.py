import numpy as np
import pandas as pd


class LetorDataset():

    def __init__(self, path: str):
        df = pd.read_csv(path, sep=" ", header=None, engine="pyarrow")

        # Find the docid column if it exists, then drop all columns after it
        has_docid = False
        for col in reversed(df.columns):
            if df[col][0] == "#docid":
                col_int = int(str(col))
                df.drop(columns=[col, col_int + 1, *range(col_int +
                        3, len(df.columns))], inplace=True)
                has_docid = True
                break

        # Set the column names
        col_names = ["relevance", "qid"]
        for i in range(2, len(df.columns)):
            if ":" in str(df[df.columns[i]][0]):
                col_names.append(df[i][0].split(":")[0])
        if has_docid:
            col_names.append("docid")
        df = df.set_axis(col_names, axis=1, copy=False)

        # Remove all occurrence of colons in the data
        df = df.applymap(lambda s: s.split(":")[1] if ":" in str(s) else s)

        # Update column types
        df["relevance"] = df["relevance"].astype("int")
        df["qid"] = df["qid"].astype("string")
        df["docid"] = df["docid"].astype("string")
        feature_cols: list[str] = []
        for col in df.columns:
            if not (col in ["relevance", "qid", "docid"]):
                df[col] = pd.to_numeric(df[col])
                feature_cols.append(str(col))

        # Set instance variables
        self.df = df
        self.qids = df["qid"].unique().tolist()
        self.feature_cols = feature_cols
        self.X = np.array(df[feature_cols])
        self.Y = np.array(df["relevance"])

        # Create a map from each qid to its corresponding features and labels
        self._qid_to_data = df.groupby(
            "qid")[[*feature_cols, "relevance"]].agg(list)

        # Set up data for docid
        if has_docid:
            self.docids = df["docid"].unique().tolist()
            self._docid_to_qid = df.groupby("docid")["qid"].agg(list)

    def get_data_for_qid(self, qid: str, *args) -> tuple[np.ndarray, np.ndarray]:
        """Return a tuple of training data and label (relevance) in numpy format
        for a given query id(s).
        """
        if len(args) == 0:
            return (
                np.stack(self._qid_to_data.loc[qid].drop("relevance").to_list(), axis=1),
                np.array(self._qid_to_data.loc[qid]["relevance"])
            )
        else:
            qids = [qid] + list(args)
            return (
                np.array(self._qid_to_data.loc[qids].drop(
                    columns=["relevance"]).explode(column=tuple(self.feature_cols)), dtype="float"),
                np.array(self._qid_to_data.loc[qids]
                         ["relevance"].explode(), dtype="int")
            )

    def get_qids_for_docid(self, docid: str) -> list[str]:
        """Return a list of query ids that result in the given doc id
        """
        return self._docid_to_qid.loc[docid]
    
    def get_num_features(self) -> int:
        return len(self.feature_cols)
