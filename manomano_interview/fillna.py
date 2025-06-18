import pandas as pd
from tqdm import tqdm


class FillNan:
    def __init__(self, label_columns, count_columns, data):
        self.data = data.copy()

        self.label_columns = label_columns
        self.count_columns = count_columns

        # get the rows with nan values :
        self.rows_with_nan = self._find_indexes_with_nan_val()

    def _find_indexes_with_nan_val(self):
        """
        Find indices with nan values.
        """
        rows_with_nan = []
        for index, row in self.data.iterrows():
            is_nan_series = row.isnull()
            if is_nan_series.any():
                rows_with_nan.append(index)
        return rows_with_nan

    def _fill_labels(self):
        """
        Fill label_* columns with nan values
        """

        for label in tqdm(self.label_columns):
            for row in self.rows_with_nan:
                if label.replace("label_", "") in self.data.description.loc[row]:
                    self.data[label].loc[row] = 1
                else:
                    self.data[label].loc[row] = 0

    def _fill_counts(self):
        """
        Fill count_* columns with nan values
        """

        def get_count(label, description):
            description = " ".join(
                [d for d in description.split(" ") if d not in ["", " ", None]]
            )
            for i, word in enumerate(description.split(" ")):
                if label.lower() in word.lower():
                    if i == 0:
                        return 1
                    try:
                        return float(description.split(" ")[i - 1].replace(",", "."))
                    except Exception:
                        return 1
            return 0

        for count in tqdm(self.count_columns):
            for row in self.rows_with_nan:
                if count.replace("count_", "") not in self.data.description.loc[
                    row
                ].replace("é", "e"):
                    self.data[count].loc[row] = 0
                else:
                    self.data[count].loc[row] = get_count(
                        count.replace("count_", ""),
                        self.data["description"].loc[row].replace("é", "e"),
                    )

    def fillna(self) -> pd.DataFrame:
        # Fill label_* columns with nan values
        self._fill_labels()

        # Fill count_* columns with nan values
        self._fill_counts()

        # remove y when we apply fillna to testset:
        return self.data
