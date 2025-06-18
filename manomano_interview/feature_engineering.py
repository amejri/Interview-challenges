from typing import List

import pandas as pd


class FeatureEngineering:
    def __init__(self, data: pd.DataFrame, categ_cols: List[str]) -> None:
        self.data = data.copy()
        self.categorical_cols = categ_cols

    # Extract year for both marketing declation and marketing authorization
    def _split_date(self) -> None:
        self.data["marketing_authorization_year"] = (
            self.data["marketing_authorization_date"] // 10000
        )
        self.data["marketing_declaration_year"] = (
            self.data["marketing_declaration_date"] // 10000
        )
        self.data.drop(columns=["marketing_authorization_date"], inplace=True)
        self.data.drop(columns=["marketing_declaration_date"], inplace=True)

    # Convert to datetime and compute the duration column:
    def _compute_period_of_validity(self) -> None:
        self.data["duration"] = (
            self.data["marketing_declaration_year"]
            - self.data["marketing_authorization_year"]
        )

    # Add other columns based on description exploration:
    #     - is_thermoformee
    #     - is pvc
    #     - is aluminium
    #     - is verre
    #     - is PVDC

    def _is_thermoformee(self) -> None:
        self.data["is_thermoforme"] = self.data["description"].apply(
            lambda x: 1 if "thermoformé" in x.lower() else 0
        )

    def _is_pvc(self) -> None:
        self.data["is_pvc"] = self.data["description"].apply(
            lambda x: 1 if "pvc" in x.lower() else 0
        )

    def _is_aluminium(self) -> None:
        self.data["is_aluminium"] = self.data["description"].apply(
            lambda x: 1 if "aluminium" in x.lower() else 0
        )

    def _is_pvdc(self) -> None:
        self.data["is_pvdc"] = self.data["description"].apply(
            lambda x: 1 if "pvdc" in x.lower() else 0
        )

    def _is_verre(self) -> None:
        self.data["is_verre"] = self.data["description"].apply(
            lambda x: 1 if "pvc" in x.lower() else 0
        )

    def _process_categorical_data(self) -> None:
        self.data = pd.get_dummies(
            self.data, columns=self.categorical_cols, drop_first=True
        )

    def transform(self) -> pd.DataFrame:
        """
        Apply all transformations
        """
        # Add date features :
        self._split_date()

        # Apply get dummies for categorical data
        self._process_categorical_data()

        # Compute validity period :
        self._compute_period_of_validity()

        # Add is thermoformee:
        self._is_thermoformee()

        # Add is PVC
        self._is_pvc()

        # Add is PVCD
        self._is_pvdc()

        # Add is verre
        self._is_verre()

        # Add is aluminium
        self._is_aluminium()

        # Drop description column
        self.data.drop(columns=["description"], inplace=True)
        return self.data
