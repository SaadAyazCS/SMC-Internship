import pandas as pd


class DataCleaner:


    def handle_missing_values(self, df):

        print("\nMissing Values Before:")
        print(df.isnull().sum())

        df = df.fillna(df.mean(numeric_only=True))

        print("\nMissing Values After:")
        print(df.isnull().sum())

        return df



    def remove_duplicates(self, df):

        print("\nDuplicates Before:")
        print(df.duplicated().sum())

        df = df.drop_duplicates()

        print("Duplicates After:")
        print(df.duplicated().sum())

        return df



    def detect_outliers(self, df):

        numeric_columns = df.select_dtypes(
            include=['int64','float64']
        )

        for column in numeric_columns:

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            IQR = Q3-Q1


            outliers = df[
                (df[column] < Q1-1.5*IQR) |
                (df[column] > Q3+1.5*IQR)
            ]


            print(
                f"\n{column} Outliers:"
            )

            print(outliers)

        return df