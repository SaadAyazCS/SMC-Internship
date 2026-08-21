import pandas as pd


class EDA:


    def summary(self, df):

        print("\nStatistical Summary")
        print("-------------------")

        print(
            df.describe()
        )


    def grouping(self, df, column):

        print(
            "\nGroup Analysis"
        )

        print(
            df.groupby(column).mean(numeric_only=True)
        )


    def filtering(self, df):

        print(
            "\nFiltered Data"
        )

        print(
            df[df.select_dtypes(
                include=['number']
            ).iloc[:,0] > 20]
        )