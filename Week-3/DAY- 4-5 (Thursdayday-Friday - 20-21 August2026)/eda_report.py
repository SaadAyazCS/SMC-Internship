import pandas as pd


class EDAReport:


    def __init__(self,file):

        self.data=pd.read_csv(file)



    def generate(self):

        with open(
            "EDA_Report.txt",
            "w"
        ) as file:


            file.write(
                "Exploratory Data Analysis Report\n"
            )

            file.write(
                "\nDataset Information\n"
            )


            file.write(
                str(self.data.info())
            )


            file.write(
                "\n\nStatistical Summary\n"
            )


            file.write(
                str(
                    self.data.describe()
                )
            )


            file.write(
                "\n\nMissing Values\n"
            )


            file.write(
                str(
                    self.data.isnull().sum()
                )
            ) 