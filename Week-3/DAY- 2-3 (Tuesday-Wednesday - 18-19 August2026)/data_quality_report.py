class DataQualityReport:


    def generate(self, df):

        report = {

            "Total Rows":
            len(df),

            "Total Columns":
            len(df.columns),

            "Missing Values":
            df.isnull().sum().sum(),

            "Duplicate Rows":
            df.duplicated().sum(),

            "Columns":
            list(df.columns)

        }


        print("\nData Quality Report")
        print("-------------------")


        for key,value in report.items():

            print(
                f"{key}: {value}"
            )


        return report