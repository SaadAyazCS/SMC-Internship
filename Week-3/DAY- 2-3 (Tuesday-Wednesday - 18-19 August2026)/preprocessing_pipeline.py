from data_cleaning import DataCleaner


class PreprocessingPipeline:


    def __init__(self):

        self.cleaner = DataCleaner()



    def process(self, dataframe):

        dataframe = self.cleaner.handle_missing_values(
            dataframe
        )


        dataframe = self.cleaner.remove_duplicates(
            dataframe
        )


        self.cleaner.detect_outliers(
            dataframe
        )


        return dataframe