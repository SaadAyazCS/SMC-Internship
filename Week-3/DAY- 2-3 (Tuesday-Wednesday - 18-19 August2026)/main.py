from data_loader import load_dataset, show_basic_info
from preprocessing_pipeline import PreprocessingPipeline
from eda import EDA
from data_quality_report import DataQualityReport



print("==============================")
print("PANDAS DATA PREPROCESSING")
print("==============================")


df = load_dataset(
    "datasets/students.csv"
)


show_basic_info(df)



pipeline = PreprocessingPipeline()


clean_data = pipeline.process(df)


print("\nClean Dataset")
print(clean_data)



eda = EDA()


eda.summary(clean_data)



eda.filtering(clean_data)



report = DataQualityReport()


report.generate(clean_data)