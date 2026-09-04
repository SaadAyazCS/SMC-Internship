from visualization import DataVisualizer
from statistical_visualization import StatisticalVisualizer
from interactive_dashboard import Dashboard
from eda_report import EDAReport



dataset="dataset/employee_data.csv"



print("Generating Matplotlib Visualizations...")

visualizer=DataVisualizer(dataset)

visualizer.generate_all()



print("Generating Seaborn Visualizations...")

stats=StatisticalVisualizer(dataset)

stats.generate_all()



print("Creating Plotly Dashboard...")

dashboard=Dashboard(dataset)

dashboard.create_dashboard()



print("Generating EDA Report...")

report=EDAReport(dataset)

report.generate()



print("EDA Completed Successfully")