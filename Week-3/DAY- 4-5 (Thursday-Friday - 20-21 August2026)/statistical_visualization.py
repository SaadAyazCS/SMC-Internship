import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class StatisticalVisualizer:


    def __init__(self,file):

        self.data=pd.read_csv(file)



    def save(self,name):

        plt.savefig(
            f"plots/{name}.png"
        )

        plt.close()



    def correlation_heatmap(self):

        plt.figure(figsize=(8,6))

        sns.heatmap(
            self.data.corr(numeric_only=True),
            annot=True
        )

        plt.title(
            "Correlation Heatmap"
        )

        self.save(
            "correlation_heatmap"
        )



    def salary_distribution(self):

        sns.histplot(
            self.data["Salary"],
            kde=True
        )

        plt.title(
            "Salary Distribution KDE"
        )

        self.save(
            "salary_distribution"
        )



    def department_salary(self):

        sns.boxplot(
            x="Department",
            y="Salary",
            data=self.data
        )

        plt.xticks(
            rotation=45
        )

        plt.title(
            "Department Salary Comparison"
        )

        self.save(
            "department_salary"
        )



    def pair_plot(self):

        sns.pairplot(
            self.data.select_dtypes(
                include="number"
            )
        )

        plt.savefig(
            "plots/pair_plot.png"
        )

        plt.close()



    def generate_all(self):

        self.correlation_heatmap()
        self.salary_distribution()
        self.department_salary()
        self.pair_plot()