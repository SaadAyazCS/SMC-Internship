import pandas as pd
import matplotlib.pyplot as plt


class DataVisualizer:

    def __init__(self, file):
        self.data = pd.read_csv(file)


    def save_plot(self, name):
        plt.savefig(f"plots/{name}.png")
        plt.close()


    def line_chart(self):

        plt.figure(figsize=(8,5))

        plt.plot(
            self.data["Employee_ID"],
            self.data["Salary"]
        )

        plt.title("Salary Trend")
        plt.xlabel("Employee ID")
        plt.ylabel("Salary")

        self.save_plot("line_chart")


    def bar_chart(self):

        dept = self.data.groupby(
            "Department"
        )["Salary"].mean()

        dept.plot(kind="bar")

        plt.title("Average Salary By Department")

        self.save_plot("bar_chart")


    def histogram(self):

        plt.hist(
            self.data["Salary"],
            bins=5
        )

        plt.title("Salary Distribution")

        self.save_plot("histogram")


    def scatter_plot(self):

        plt.scatter(
            self.data["Experience"],
            self.data["Salary"]
        )

        plt.xlabel("Experience")
        plt.ylabel("Salary")

        plt.title("Experience vs Salary")

        self.save_plot("scatter_plot")


    def pie_chart(self):

        count = self.data["Department"].value_counts()

        plt.pie(
            count,
            labels=count.index,
            autopct="%1.1f%%"
        )

        plt.title("Department Distribution")

        self.save_plot("pie_chart")


    def box_plot(self):

        plt.boxplot(
            self.data["Salary"]
        )

        plt.title("Salary Outliers")

        self.save_plot("box_plot")


    def generate_all(self):

        self.line_chart()
        self.bar_chart()
        self.histogram()
        self.scatter_plot()
        self.pie_chart()
        self.box_plot()