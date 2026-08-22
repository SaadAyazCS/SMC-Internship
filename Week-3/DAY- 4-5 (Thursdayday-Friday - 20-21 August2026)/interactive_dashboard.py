import pandas as pd
import plotly.express as px


class Dashboard:


    def __init__(self,file):

        self.data=pd.read_csv(file)



    def create_dashboard(self):


        fig1 = px.scatter(
            self.data,
            x="Experience",
            y="Salary",
            color="Department",
            title="Experience vs Salary"
        )


        fig2 = px.bar(
            self.data,
            x="Department",
            y="Salary",
            title="Department Salary"
        )


        fig3 = px.histogram(
            self.data,
            x="Salary",
            title="Salary Distribution"
        )


        fig1.write_html(
            "dashboard/scatter.html"
        )

        fig2.write_html(
            "dashboard/bar.html"
        )

        fig3.write_html(
            "dashboard/histogram.html"
        )