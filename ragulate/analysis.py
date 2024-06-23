from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.io import write_image
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .utils import get_tru


class Analysis:

    def get_all_data(self, recipes: List[str]) -> DataFrame:
        df_all = pd.DataFrame()

        all_metrics: List[str] = []

        for recipe in recipes:
            tru = get_tru(recipe_name=recipe)

            for app in tru.get_apps():
                dataset = app["app_id"]
                df, metrics = tru.get_records_and_feedback([dataset])
                all_metrics.extend(metrics)

                columns_to_keep = metrics + [
                    "record_id",
                    "latency",
                    "total_tokens",
                    "total_cost",
                ]
                columns_to_drop = [
                    col for col in df.columns if col not in columns_to_keep
                ]

                df.drop(columns=columns_to_drop, inplace=True)
                df["recipe"] = recipe
                df["dataset"] = dataset

                # set negative values to None
                for metric in metrics:
                    df.loc[df[metric] < 0, metric] = None

                df_all = pd.concat([df_all, df], axis=0, ignore_index=True)

            tru.delete_singleton()

        df_all.reset_index(drop=True, inplace=True)

        return df_all, list(set(all_metrics))

    def output_plots_by_dataset(self, df: DataFrame, metrics: List[str]):
        # Melt the DataFrame to long format
        df_melted = pd.melt(df, id_vars=['record_id', 'recipe', 'dataset'],
                    value_vars=['answer_correctness', 'answer_relevance', 'context_relevance', 'groundedness', 'latency'],
                    var_name='metric', value_name='value')

        # Set the theme for the plot
        sns.set_theme(style="darkgrid")

        # Custom function to set bin ranges
        def custom_hist(data, **kws):
            metric = data['metric'].iloc[0]
            discrete = metric != "latency"
            bin_range = (0,1) if discrete else None
            sns.histplot(data, x='value', stat='percent', bins=10, binrange=bin_range, **kws)

        # Create the FacetGrid
        g = sns.FacetGrid(df_melted, col="metric", row="recipe", margin_titles=True, height=3, aspect=1, sharex="col")

        # Map the custom histogram function to the FacetGrid
        g.map_dataframe(custom_hist)

        for ax in g.axes.flat:
            ax.set_ylim(0, 100)

        g.set_axis_labels("Value", "Percentage")

        # Save the plot as a PNG file
        g.savefig("grid_of_histograms.png")

        # Close the plot to avoid displaying it
        plt.close()


    def compare(self, recipes: List[str]):
        df, metrics = self.get_all_data(recipes=recipes)
        self.output_plots_by_dataset(df=df, metrics=metrics)
