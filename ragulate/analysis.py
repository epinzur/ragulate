from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.io import write_image

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
        recipes = sorted(df["recipe"].unique(), key=lambda x: x.lower())
        datasets = sorted(df["dataset"].unique(), key=lambda x: x.lower())

        # generate an array of rainbow colors by fixing the saturation and lightness of the HSL
        # representation of color and marching around the hue.
        c = [
            "hsl(" + str(h) + ",50%" + ",50%)"
            for h in np.linspace(0, 360, len(recipes) + 1)
        ]

        height = max((len(metrics) * len(recipes) * 20) + 150, 450)

        for dataset in datasets:
            fig = go.Figure()
            test_index = 0
            for recipe in recipes:
                y = []
                x = []
                for metric in metrics:
                    dx = df[metric][df["recipe"] == recipe][df["dataset"] == dataset]
                    x.extend(dx)
                    y.extend([metric] * len(dx))

                fig.add_trace(
                    go.Box(
                        y=y,
                        x=x,
                        name=recipe,
                        marker_color=c[test_index],
                        visible=True,
                    )
                )
                test_index += 1

            fig.update_traces(
                orientation="h",
                boxmean=True,
                jitter=1,
            )
            fig.update_layout(boxmode="group", height=height, width=900)
            fig.update_layout(
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                )
            )
            fig.update_layout(yaxis_title="metric", xaxis_title="score")

            write_image(fig, f"./{dataset}.png")

    def compare(self, recipes: List[str]):
        df, metrics = self.get_all_data(recipes=recipes)
        self.output_plots_by_dataset(df=df, metrics=metrics)
