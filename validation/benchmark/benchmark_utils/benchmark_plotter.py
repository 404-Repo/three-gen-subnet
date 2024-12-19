from pathlib import Path

import numpy as np
import plotly.graph_objects as go


class Plotter:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_line_chart(input_data: list[np.ndarray], x_label: str, y_label: str, dataset_name: str) -> None:
        fig = go.Figure()

        for d in input_data:
            fig.add_trace(go.Scatter(x=np.arange(0, len(d)), y=d, mode="lines+markers", name=y_label))
            fig.update_layout(xaxis=dict(title=dict(text=x_label)), yaxis=dict(title=dict(text=y_label)))
            reply = input("Show interactive graph? [y/n]")
            if reply == "y":
                fig.show()

        image_file = Path("plots") / (dataset_name + "_" + y_label + ".png")
        fig.write_image(image_file, scale=2)
