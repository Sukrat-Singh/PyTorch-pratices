import numpy as np
from visdom import Visdom


class VisdomHelper:
    """
    A simple utility class for visualizing training progress and logs
    using the Visdom dashboard.
    """

    def __init__(self, server=None, port=8097):
        """
        Initialize a Visdom session and a dictionary to track created plots.

        Args:
            server (str, optional): Visdom server address. Defaults to localhost.
            port (int, optional): Port number of the Visdom server. Defaults to 8097.
        """
        self.viz = Visdom(server=server, port=port)
        self.windows = {}

    def plot_scalar(self, y_value, x_step, title="Metric", xlabel="Step", ylabel="Value"):
        """
        Create or update a line plot for scalar values like loss or accuracy.

        Args:
            y_value (float): The scalar value to plot.
            x_step (int): The corresponding x-axis step (e.g., epoch or iteration).
            title (str): Plot title (used as unique identifier for each plot).
            xlabel (str): Label for the X-axis.
            ylabel (str): Label for the Y-axis.
        """
        # Convert to NumPy arrays (Visdom requires array-like data)
        x_data = np.array([x_step], dtype=np.int64)
        y_data = np.array([y_value], dtype=np.float32)

        # Create a new plot if it doesn't exist
        if title not in self.windows:
            win = self.viz.line(
                X=x_data,
                Y=y_data,
                opts=dict(
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                ),
            )
            self.windows[title] = win
        else:
            # Append new data to the existing plot
            self.viz.line(
                X=x_data,
                Y=y_data,
                win=self.windows[title],
                update="append",
            )

    def show_text(self, message, title="Log"):
        """
        Display or update a text window in Visdom.

        Args:
            message (str): The text to display.
            title (str): The name of the text window.
        """
        # If a text window with this title already exists, overwrite it
        if title in self.windows:
            self.viz.text(message, win=self.windows[title], opts=dict(title=title))
        else:
            win = self.viz.text(message, opts=dict(title=title))
            self.windows[title] = win
