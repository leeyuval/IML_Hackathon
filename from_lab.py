import numpy as np
import plotly.graph_objects as go

np.random.seed(1)
m = 250


def triangle():
    x = np.random.uniform(low=-5, high=5, size=(m, 2))
    return (x, np.array(x[:, 0] < x[:, 1], dtype=np.int)), "Linear Separation"


(X, y), title = triangle()

go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                           marker=dict(color=y, line=dict(color="black", width=1),
                                       ))],
          layout=go.Layout(title=rf"$\textbf{{(1) {title} Dataset}}$")).show()
