import numpy as np

from napari.layers.shapes._shapes_models.shape import Shape
from napari.layers.shapes._shapes_utils import create_box
from napari.utils.translations import trans


class Line(Shape):
    """Class for a single line segment

    Parameters
    ----------
    data : (2, D) array
        Line vertices.
    edge_width : float
        thickness of lines and edges.
    z_index : int
        Specifier of z order priority. Shapes with higher z order are displayed
        ontop of others.
    dims_order : (D,) list
        Order that the dimensions are to be rendered in.
    """

    def __init__(
        self,
        data,
        *,
        edge_width=1,
        z_index=0,
        dims_order=None,
        ndisplay=2,
    ) -> None:
        super().__init__(
            edge_width=edge_width,
            z_index=z_index,
            dims_order=dims_order,
            ndisplay=ndisplay,
        )
        self._filled = False
        self.data = data
        self.name = 'line'

    @property
    def data(self):
        """(2, D) array: line vertices."""
        return self._data

    @data.setter
    def data(self, data):
        data = np.array(data).astype(np.float32)

        if len(self.dims_order) != data.shape[1]:
            self._dims_order = list(range(data.shape[1]))

        if len(data) != 2:
            raise ValueError(
                trans._(
                    'Data shape does not match a line. A line expects two end vertices, {number} provided.',
                    deferred=True,
                    number=len(data),
                )
            )

        self._data = data
        self._bounding_box = np.array(
            [
                np.min(data, axis=0),
                np.max(data, axis=0),
            ]
        )

        self._update_displayed_data()

    def _update_displayed_data(self) -> None:
        """Update the data that is to be displayed."""
        # For path connect every all data
        self._clean_cache()
        self._set_meshes(self.data_displayed, face=False, closed=False)
        # in this case we have only 2D data (based on docstring)
        self._box = create_box(self.data_displayed)  # type: ignore[arg-type]

        self.slice_key = np.round(
            self._bounding_box[:, self.dims_not_displayed]
        ).astype('int')
