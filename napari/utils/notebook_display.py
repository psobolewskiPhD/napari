import base64
import html
from io import BytesIO

import nh3

from napari.utils.io import imsave_png

__all__ = ['nbscreenshot']


class NotebookScreenshot:
    """Display napari screenshot in the jupyter notebook.

    Functions returning an object with a _repr_png_() method
    will displayed as a rich image in the jupyter notebook.

    https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer.
    canvas_only : bool, optional
        If True includes the napari viewer frame in the screenshot,
        otherwise just includes the canvas. By default, True.

    Examples
    --------
    ```
    import napari
    from napari.utils import nbscreenshot
    from skimage.data import chelsea

    viewer = napari.view_image(chelsea(), name='chelsea-the-cat')
    nbscreenshot(viewer)

    # screenshot just the canvas with the napari viewer framing it
    nbscreenshot(viewer, canvas_only=False)
    ```
    """

    def __init__(
        self,
        viewer,
        *,
        canvas_only=False,
        alt_text=None,
    ) -> None:
        """Initialize screenshot object.

        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer
        canvas_only : bool, optional
            If False include the napari viewer frame in the screenshot,
            and if True then take screenshot of just the image display canvas.
            By default, False.
        alt_text : str, optional
            Image description alternative text, for screenreader accessibility.
            Good alt-text describes the image and any text within the image
            in no more than three short, complete sentences.
        """
        self.viewer = viewer
        self.canvas_only = canvas_only
        self.image = None
        self.alt_text = self._clean_alt_text(alt_text)

    def _clean_alt_text(self, alt_text):
        """Clean user input to prevent script injection."""
        if alt_text is not None:
            # nh3 won't recognize escaped tags, so always unescape
            alt_text = html.unescape(str(alt_text))
            # sanitize html and remove all tags
            alt_text = nh3.clean(alt_text, tags=set())
            # disallow empty strings or only whitespace
            if alt_text == '' or alt_text.isspace():
                alt_text = None
        return alt_text

    def _repr_png_(self):
        """PNG representation of the viewer object for IPython.

        Returns
        -------
        In memory binary stream containing PNG screenshot image.
        """
        from napari._qt.qt_event_loop import get_app

        get_app().processEvents()
        self.image = self.viewer.screenshot(
            canvas_only=self.canvas_only, flash=False
        )
        with BytesIO() as file_obj:
            imsave_png(file_obj, self.image)
            file_obj.seek(0)
            png = file_obj.read()
        return png

    def _repr_html_(self):
        png = self._repr_png_()
        url = 'data:image/png;base64,' + base64.b64encode(png).decode('utf-8')
        _alt = html.escape(self.alt_text) if self.alt_text is not None else ''
        return f'<img src="{url}" alt="{_alt}"></img>'


nbscreenshot = NotebookScreenshot
