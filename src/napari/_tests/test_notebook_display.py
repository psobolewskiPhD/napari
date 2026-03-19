import html
from unittest.mock import Mock

import numpy as np
import pytest

from napari._tests.utils import skip_on_win_ci
from napari._version import __version__
from napari.utils import nbscreenshot


@skip_on_win_ci
def test_nbscreenshot(make_napari_viewer):
    """Test taking a screenshot."""
    viewer = make_napari_viewer()

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)

    rich_display_object = nbscreenshot(viewer)
    assert hasattr(rich_display_object, '_repr_png_')
    # Trigger method that would run in jupyter notebook cell automatically
    png_bytes = rich_display_object._repr_png_()
    assert rich_display_object.image is not None
    # Test digital watermark is included in bytes of .png file
    version_byte_string = __version__.encode('utf-8')
    assert b'napari version' in png_bytes
    assert version_byte_string in png_bytes


@skip_on_win_ci
@pytest.mark.parametrize(
    ('alt_text_input', 'expected_alt_text'),
    [
        (None, None),
        ('Good alt text', 'Good alt text'),
        # Naughty strings https://github.com/minimaxir/big-list-of-naughty-strings
        # ASCII punctuation
        (r",./;'[]\-=", ",./;'[]\\-="),
        # ASCII punctuation 2, skipping < because that is interpreted as the start
        # of an HTML element.
        ('>?:"{}|_+', '&gt;?:"{}|_+'),
        ('!@#$%^&*()`~', '!@#$%^&amp;*()`~'),  # ASCII punctuation 3
        # # Emojis
        ('😍', '😍'),  # emoji 1
        (
            '👨‍🦰 👨🏿‍🦰 👨‍🦱 👨🏿‍🦱 🦹🏿‍♂️',
            '👨‍🦰 👨🏿‍🦰 👨‍🦱 👨🏿‍🦱 🦹🏿‍♂️',
        ),  # emoji 2
        (r'¯\_(ツ)_/¯', '¯\\_(ツ)_/¯'),  # Japanese emoticon
        # # Special characters
        (
            '田中さんにあげて下さい',
            '田中さんにあげて下さい',
        ),  # two-byte characters
        (
            '表ポあA鷗ŒéＢ逍Üßªąñ丂㐀𠀀',  # noqa: RUF001
            '表ポあA鷗ŒéＢ逍Üßªąñ丂㐀𠀀',  # noqa: RUF001
        ),  # special unicode chars
        ('گچپژ', 'گچپژ'),  # Persian special characters
        # # Script injection
        ('<script>alert(0)</script>', None),  # script injection 1
        ('&lt;script&gt;alert(&#39;1&#39;);&lt;/script&gt;', None),
        ('<svg><script>123<1>alert(3)</script>', None),
    ],
)
def test_safe_alt_text(alt_text_input, expected_alt_text):
    display_obj = nbscreenshot(Mock(), alt_text=alt_text_input)
    if not expected_alt_text:
        assert not display_obj.alt_text
    else:
        # pytest escapes strings by default, including quotes
        # nh3.clean does not escape quotes
        assert html.unescape(display_obj.alt_text) == html.unescape(
            expected_alt_text
        )


def test_invalid_alt_text():
    display_obj = nbscreenshot(Mock(), alt_text=' ')
    assert display_obj.alt_text is None

    display_obj = nbscreenshot(Mock(), alt_text='')
    assert display_obj.alt_text is None
