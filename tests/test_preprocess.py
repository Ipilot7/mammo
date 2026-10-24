import numpy as np
import pytest

from mammo.labels import transform_box
from mammo.preprocess import dicom_to_gray


def test_transform_box_maps_and_clips_to_converted_image():
    box = transform_box(
        xmin=5,
        ymin=15,
        xmax=70,
        ymax=60,
        x0=10,
        y0=10,
        scale=2,
        width=100,
        height=80,
    )

    assert box == pytest.approx((0, 10, 100, 80))


def test_monochrome1_is_inverse_of_monochrome2_except_rounding():
    class Dataset:
        pixel_array = np.array([[0, 25], [50, 100]], dtype=np.uint16)
        RescaleSlope = 1
        RescaleIntercept = 0
        WindowCenter = 50
        WindowWidth = 101
        PixelPaddingValue = None
        PhotometricInterpretation = "MONOCHROME2"

    mono2 = dicom_to_gray(Dataset())
    dataset = Dataset()
    dataset.PhotometricInterpretation = "MONOCHROME1"
    mono1 = dicom_to_gray(dataset)

    assert np.abs(mono1.astype(int) + mono2.astype(int) - 255).max() <= 1
