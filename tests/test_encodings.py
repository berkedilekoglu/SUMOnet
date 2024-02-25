import numpy as np

from sumonet.utils.encodings import Encoding


def test_blosum62_encoding():
    encoder = Encoding(encoderType='blosum62')
    sequence = ["AAA"]
    encoder.bl_encoder(sequence)
    encoded_array = encoder.get_sequence()
    print(encoded_array)
    expected_array = np.array([[ 4., -1., -2., -2.,  0., -1., -1.,  0., -2., -1., -1., -1., -1.,
        -2., -1.,  1.,  0., -3., -2.,  0., -2., -1.,  0., -4.,  4., -1.,
        -2., -2.,  0., -1., -1.,  0., -2., -1., -1., -1., -1., -2., -1.,
         1.,  0., -3., -2.,  0., -2., -1.,  0., -4.,  4., -1., -2., -2.,
         0., -1., -1.,  0., -2., -1., -1., -1., -1., -2., -1.,  1.,  0.,
        -3., -2.,  0., -2., -1.,  0., -4.]], dtype=np.float32)
    assert np.array_equal(encoded_array, expected_array)