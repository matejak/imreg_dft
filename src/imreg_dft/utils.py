import numpy as np
import numpy.fft as fft


def filter(img, lows=None, highs=None):
    """
    Given an image, it applies a list of high-pass and low-pass filters on its
    Fourier spectrum.
    """
    if lows is None:
        lows = []
    if highs is None:
        highs = []

    dft = fft.fft2(img)
    for spec in highs:
        highpass(dft, spec[0], spec[1])
    for spec in lows:
        lowpass(dft, spec[0], spec[1])
    ret = np.real(fft.ifft2(dft))
    return ret


def highpass(dft, lo, hi):
    mask = _xpass((dft.shape), lo, hi)
    dft *= (1 - mask)


def lowpass(dft, lo, hi):
    mask = _xpass((dft.shape), lo, hi)
    dft *= mask


def _xpass(shape, lo, hi):
    """
    Computer a pass-filter mask with values ranging from 0 to 1.0
    The mask is low-pass, application has to be handled by a calling funcion.
    """
    assert lo < hi, \
        "Filter order wrong, low '%g', high '%g'" % (lo, hi)
    assert lo > 0, \
        "Low filter lower than zero (%g)" % lo
    # High can be as high as possible

    dom_x = np.fft.fftfreq(shape[0])[:, np.newaxis]
    dom_y = np.fft.fftfreq(shape[1])[np.newaxis, :]

    # freq goes 0..0.5, we want from 0..1, so we multiply it by 2.
    dom = np.sqrt(dom_x ** 2 + dom_y ** 2) * 2

    res = np.ones(dom.shape)
    res[dom >= hi] = 0.0
    mask = (dom > lo) * (dom < hi)
    res[mask] = 1 - (dom[mask] - lo) / (hi - lo)

    return res


def get_apofield(shape, aporad):
    apos = np.hanning(aporad * 2)
    vecs = []
    for dim in shape:
        assert dim > aporad * 2
        toapp = np.ones(dim)
        toapp[:aporad] = apos[:aporad]
        toapp[-aporad:] = apos[-aporad:]
        vecs.append(toapp)
    apofield = np.outer(vecs[0], vecs[1])
    return apofield
