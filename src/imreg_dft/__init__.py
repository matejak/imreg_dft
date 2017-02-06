# -*- coding: utf-8 -*-

# We may import this during setup invocation
# because of the version we have to query
# However, i.e. numpy may not be installed at setup install time.
try:
    from imreg_dft.imreg import (translation, similarity, transform_img,
                                 transform_img_dict, imshow)
except ImportError as exc:
    print("Unable to import the main package: %s" % exc)


__version__ = "2.0.1a"
