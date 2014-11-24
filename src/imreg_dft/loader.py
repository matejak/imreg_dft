import imreg_dft


class Loader(object):
    def __init__(self, order, priority):
        self.loaded = None
        self.order = order
        self.priority = priority

    def guessCanLoad(self, fname):
        """
        Guess whether we can load a filename just according to the name
        (extension)
        """
        pass

    def load2reg(self, fname, opts=None):
        """
        Given a filename, it loads it and returns in a form suitable for
        registration (i.e. float, flattened, ...).
        """
        if opts is None:
            opts = {}
        return self._load2reg

    def _load2reg(self, fname, opts):
        raise NotImplementedError("Use the derived class")

    def _save(self, fname, data, opts):
        raise NotImplementedError("Use the derived class")

    def save(self, fname, tform_dict, save_opts):
        """
        Given the registration result, save the transformed input.
        """
        assert self.loaded is not None, \
            "lalala"
        tformed = imreg_dft.transform_img_dict(self.loaded, tform_dict,
                                               order=self.order)
        if save_opts is None:
            save_opts = {}
        self._save(fname, tformed, save_opts)


class MatLoader(Loader):
    def __init__(self, order):
        super(MatLoader, self).__init__(order, 10)

    def _load2reg(self, fname, opts):
        from scipy import io
        mat = io.loadmat(fname)
        valid = [key for key in mat if not key.startswith("_")]
        if "input" not in opts:
            if len(valid) != 1:
                raise RuntimeError(
                    "You have to supply an input key, there is an ambiguity of "
                    "what to load")
            else:
                key = valid[0]
        else:
            key = opts["input"]
            assert key in valid
        ret = mat[key]
        self._loaded_all = mat
        self._key = key
        self.loaded = ret
        return ret

    def _save(self, fname, tformed, opts):
        from scipy import io
        if "output" not in opts:
            key = self._key
        else:
            key = opts["output"]
        out = self._loaded_all
        out[key] = tformed
        io.savemat(fname, out)

    def guessCanLoad(self, fname):
        return fname.endswith(".mat")


class PILLoader(Loader):
    def __init__(self, order):
        super(PILLoader, self).__init__(order, 50)

    def _load2reg(self, fname, opts):
        from scipy import misc
        loaded = misc.imread(fname)
        self.loaded = loaded
        ret = loaded.mean(axis=0)
        return ret

    def _save(self, fname, tformed, opts):
        from scipy import misc
        img = misc.toimage(tformed)
        img.save(fname)

    def guessCanLoad(self, fname):
        "We think that we can do everything"
        return True


class HDRLoader(Loader):
    def __init__(self, order):
        super(HDRLoader, self).__init__(order, 10)

    def guessCanLoad(self, fname):
        return fname.endswith(".hdr")

    def _load2reg(self, fname, opts):
        """Return image data from img&hdr uint8 files."""
        import numpy as np
        basename = fname.rstrip(".hdr")
        with open(basename + '.hdr', 'r') as fh:
            hdr = fh.readlines()
        img = np.fromfile(basename + '.img', np.uint8, -1)
        img.shape = int(hdr[4].split()[-1]), int(hdr[3].split()[-1])
        if opts.get("norm", True):
            img = img.astype(np.float64)
            img /= 255.0
        return img

    def _save(self, fname, tformed, opts):
        from scipy import misc
        img = misc.toimage(tformed)
        img.save(fname)
