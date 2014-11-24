class LoaderSet(object):
    LOADERS = []
    # singleton-like functionality
    we = None

    def __init__(self):
        if LoaderSet.we is not None:
            return LoaderSet.we
        loaders = [loader() for loader in LoaderSet.LOADERS]
        self.loader_dict = {}
        for loader in loaders:
            self.loader_dict[loader.name] = loader
        self.loaders = sorted(loaders, key=lambda x: x.priority)
        LoaderSet.we = self

    def choose_loader(self, fname):
        for loader in self.loaders:
            if loader.guessCanLoad(fname):
                return loader
        # Ouch, no loader available!
        return None

    def get_loader(self, lname):
        if lname not in self.loader_dict:
            msg = "No loader named '%s'." % lname
            msg += " Choose one of %s." % self.loader_dict.keys()
            raise KeyError(msg)
        return self.loader_dict(lname)


def loader(lname):
    def wrapped(cls):
        cls.name = lname
        LoaderSet.LOADERS.append(cls)
        return cls
    return wrapped


class Loader(object):
    name = None

    def __init__(self, priority):
        self.loaded = None
        self.priority = priority

    def guessCanLoad(self, fname):
        """
        Guess whether we can load a filename just according to the name
        (extension)
        """
        return False

    def load2reg(self, fname, opts=None):
        """
        Given a filename, it loads it and returns in a form suitable for
        registration (i.e. float, flattened, ...).
        """
        if opts is None:
            opts = {}
        return self._load2reg(fname, opts)

    def get2save(self):
        assert self.loaded is not None, \
            "lalala"
        return self.loaded

    def _load2reg(self, fname, opts):
        raise NotImplementedError("Use the derived class")

    def _save(self, fname, data, opts):
        raise NotImplementedError("Use the derived class")

    def save(self, fname, what, save_opts):
        """
        Given the registration result, save the transformed input.
        """
        if save_opts is None:
            save_opts = {}
        self._save(fname, what, save_opts)


@loader("mat")
class MatLoader(Loader):
    def __init__(self):
        super(MatLoader, self).__init__(10)

    def _load2reg(self, fname, opts):
        from scipy import io
        mat = io.loadmat(fname)
        valid = [key for key in mat if not key.startswith("_")]
        if "input" not in opts:
            if len(valid) != 1:
                raise RuntimeError(
                    "You have to supply an input key, there is an ambiguity "
                    "of what to load")
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


@loader("pil")
class PILLoader(Loader):
    def __init__(self):
        super(PILLoader, self).__init__(50)

    def _load2reg(self, fname, opts):
        from scipy import misc
        loaded = misc.imread(fname)
        self.loaded = loaded
        ret = loaded
        if ret.ndim == 3:
            ret = loaded.mean(axis=2)
        return ret

    def _save(self, fname, tformed, opts):
        from scipy import misc
        img = misc.toimage(tformed)
        img.save(fname)

    def guessCanLoad(self, fname):
        "We think that we can do everything"
        return True


@loader("hdr")
class HDRLoader(Loader):
    def __init__(self):
        super(HDRLoader, self).__init__(10)

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
        import numpy as np
        # Shouldn't happen, just to make sure
        tformed[tformed > 1.0] = 1.0
        tformed[tformed < 0.0] = 0.0
        tformed *= 255.0
        uint = tformed.astype(np.uint8)
        uint.tofile(fname)


loaders = LoaderSet()
