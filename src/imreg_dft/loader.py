# -*- coding: utf-8 -*-
# loader.py

# Copyright (c) 2014-?, Matěj Týč
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
The module contains a layer of functionality that allows
abstract saving and loading of files.

The concept is that there is one loader instance per file loaded.
When we want to save a file, we use a loading loader to provide data to
save and then we instantiate a saving loader (if needed) and save the data.

A loader class inherits from :class:`Loader`.
A singleton class :class:`LoaderSet` is the main public interface
of this module. available as a global variable ``LOADERS``.
It keeps track of all registered loaders and takes care
after them (presents them with options, requests etc.)
Loaders are registered as classes using the decorator :func:`loader`.

Individual loaders absolutely have to implement methods :meth:`Loader._save`
and :meth:`Loader._load2reg`.

This module facilitates integration of its functionality by defining
:func:`update_parser` and :func:`settle_loaders`. While the first one can
add capabilities to a parser (or parser group), the second one updates
``LOADERS`` accordingly while given parsed arguments.

Rough edges (but not rough enough to be worth the trouble):

* You can't force different loaders for image, template and output. If you
  need this, you have to rely on autodetection based on file extension.
* Similarly, there is a problem with loader options --- they are shared among
  all loaders. This is both a bug and a feature though.
* To show the loaders help, you have to satisfy the parser by specifying
  a template and image file strings (they don't have to be real filenames tho).
"""

import sys


def _str2nptype(stri):
    import numpy as np
    msg = ("The string '%s' is supposed to correspond to a "
           "numpy type" % stri)
    try:
        typ = eval("np." + stri)
    except Exception as exc:
        msg += " but it is not the case at all - %s." % exc.message
        raise ValueError(msg)
    typestr = type(typ).__name__
    # We allow mock object for people who know what they are doing.
    if typestr not in ("type", "Mock"):
        msg += " but it is a different animal than 'type': '%s'" % typestr
        raise ValueError(msg)
    return typ


def _str2flat(stri):
    assert stri in "R,G,B,V".split(","), \
        "Flat value has to be one of R, G, B, V, is '%s' instead" % stri
    return stri


def flatten(image, char):
    if image.ndim < 3:
        return image
    char2idx = dict(R=0, G=1, B=2)
    ret = None
    if char == "V":
        ret = image.mean(axis=2)
    elif char in char2idx:
        ret = image[:, :, char2idx[char]]
    else:
        # Shouldn't happen
        assert False, "Unhandled - invalid flat spec '%s'" % char
    return ret


class LoaderSet(object):
    _LOADERS = []
    # singleton-like functionality
    _we = None

    def __init__(self):
        if LoaderSet._we is not None:
            return LoaderSet._we
        loaders = [loader() for loader in LoaderSet._LOADERS]
        self.loader_dict = {}
        for loader in loaders:
            self.loader_dict[loader.name] = loader
        self.loaders = sorted(loaders, key=lambda x: x.priority)
        LoaderSet._we = self

    def _choose_loader(self, fname):
        """
        Use autodetection to select a loader to use.

        Returns:
            Loader instance or None if no loader can be used.
        """
        for loader in self.loaders:
            if loader.guessCanLoad(fname):
                return loader
        # Ouch, no loader available!
        return None

    def get_loader(self, fname, lname=None):
        """
        Try to select a loader. Either we know what we want, or an
        autodetection will take place.
        Exceptions are raised when things go wrong.
        """
        if lname is None:
            ret = self._choose_loader(fname)
            if ret is None:
                msg = ("No loader wanted to load '%s' during autodetection"
                       % fname)
                raise IOError(msg)
        else:
            ret = self._get_loader(lname)
        # Make sure that we don't return the same instance multiple times
        ret = ret.spawn()
        return ret

    def _get_loader(self, lname):
        if lname not in self.loader_dict:
            msg = "No loader named '%s'." % lname
            msg += " Choose one of %s." % self.loader_dict.keys()
            raise KeyError(msg)
        return self.loader_dict(lname)

    def get_loader_names(self):
        """
        What are the names of loaders that we know.
        """
        ret = self.loader_dict.keys()
        return tuple(ret)

    @classmethod
    def add_loader(cls, loader_cls):
        """
        Use this method (at early run-time) to register a loader
        """
        cls._LOADERS.append(loader_cls)

    def print_loader_help(self, lname=None):
        """
        Print info about loaders.
        Either print short summary about all loaders, or focus just on one.
        """
        if lname is None:
            msg = "Available loaders: %s\n" % (self.get_loader_names(),)
            # Lowest priority first - they are usually the most general ones
            for loader in self.loaders[::-1]:
                msg += "\n\t%s: %s\n\tAccepts options: %s\n" % (
                    loader.name, loader.desc, tuple(loader.opts.keys()))
        else:
            loader = self.loader_dict[lname]
            msg = "Loader '%s':\n" % loader.name
            msg += "\t%s\n" % loader.desc
            msg += "Accepts options:\n"
            for opt in loader.opts:
                msg += "\t'%s' (default '%s'): %s\n" % (
                    opt, loader.defaults[opt], loader.opts[opt], )
        print(msg)

    def distribute_opts(self, opts):
        """
        Propagate loader options to all loaders.
        """
        if opts is None:
            # don't return, do something so possible problems surface.
            opts = {}
        for loader in self.loaders:
            loader.setOpts(opts)


def loader(lname, priority):
    """
    A decorator interconnecting an abstract loader with the rest of imreg_dft
    It sets the "nickname" of the loader and its priority during autodetection
    """
    def wrapped(cls):
        cls.name = lname
        cls.priority = priority
        LoaderSet.add_loader(cls)
        return cls
    return wrapped



class Loader(object):
    """

    .. automethod:: _save
    .. automethod:: _load2reg
    """
    name = None
    priority = 10
    desc = ""
    opts = {}
    defaults = {}
    str2val = {}

    def __init__(self):
        self.loaded = None
        self._opts = {}
        # First run, the second will hopefully follow later
        self.setOpts({})
        # We may record some useful stuff for saving during loading
        self.saveopts = {}

    def spawn(self):
        """
        Makes a new instance of the object's class
        BUT it conserves vital data.
        """
        cls = self.__class__
        ret = cls()
        # options passed on command-line
        ret._opts = self._opts
        return ret

    def setOpts(self, options):
        for opt in self.opts:
            stri = options.get(opt, self.defaults[opt])
            val = self.str2val.get(opt, lambda x: x)(stri)
            self._opts[opt] = val

    def guessCanLoad(self, fname):
        """
        Guess whether we can load a filename just according to the name
        (extension)
        """
        return False

    def load2reg(self, fname):
        """
        Given a filename, it loads it and returns in a form suitable for
        registration (i.e. float, flattened, ...).
        """
        try:
            ret = self._load2reg(fname)
        except IOError as err:
            print("Couldn't load '%s': %s" % (fname, err.strerror))
            sys.exit(1)

        return ret

    def get2save(self):
        assert self.loaded is not None, \
            "Saving without loading beforehand, which is not supported. "
        return self.loaded

    def _load2reg(self, fname):
        """
        To be implemented by derived class.
        Load data from fname in a way that they can be used in the
        registration process (so it is a 2D array).
        Possibly take into account options passed upon the class creation.
        """
        raise NotImplementedError("Use the derived class")

    def _save(self, fname):
        """
        To be implemented by derived class.
        Save data to fname, possibly taking into account previous loads
        and/or options passed upon the class creation.
        """
        raise NotImplementedError("Use the derived class")

    def save(self, fname, what, loader):
        """
        Given the registration result, save the transformed input.
        """
        sopts = loader.saveopts
        self.saveopts.update(sopts)
        self._save(fname, what)


@loader("mat", 10)
class _MatLoader(Loader):
    desc = "Loader of .mat (MATLAB v5) binary files"
    opts = {"in": "The structure to load (empty => autodetect)",
            "out": "The structure to save the result to (empty => the same "
                   "as the 'in'",
            "type": "Name of the numpy data type for the output (such as "
                    "int, uint8 etc.)",
            "flat": "How to flatten (the possibly RGB image) for the "
                    "registration. Values can be R, G, B or V (V for value - "
                    "a number proportional to average of R, G and B)",
            }
    defaults = {"in": "", "out": "", "type": "float", "flat": "V"}
    str2val = {"type": _str2nptype, "flat": _str2flat}

    def __init__(self):
        super(_MatLoader, self).__init__()
        # By default, we have not loaded anything
        self.saveopts["loaded_all"] = {}

    def _load2reg(self, fname):
        from scipy import io
        mat = io.loadmat(fname)
        if self._opts["in"] == "":
            valid = [key for key in mat if not key.startswith("_")]
            if len(valid) != 1:
                raise RuntimeError(
                    "You have to supply an input key, there is an ambiguity "
                    "of what to load, candidates are: %s" % (tuple(valid),))
            else:
                key = valid[0]
        else:
            key = self._opts["in"]
            keys = mat.keys()
            if not key in keys:
                raise LookupError(
                    "You requested load of '%s', but you can only choose from"
                    " %s" % (tuple(keys),))
        ret = mat[key]
        self.saveopts["loaded_all"] = mat
        self.saveopts["key"] = key
        self.loaded = ret
        # flattening is a no-op on 2D images
        ret = flatten(ret, self._opts["flat"])
        return ret

    def _save(self, fname, tformed):
        from scipy import io
        if self._opts["out"] == "":
            assert "key" in self.saveopts, \
                "Don't know how to save the output - what .mat struct?"
            key = self.saveopts["key"]
        else:
            key = self._opts["out"]
        out = self.saveopts["loaded_all"]
        out[key] = tformed.astype(self._opts["type"])
        io.savemat(fname, out)

    def guessCanLoad(self, fname):
        return fname.endswith(".mat")


@loader("pil", 50)
class _PILLoader(Loader):
    desc = "Loader of image formats that Pillow (or PIL) can support"
    opts = {"flat": _MatLoader.opts["flat"]}
    defaults = {"flat": _MatLoader.defaults["flat"]}
    str2val = {"flat": _MatLoader.str2val["flat"]}

    def __init__(self):
        super(_PILLoader, self).__init__()

    def _load2reg(self, fname):
        from scipy import misc
        loaded = misc.imread(fname)
        self.loaded = loaded
        ret = loaded
        # flattening is a no-op on 2D images
        ret = flatten(ret, self._opts["flat"])
        return ret

    def _save(self, fname, tformed):
        from scipy import misc
        img = misc.toimage(tformed)
        img.save(fname)

    def guessCanLoad(self, fname):
        "We think that we can do everything"
        return True


@loader("hdr", 10)
class _HDRLoader(Loader):
    desc = ("Loader of .hdr and .img binary files. Supply the '.hdr' as input,"
            "a '.img' with the same basename is expected.")
    opts = {"norm": "Whether to divide the value by 255.0 (0 for not to)"}
    defaults = {"norm": "1"}

    def __init__(self):
        super(_HDRLoader, self).__init__()

    def guessCanLoad(self, fname):
        return fname.endswith(".hdr")

    def _load2reg(self, fname):
        """Return image data from img&hdr uint8 files."""
        import numpy as np
        basename = fname.rstrip(".hdr")
        with open(basename + '.hdr', 'r') as fh:
            hdr = fh.readlines()
        img = np.fromfile(basename + '.img', np.uint8, -1)
        img.shape = int(hdr[4].split()[-1]), int(hdr[3].split()[-1])
        if int(self._opts["norm"]):
            img = img.astype(np.float64)
            img /= 255.0
        return img

    def _save(self, fname, tformed):
        import numpy as np
        # Shouldn't happen, just to make sure
        tformed[tformed > 1.0] = 1.0
        tformed[tformed < 0.0] = 0.0
        tformed *= 255.0
        uint = tformed.astype(np.uint8)
        uint.tofile(fname)


def _parse_opts(stri):
    from argparse import ArgumentTypeError
    components = stri.split(",")
    ret = {}
    for comp in components:
        sides = comp.split("=")
        if len(sides) != 2:
            raise ArgumentTypeError(
                "The options spec has to look like 'option=value', got %s."
                % comp)
        lhs, rhs = sides
        valid_optname = False
        for loader in LOADERS.loaders:
            if lhs in loader.opts:
                valid_optname = True
                break
        if not valid_optname:
            raise ArgumentTypeError(
                "The option '%s' is not understood by any loader" % lhs)
        ret[lhs] = rhs
    return ret


def update_parser(parser):
    parser.add_argument(
        "--loader", choices=LOADERS.get_loader_names(), default=None,
        help="Force usage of a concrete loader (default is autodetection). "
        "If you plan on using two types of loaders to load input, or save the"
        " output, autodetection is the only way to achieve this.")
    parser.add_argument(
        "--loader-opts", default=None, type=_parse_opts,
        help="Options for a loader "
        "(use --loader to make sure that one is used or read the docs.)")
    parser.add_argument(
        "--help-loader", default=False, action="store_true",
        help="Get help on all loaders or on the current loader "
        "and its options.")


def settle_loaders(args, fnames=None):
    if args.help_loader:
        LOADERS.print_loader_help(args.loader)
        sys.exit(0)
    LOADERS.distribute_opts(args.loader_opts)
    ret = {}
    loaders = []
    if fnames is not None:
        for fname in fnames:
            loader = LOADERS.get_loader(fname, args.loader)
            loaders.append(loader)
    ret["loaders"] = loaders
    return ret


LOADERS = LoaderSet()
