
Advanced command-line
=====================

Apart from simple problems, the command-line is able to solve more tricky cases such as cases when

* one (or more) of rotation, scaling, or translation is known and
* the image field of view is a (small) subset of the template's.

Using constraints
-----------------

``imreg_dft`` allows you to specify a constraint on any transformation.
It works the same way for all values.
You can specify the expected value and the uncertainity by specifying a mean (:math:`\mu`) and standard deviation (:math:`\sigma`) of the variable.

Values is proportionally reduced in the phase correlation phase of the algorithm.

 ..  TODO: Add an image here

You can use (separately or all at once) options ``--angle``, ``--scale``, ``--tx`` and ``ty``.
Notice that since the translation comes after scale change and rotation, it doesn't make much sense to use either ``--tx`` or ``--ty`` without having strong control over ``--angle`` *and* ``--scale``.

You can either:

* Ignore the options --- the default are null constraints.

* Specify a null constraint explicitly by writing the delimiting comma not followed by anything (i.e. ``--angle -30,``).

* Pass the mean argument but omit the stdev part, in which case it is assumed zero (i.e. ``--angle -30`` is the same as ``--angle -30,0``).
  Zero standard deviation is directive --- the angle value that is closest to -30 will be picked.

* Pass both parts of the argument --- mean and stdev, i.e. ``--angle -30,1``, in which case angles below -33 and above -27 are virtually ruled out.

  .. note::
     
     The Python argument parsing may not like argument value ``-30,2`` because ``-`` is the single-letter argument prefix and ``-30,2`` is not a number.
     On unix-like systems, you may circumvent this by writing ``--angle ' -30,2``.
     Now, the argument value begins by space and this doesn't make any trouble down the road.

Large templates
---------------

``imreg_dft`` works well on images that show the same object that is contained within the field of view.
However, the method fails when the fields of view don't match and are in subset-superset relationship.

Normally, the image will be "extended", but that may not work.
Therefore, if the ``image`` is the *smaller* of the two, i.e. ``template`` encompasses it, you can use the ``--tile`` argument.
Then, the template will be internally subdivided into tiles (of similar size to the image size) and individual tiles will be matched against the image and the tile that matches the best will determine the transformation.

The ``--show`` option will show matching over the best fitting tile and you can use the ``--output`` option to save the registered image (that is enlarged to the shape of the template).

.. TODO: Get tile size from constraints, alow specification of density
