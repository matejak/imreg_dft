Developer reference
===================

The following is a functionality that may be useful, but it is not considered as public API and it may somehow evolve over time.

.. _loaders_devel:

loader module
-------------

.. automodule:: imreg_dft.loader
   :members:
   :undoc-members:

utils module
------------

This module contains various support functions closely related to image registration.
They are used mainly by the ``ird`` tool.

.. automodule:: imreg_dft.utils
   :members:
   :undoc-members:

Become part of it!
------------------

Do you like the project?
Do you feel inspired?
Do you want to help out?

You are warmly welcome to do so!

How to contribute
+++++++++++++++++

The process is pretty standard if you are used to Github.

most likely

#. Become familiar with `git <http://git-scm.com/>`_ and learn how to use it properly, i.e. tell git_ who you are so it can label the commits you've made::

      git config --global user.email you@yourdomain.example.com
      git config --global user.name "Your Name Comes Here"

#. You can do two things now:
   
   a. Fork ``imreg_dft`` using Github web interface and clone it.

   #. If you want to make a minor modification and/or don't have a Github account, just clone ``imreg_dft``::

       git clone https://github.com/matejak/imreg_dft 
       cd imreg_dft

#. Make a 'feature branch'.
   This will be where you work on your bug fix or whatever.
   It's nice and safe and leaves you with access to an unmodified copy of the code in the main branch::

       git branch the-fix-im-thinking-of
       git checkout the-fix-im-thinking-of

   Then, do some edits, and commit them as you go.

#. Finally, you have to deliver your precious work in a smart way to the project.
   How to do this depends on whether you have created a pull request using Github or whether you went the simpler, but hardcore way.
   So, you have to do either

   a. use again the Github interface, select your feature branch there and do some clicking stuff to create a pull request,

   #. or make your commits into patches.
      You want all the commits since you branched from the ``master`` branch::

         git format-patch -M -C master

      You will now have several files named for the commits::

         0001-BF-added-tests-for-Funny-bug.patch
         0002-BF-added-fix-for-Funny-bug.patch

      Send these files to the current project maintainer.
