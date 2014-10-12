Command-line tool overview
==========================

The package contains one Python :abbr:`CLI (command-line interface)` script.
Although you are more likely to use the ``imreg_dft`` functionality from your own Python programs, you can still have some use to the ``ird`` front-end.
There are three main reasons why you would want to use it:

#. Quickly find out whether it works for you, having the results shown in a pop-up window:

   .. code-block:: shell-session

     [user@linuxbox ~]$ ird la.png lo.png --show

#. Have the results print in a defined way:

   .. code-block:: shell-session

     [user@linuxbox ~]$ ird la.png lo.png --print-result 'rot:%(rotation)g\nscale:%(scale)\ntranslation:%(tx)d,%(ty)d'
     rot:45
     scale:0.987
     translation:2,4
