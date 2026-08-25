Preprocessing
=============

Raw Maxwell ``.raw.h5`` in, cleaned ``.npz`` out. :mod:`mxtreme.extract` reads the file,
:mod:`mxtreme.clean` supplies the composable steps, :mod:`mxtreme.pipeline` runs them in order, and
:mod:`mxtreme.io` writes the result to the managed store.

Extraction
----------

.. automodule:: mxtreme.extract
   :members:

Cleaning steps
--------------

.. automodule:: mxtreme.clean
   :members:

Pipeline
--------

.. automodule:: mxtreme.pipeline
   :members:

Reading and writing
-------------------

.. automodule:: mxtreme.io
   :members:
