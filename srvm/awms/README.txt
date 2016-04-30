This directory contains various fairly general utility functions for matlab.
Apart from supplying functionality that seems strangely missing (``assert``,
``shuf`` to shuffle a vector, ``geneq`` to provide a proper equality
predicate), one recurring theme is allowing a more functional programming
style (i.e. minimizing the need to assign spurious dummy variables) -- see
e.g. ``argmax``, ``restruct`` and ``aref`` as well as ``map``. Another is
supplying some APL-style constructs (``outer``, ``pairwise``). The last on is
simply saving typing (``ploty``, ``imagesc``).

This code is simply a collection of quick hacks for my convenience and was not
written with efficiency or generality in mind.

-- Alexander Schmolck, 2006-05-12 