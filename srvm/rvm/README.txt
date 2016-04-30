===================
sRVM for Matlab(tm)
===================
:copyright: 2006 Alexander Schmolck and Richard Everson
:date: 2006-05-10

.. contents:: 

Overview
========

The sRVM is a powerful smoothness prior extension of Tipping's `Relevance
Vector Machine`_ and can be used for arbitrary (sparse) regression or
shrinkage tasks that you might currently use e.g. RVM, SVM_ or wavelet
shrinkage for. Although The default settings (see below) give pretty good
results for many types of signals one might encounter in practice. Please
refer to our paper [sRVM06]_ for a detailed discussion.


.. _Relevance Vector Machine: http://www.relevancevector.com
.. _SVM: http://www.support-vector.net

Installation
============

1. unpack (e.g. in ``/home/joe/matlab/``):: 

     > tar xzf srvm.tgz

2. Edit your ``startup.m`` file to include ``/home/joe/matlab/srvm`` and
   ``/home/joe/matlab/awms`` in the matlab search-path, by adding something
   like::

     addpath('/home/joe/matlab/awms');
     addpath('/home/joe/matlab/srvm');

Dependencies 
------------ 
The code is tested under Matlab 7 -- earlier versions of `Matlab(tm)`_ might
require some modifications.

No particular 3rd party library is required, but in order to use symmlet or
haar kernels it is necessary to have some wavelet toolkit installed. The
simplest choice is to use the free WaveLab_ package, because the code in
``@wtMat`` and ``makeKernel`` expects interface-compatibility to the WaveLab_
functions ``MakeONFilter``, ``FWT_PO`` and ``IWT_PO``. However a trivial
wrapper to any other wavelet toolkit should also suffice. The ``MakeSignal``
function from WaveLab_ is also used by ``rvmopts`` to create test signals, but
an alternative ``makeSignal`` function is also supplied and will be used if
``MakeSignal`` is not in the path.

.. _Matlab(tm): http://www.mathworks.com
.. _WaveLab: http://www-stat.stanford.edu/~wavelab/

Usage
=====

Quick Intro
-----------

For a quick illustration create a signal (``yy`` in the example) and add noise
to it to obtain some fake regression targets ``tt``::

 > yy = sinc(linspace(-10,10,128)'./pi);
 > tt = yy + 1/7 .* std(yy) .* randn(128,1);

to then do sRVM regression with the default options, type::

 > opts = rvmopts(struct('tt',tt))
 > stats = rvm(opts)

.. image:: sinc-bare.png
   :alt: Sinc rvm regression results

With the default plot options the targets appear as blue dots and the
regression result ``yy_hat`` as magenta line. You might want to do individual
plots of ``yy``, ``tt`` and the regression result ``stats.yy_hat`` yourself
and check how well the noise has been estimated (``stats.sigma_sq /
var(yy-tt)``).

Note that for the default symmlet wavelet kernels ``N=length(tt)`` must be a
power of 2. To use for example an gaussian kernel with width ``r=1.0`` instead
(which does not have this restriction and also doesn't require Wavelab_ or a
similar wavelet package), do::

 > opts = rvmopts(struct('tt', tt, 'kernel', 'gauss', 'r', 3.0))

instead (you may notice that this type of kernel is particularly well suited
to simple Sinc-style datasets).

Here is a slightly more interesting example that the classical RVM would have
some trouble with (try setting ``opts.priorType = 'None'`` to obtain classical
RVM behavior for any of the examples; also try zooming into the left part of
the plot)::

    > opts = rvmopts(struct('sig', 'Doppler', 'N', 4096, 'kernel', 'symmlet','SNR', 6.0))
    > stats = rvm(opts)

.. image:: doppler-bare.png
   :alt: Doppler rvm regression results


Detailed Description
--------------------

The only files that the user would normally want to use directly are
``rvmopts.m``, ``rvm.m`` (and to a lesser extent ``plotRvm.m`` and
``makeKernel.m`` ). 

Calling ``rvm``
'''''''''''''''
``rvm`` expects a structure with options (``opts`` in these examples) and
returns a structure with results (``stats`` in these examples).

Option Structure (``opts``)
...........................

Typically one will want to create the options using the ``rvmopts`` command
(see `Calling rvmopts`_), but to just vary a certain option in an existing
``opts`` structure without modifying the original you might find it useful to
use the ``restruct`` utility in ``awms``, e.g. to turn off plotting and
verbose messages try ``rvm(restruct(opts, 'plot', false, 'chatty', false))``.


Result Structure (``stats``)
............................

The most important fields in ``stats`` are as follows:


``yy_hat``
  The mean posterior prediction for the true signal ``yy``.

``mmu_M``
  The posterior mean regression weights.

``sel``
  A boolean vector signifying the selected components.

``mmu``
  ``mmu_M(sel)``.

``sigma_sq``
  The estimated value for the noise variance.

``aalpha``
  The final values for the weight precision vector ``aalpha``.

``Post``
  The log posterior probability of the result (L_mathcal_hat in the article).

``steps``
  The number of steps that it took to reach the result.

``realSteps``
  The number of steps during which an actual addition/deletion/reestimation
  happened.

``converged``
  False if the model just stopped because ``maxSteps`` was exceeded, true
  otherwise.

Calling ``rvmopts``
'''''''''''''''''''
``rvmopts`` takes a structure with parameters for the sRVM and supplies some
sensible default values for everything that is not specified -- in the most
extreme case, if nothing is specified at all (``opts = rvmopts(struct()))``,
some fake data is generated for demonstration, too. Typically one would want
to specify at least the regression targets ``tt`` and the kernel type, which
can be done as follows:

- to specify the targets for regression one can either 

  a) pass a column vector ``tt`` (and optionally the true signal ``yy`` if
     known, for calculating the MSE of the result ``yy_hat``)

  b) pass ``N`` (default 512), the number of data points and ``sig`` a signal
     name that is either ``'Sinc'`` or recognized by ``MakeSignal`` (or
     ``makeSignal``), (default ``'Doppler'``). To control the amount of
     noise in the targets one can either specify:

        i) ``SNR`` the signal to noise ratio of the desired targets or

        ii) ``sigma`` the std of the noise 

     The seeding for the the random noise added to the signal is controlled by
     the first entry in the ``seeds`` vector.

- to specify the kernel to use one can either pass

  a) ``PPhi`` -- a kernel matrix or

  b) ``kernel``, a kernel name that is recognized by ``MakeKernel`` (e.g.
     ``'symmlet'`` (default), ``'lspline'``, ``'gauss'`` or ``'tpspline'``).
     For some kernels one should also supply a width parameter ``r`` (default
     3.0).

- the amount of sparsity is controlled by setting ``priorType`` which can be
  one of (in increasing order of sparsity):

  - ``None`` (for classical rvm behavior)

  - ``AIC``

  - ``BIC`` (default)

  - ``RIC``

- to control the noise restimation process one can either

  a) set ``reest__sigma`` to a certain perior of realSteps at which noise
     reestimation is to occur (default 10) and optionally set ``sigma_0``, the
     initial guess (default ``(std(tt)/100).^2``). An imaginary value will
     result in a multiple of the true ``sigma``, if it is also specified.

  b) set ``reest__sigma`` to 0 to turn off noise reestimation and give the
     desired noise level via the ``sigma`` parameter. A multiple of the true
     ``sigma`` might be specified by setting ``sigma_0`` to an imaginary
     value, e.g. ``2j``.
     
- the seeding of the random number generator for fake target creation and the
  order in which component selection proceeds is respectively controlled by
  the first and second element in the 2 element vector ``seeds`` (default
  ``rand(2,1)``).
  
- further options include

    ``plot`` 
      Plot results if nonzero; if > 1 also plot progressive results during run.

    ``chatty``
      An integer >=0 that controls the verbosity of printouts during execution.

    ``save``
      A boolean that controls whether the experiment is saved in a ``.mat``-file,
      the name of which is partly controlled by the ``DUB`` parameter

    ``maxSteps``
      The maximum number of steps that is allowed before (default 1E6).
 
Calling ``plotRvm``
'''''''''''''''''''
If the ``opts`` structure passed to ``rvm`` has ``'opts.plot == true``
(default) then results will be automatically plotted, but you can also
manually create a plot by using ``plotRvm(opts, stats)``.


Calling ``makeKernel``
''''''''''''''''''''''
The desired kernel can in most cases by created by just passing the
appropriate parameters to ``rvmopts``, e.g. 
``rvmopts('kernel', 'gauss', 'r', 3.0, ...)``.

However calling ``MakeKernel`` can be useful to built more customized (e.g.
overcomplete) kernels. The usage is::

> K = makeKernel(type, N, kernelOpts)

Where ``[N,N] = size(K)`` and type can currently be one of:

- ``symmlet``, ``haar``, ``gauss``, ``lspline`` (linear spline) and
  ``tpspline`` (thin plate spline)

- the optional ``kernelOpts`` structure can be used to specify 
  
  - the kernel width (``r``) for kernel for which it makes sense (spline
    kernels)

  - via ``pseudoMatrix`` option (default ``true``),` whether K will be a
    proper matrix or just an internally more efficient matrix like-object, for
    kernels were matrix-multiplication is mathematically equivalent to a much
    more efficient transform (currently wavelet kernels, i.e. ``symmlet`` and
    ``haar``)

Notes
=====
Bear in mind that this code is mostly meant as an illustration and
implementation for the method described our research paper [sRVM06]_, not as
production quality software, so it will certainly suffer from various
shortcomings.

Queries and bug reports should be directed to <a.schmolck@gmx.net> with sRVM
in the subject line (no html mail, please).


References
==========

.. [sRVM06] Alexander Schmolck and Richard Everson
    (submitted 2006), *"sRVM: a smoothness prior extension of the Relevance Vector
    Machine"* (`download draft`_).

.. _`download draft`: rvm-article.pdf

Download
========

The code is available as gzipped tar archive: srvm.tgz_

.. _srvm.tgz: srvm.tgz