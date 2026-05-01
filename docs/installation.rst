Installation
============

Requirements
------------

- Python 3.12
- PyTorch >= 2.4.0
- CUDA >= 12.4
- A C++ compiler (g++ is used by default)

Setting Up the Environment
--------------------------

Create a fresh conda/mamba environment:

.. code-block:: bash

   mamba create -n torchff python=3.12
   mamba activate torchff

Install PyTorch with CUDA support:

.. code-block:: bash

   pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

Install testing dependencies (optional):

.. code-block:: bash

   pip install pytest

Installing torchff
------------------

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/THGLab/torchff-lib.git
   cd torchff-lib
   python setup.py install

For development mode (editable install):

.. code-block:: bash

   pip install -e .

Verifying the Installation
--------------------------

After installation, verify that the package is importable:

.. code-block:: python

   import torchff
   print("torchff installed successfully")

To verify CUDA extensions are available (requires GPU):

.. code-block:: python

   import torch
   from torchff.bond import HarmonicBond

   bond_fn = HarmonicBond(use_customized_ops=True)
   print("CUDA extensions loaded successfully")
