====================
Installation
====================

You can either use `pip` or clone this repo to install OpenAttack.

1. Using pip (recommended)
-----------------------------

.. code-block:: sh

    pip install OpenAttack

2. Cloning this repo
-----------------------------

.. code-block:: sh

    git clone https://github.com/thunlp/OpenAttack.git
    cd OpenAttack
    python setup.py install

After installation, you can try running `demo.py` to check if OpenAttack works well:

.. code-block:: sh

    python demo.py

.. image:: /images/demo.gif

Mandatory Requirements
-----------------------------

There are only three mandatory packages required by OpenAttack. They will be automatically installed
during running `python setup.py install`.

* **tqdm** >= 4.45.0
* **nltk** >= 3.5
* **numpy** >= 1.18.1
* **transformers** >= 4.0.0
* **torch** >= 1.5.1

Optional Requirements
-----------------------------

The following packages are used in some sub-modules of OpenAttack. They
are not required unless you use some sub-modules that need these packages.
You can install them manually when needed.

**Python Packages:**

* **tensorflow** >= 2.0.0
* **tensorflow_hub**
* **OpenHowNet**
* **editdistance**
* **language_tool_python**
* **sklearn**
* **sentence_transformers**

**Others:**

* **Java**