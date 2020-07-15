====================
Installation
====================

Using Source Code
-------------------

You can install OpenAttack by cloning github repository and run ``python setup.py install``.

.. code-block:: sh

    git clone https://github.com/thunlp/OpenAttack.git
    cd OpenAttack
    python setup.py install

After running the above script successfully, you can use demp.py to check that OpenAttack works well.

.. code-block:: sh

    python demo.py


Mandatory Requirements
--------------------------

There are only three mandatory packages required by OpenAttack. They will be automatically installed
during running `python setup.py install`.

* **tqdm** >= 4.45.0
* **nltk** >= 3.5
* **numpy** >= 1.18.1

Optional Requirements
---------------------------

The following packages are used in some sub-modules of OpenAttack. They
are not required unless you use some sub-modules that need these packages.
You can install them manually when needed.

* **pytorch** >= 1.5.0
* **tensorflow** >= 2.0.0
* **tensorflow_hub**
* **transformers**
* **OpenHowNet**
* **editdistance**
* **language_tool_python**
* **torchtext** == 0.1.1 *(The latest version of torchtext is not supported now. This will be improved in the futrue.)*
