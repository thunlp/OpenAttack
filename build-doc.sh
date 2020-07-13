sphinx-apidoc -o docs/source/ -f --templatedir docs/source/_templates/ OpenAttack
rm docs/source/modules.rst
sphinx-build -M clean "docs/source" "docs/build" -W
sphinx-build -M html "docs/source" "docs/build"