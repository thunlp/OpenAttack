python3 build-doc.py docs/source/apis
sphinx-build -M clean "docs/source" "docs/build" -W
sphinx-build -M html "docs/source" "docs/build"