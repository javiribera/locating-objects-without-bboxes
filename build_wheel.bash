#!/bin/bash

python setup.py bdist_wheel
mv dist/*.whl .
rm -R build dist object_locator.egg-info

echo 'DONE'
