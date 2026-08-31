# Contributing

Install the documented Python dependencies before running the notebook:

~~~bash
python -m pip install tensorflow opencv-python numpy
~~~

Keep dataset provenance, class definitions, preprocessing, and model hashes with evaluation notes. Use [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) when replacing the Keras artifact.

Do not commit webcam frames, restricted training data, credentials, or claims that exceed the measured evaluation. Explain lighting and background conditions for any new result.
