Very concise report on some choices:

General choices:

pdoc use: We have chose pdoc for the documentation generation because it
    supports google-style docstrings (our favourites), and because it is
    simple to use, yet sufficient for our needs.

editing datasets gui: In the Datasets page of the streamlit app, we have
    provided the capability of editing datasets and saving them as
    new datasets. We consider this nice and useful, but are unsure if it was
    a requirement or an addition on our part. The instruction: "a page where
    you can manage the datasets" is quite ambiguous.


Specific choices:

Artifact's read method: One might notice the artifact's read method is too specific
    for such a general class. This is true. That method should have been the read method
    of the Dataset class only (a child of Artifact). However, in the provided code,
    in the system.py file, the artifact registry works with pure artifacts. Even if I pass it a dataset,
    when it will recreate it, it will be a simple Artifact again. Touching and changing that code too much
    is beyond the duties of this assignment. Thus, I gave the Dataset's read method to Artifact.

Pipeline's artifact save/load: We found the preimplemented functions provided for saving the pipeline
    as an artifact quite hard to work with (also rather underdocumented), so we have created our own.
    As a result, we ignore the 'artifacts' method, but use the 'to_artifact' method we made.


Notes on the docs: The html documentation is provided in the docs folder. It was not specified
    how it should be displayed, so we have simply put it in the docs folder as that seems to be the
    standard.
