import setuptools

setuptools.setup (
    name            = "src",
    version         = "0.0.1",
    description     = "A highly customisable version of Spectral Clustering and Data Generation",
    author          = "Cassim Patel",
    author_email    = "sc20cp@leeds.ac.uk",
    packages        = setuptools.find_packages(),
    python_requires = '>=3.7',
    install_requires=[
        # TODO: add all packages that yours depends on
        # 'markdown',
    ],
)