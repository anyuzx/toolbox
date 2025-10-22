# Try relative imports first (when installed as package), then absolute (when used as script)
try:
    from . import contactmap
    from . import LammpsLog
    from . import LammpsData
    from . import misc
    
    # Optional Cython extension - only import if built
    try:
        from . import _matrixnorm
    except ImportError:
        _matrixnorm = None
        import warnings
        warnings.warn(
            "Could not import _matrixnorm Cython extension. "
            "Some contactmap functionality (normalize, get_OE, get_zscore, get_PearsonCoeff, get_subchain_contact) "
            "will not be available. To enable, run: pip install -e . or python setup.py build_ext --inplace",
            ImportWarning
        )
except ImportError:
    # Fallback to absolute imports
    import contactmap
    import LammpsLog
    import LammpsData
    import misc
    
    # Optional Cython extension - only import if built
    try:
        import _matrixnorm
    except ImportError:
        _matrixnorm = None
        import warnings
        warnings.warn(
            "Could not import _matrixnorm Cython extension. "
            "Some contactmap functionality (normalize, get_OE, get_zscore, get_PearsonCoeff, get_subchain_contact) "
            "will not be available. To enable, run: pip install -e . or python setup.py build_ext --inplace",
            ImportWarning
        )
