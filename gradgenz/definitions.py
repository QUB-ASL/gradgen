import pkg_resources


def templates_dir():
    """Directory where the templates are found (for internal use, mainly)"""
    return pkg_resources.resource_filename('gradgenz', 'templates/')


def templates_subdir(subdir=None):
    """
    Directory where the templates are found and subfolder relative
    to that path(for internal use, mainly)
    """
    if subdir is None:
        return templates_dir()
    return pkg_resources.resource_filename('gradgenz', 'templates/%s/' % subdir)
