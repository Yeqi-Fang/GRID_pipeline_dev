from .FigureManager import FigureManager
import warnings

warnings.warn('`grafica` is not maintained anymore, all my plotly utils were moved to https://github.com/SengerM/plotly_utils', DeprecationWarning, stacklevel=2)

manager = FigureManager()

def new(*args, **kwargs):
	"""A shorthand wrapper around grafica.manager.new"""
	return manager.new(*args, **kwargs)

def save_unsaved(*args, **kwargs):
	"""A shorthand wrapper around grafica.manager.save_unsaved"""
	return manager.save_unsaved(*args, **kwargs)

def show(*args, **kwargs):
	return manager.show(*args, **kwargs)
