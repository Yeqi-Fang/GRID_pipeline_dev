import numpy as np
from .traces import Trace, Histogram

_VALID_AXIS_SCALES = {'lin','log'}

class Figure:
	"""
	This class defines the interface and handles all the information until
	the plots are "drawn" with a specific package.
	"""
	
	def __init__(self):
		self._show_title = True
		self.traces = []
	
	def show(self):
		"""Must override this method when inheriting."""
		raise NotImplementedError(f'Not implemented yet for the plotting package you are using! (Specifically for the class {self.__class__.__name__}.)')
	
	# Figure properties ------------------------------------------------
	# 	Figure properties are those things that belong to the figure
	# 	itself and not to particular traces. Examples: title, x scale, etc.
	
	@property
	def title(self):
		return self._title if hasattr(self, '_title') else None
	@title.setter
	def title(self, title):
		self._title = str(title)
		self.draw_layout() # Update the "drawn figure".
	
	@property
	def show_title(self):
		return self._show_title if hasattr(self, '_show_title') else None
	@show_title.setter
	def show_title(self, show: bool):
		if show not in [True, False]:
			raise TypeError(f'<show_title> expects either True or False, received {show}.')
		self._show_title = show
		self.draw_layout() # Update the "drawn figure".
	
	@property
	def subtitle(self):
		return self._subtitle if hasattr(self, '_subtitle') else None
	@subtitle.setter
	def subtitle(self, subtitle):
		self._subtitle = str(subtitle)
		self.draw_layout() # Update the "drawn figure".
	
	@property
	def xlabel(self):
		return self._xlabel if hasattr(self, '_xlabel') else None
	@xlabel.setter
	def xlabel(self, xlabel):
		self._xlabel = str(xlabel)
		self.draw_layout() # Update the "drawn figure".
	
	@property
	def ylabel(self):
		return self._ylabel if hasattr(self, '_ylabel') else None
	@ylabel.setter
	def ylabel(self, ylabel):
		self._ylabel = str(ylabel)
		self.draw_layout() # Update the "drawn figure".
	
	@property
	def xscale(self):
		return self._xscale if hasattr(self, '_xscale') else None
	@xscale.setter
	def xscale(self, xscale):
		if xscale not in _VALID_AXIS_SCALES:
			raise ValueError(f'<xscale> must be one of {_VALID_AXIS_SCALES}, received {xscale}.')
		self._xscale = xscale
		self.draw_layout() # Update the "drawn figure".
	
	@property
	def yscale(self):
		return self._yscale if hasattr(self, '_yscale') else None
	@yscale.setter
	def yscale(self, yscale):
		if yscale not in _VALID_AXIS_SCALES:
			raise ValueError(f'<yscale> must be one of {_VALID_AXIS_SCALES}, received {yscale}.')
		self._yscale = yscale
		self.draw_layout() # Update the "drawn figure".
	
	@property
	def aspect(self):
		return self._aspect if hasattr(self, '_aspect') else None
	@aspect.setter
	def aspect(self, aspect):
		VALID_ASPECTS = {'equal',None}
		if aspect not in VALID_ASPECTS:
			raise ValueError(f'<aspect> must be one of {VALID_ASPECTS}, received {aspect}.')
		self._aspect = aspect
		self.draw_layout() # Update the "drawn figure".
	
	def set(self, **kwargs):
		for key in kwargs.keys():
			if not hasattr(self, key):
				raise ValueError(f'Cannot set <{key}>, invalid property.')
			setattr(self, f'{key}', kwargs[key])
	
	# Drawing methods --------------------------------------------------
	# The interface of these methods is defined here, but each specific
	# plotter must override them implementing the details on how to draw
	# each thing. Here (in the Figure class) we are at a too high level
	# to know how to implement each of these things, but they will for
	# sure be needed.
	# Do not change the signature of these "drawing methods"!
	
	def show(self, **kwargs):
		"""Must override this method when inheriting.
		This method must display the figure in its current state."""
		raise NotImplementedError(f'Not implemented yet for the plotting package you are using! (Specifically for the class {self.__class__.__name__}.)')
	
	def save(self, file_name=None, **kwargs):
		"""Must override this method when inheriting.
		This method must save the figure in its current state to a persistent
		file in the hard drive."""
		raise NotImplementedError(f'Not implemented yet for the plotting package you are using! (Specifically for the class {self.__class__.__name__}.)')
	
	def draw_layout(self):
		"""Must override this method when inheriting.
		This method must draw all the "figure properties", e.g. the title,
		the axes labels, etc."""
		raise NotImplementedError(f'Not implemented yet for the plotting package you are using! (Specifically for the class {self.__class__.__name__}.)')
	
	def draw_trace(self, trace: Trace):
		"""Must override this method when inheriting.
		This method must draw the trace object received."""
		raise NotImplementedError(f'Not implemented yet for the plotting package you are using! (Specifically for the class {self.__class__.__name__}.)')
	
	# ------------------------------------------------------------------
	
	def add_trace(self, trace: Trace):
		"""Adds a trace to the figure. The user should not interact very
		much with this method, the idea is to create one method for each
		type of trace to ease the life of the user, e.g. self.scatter
		receives the required information to create a Scatter trace and 
		automatically calls self.add_trace(statter) later on. So the user
		interacts with self.scatter. Of course he can call 
			self.add_trace(Scatter(bla bla bla))
		instead of
			self.scatter(bla bla bla)
		but it is more cumbersome which goes against the philosophy of
		this package."""
		if not isinstance(trace, Trace):
			raise TypeError(f'<trace> must be an instance of Trace, received an object of type {type(trace)}')
		self.traces.append(trace)
		self.draw_trace(trace)
	
	# Methods to ease the life of the user -----------------------------
	
	
	
	def histogram(self, samples, **kwargs):
		"""Given an array of samples produces a histogram.
		For optional kwargs see documentation of traces.Histogram."""
		if kwargs.get('color') is None:
			kwargs['color'] = self.pick_default_color()
		# print(kwargs)
		# subplots = kwargs.pop('subplots', False)
		# if subplots:
		# 	col = kwargs.pop('col')
		# 	row = kwargs.pop('row')
		# print(kwargs)
		self.add_trace(Histogram(samples, **kwargs))
	
	
	
	# ------------------------------------------------------------------
	
	DEFAULT_COLORS = [
		(255, 59, 59),
		(52, 71, 217),
		(4, 168, 2),
		(224, 146, 0),
		(224, 0, 183),
		(0, 230, 214),
		(140, 0, 0),
		(9, 0, 140),
		(107, 0, 96),
	]
	def pick_default_color(self):
		color = self.DEFAULT_COLORS[0]
		self.DEFAULT_COLORS = self.DEFAULT_COLORS[1:] + [self.DEFAULT_COLORS[0]]
		return color
