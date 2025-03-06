from .validation import validate_alpha, validate_color, validate_label, validate_linestyle, validate_linewidth, validate_marker
import numpy as np
from scipy.stats import gaussian_kde

VALID_ZSCALES = {'lin','log'}

class Trace:
	"""Most basic trace definition. Other traces should inherit from this
	parent class. This class (and sub classes) are intended to be just
	containers of information, with validation. Nothing else."""
	def __init__(self, label=None):
		self._label = validate_label(label)
	
	@property
	def label(self):
		if hasattr(self, '_label'):
			return self._label
		else:
			return None



class Histogram(Trace):
	def __init__(self, samples, color, marker=None, linestyle='solid', linewidth=None, alpha=1, label=None, density=False, bins='auto', showlegend=True, **kwargs):
		"""Given an array of samples produces a histogram.
		- color: RGB tuple.
		- marker: One of {'.','o','+','x','*', None}.
		- linestyle: One of {'solid','dotted','dashed', 'none', None}.
		- linewidth: Float number.
		- alpha: Float number.
		- label: String.
		- density: Same as homonym argument in numpy.histogram.
		- bins: Same as homonym argument in numpy.histogram."""
		super().__init__(label)
		self._color = validate_color(color)
		self._marker = validate_marker(marker)
		self._linestyle = validate_linestyle(linestyle)
		self._linewidth = validate_linewidth(linewidth)
		self._alpha = validate_alpha(alpha)
		self.showlegend = showlegend
		self.kwargs = kwargs
		if not hasattr(samples, '__iter__'):
			raise ValueError(f'<samples> must be iterable.')
		self._samples = samples
		# The following is for handling to whoever is going to plot this a collection of xy points to draw this as a scatter plot.
		samples = np.array(samples)
		hist, bin_edges = np.histogram(
			samples[(~np.isnan(samples))&(~np.isinf(samples))],
			bins = bins,
			density = density,
		)
		if density == False:
			hist[-1] -= sum(samples==bin_edges[-1])
		else:
			hist *= np.diff(bin_edges)*len(samples)
			hist[-1] -= sum(samples==bin_edges[-1])
			hist /= np.diff(bin_edges)*len(samples)
		x = [-float('inf')]
		if density == False:
			y = [sum(samples<bin_edges[0])]
		else:
			if sum(samples<bin_edges[0]) == 0:
				y = [0]
			else:
				y = [float('NaN')]
		for idx,count in enumerate(hist):
			x.append(bin_edges[idx])
			x.append(bin_edges[idx])
			y.append(y[-1])
			y.append(count)
		x.append(bin_edges[-1])
		y.append(y[-1])
		x.append(bin_edges[-1])
		if density == False:
			y.append(sum(samples>=bin_edges[-1]))
		else:
			if sum(samples>=bin_edges[-1]) == 0:
				y.append(0)
			else:
				y.append(float('NaN'))
		x.append(float('inf'))
		y.append(y[-1])
		self._x = np.array(x)
		self._y = np.array(y)
		
		self._bin_edges = bin_edges
		self._bin_counts = np.array([y[0]] + list(hist) + [y[-1]])
	
	@property
	def color(self):
		return self._color
	
	@property
	def marker(self):
		return self._marker
	
	@property
	def linestyle(self):
		return self._linestyle
	
	@property
	def linewidth(self):
		return self._linewidth
	
	@property
	def alpha(self):
		return self._alpha
	
	@property
	def x(self):
		return self._x
	
	@property
	def y(self):
		return self._y
	
	@property
	def bin_edges(self):
		return self._bin_edges
		
	@property
	def bin_counts(self):
		return self._bin_counts

		