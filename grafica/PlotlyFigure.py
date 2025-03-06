from .figure import Figure
from .traces import Histogram
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import numpy as np
import warnings

class PlotlyFigure(Figure):
	def __init__(self, subplots=False, **kwargs):
		super().__init__()
		self.plotly_figure = go.Figure()
		if subplots:
			self.plotly_figure = make_subplots(rows=2, cols=2)
			# self.col = kwargs.get('col', 1)
			# self.row = kwargs.get('row', 1)
			# self.kwargs = kwargs
	# Methods that must be overridden ----------------------------------
	
	def show(self):
		# Overriding this method as specified in the class Figure.
		self.plotly_figure.show()
	
	def save(self, file_name=None, include_plotlyjs='cdn', auto_open=False, **kwargs):
		# Overriding this method as specified in the class Figure.
		if file_name is None:
			file_name = self.title
		if file_name is None: # If it is still None...
			raise ValueError(f'Please provide a name for saving the figure to a file by the <file_name> argument.')
		if file_name[-5:] != '.html':
			file_name += '.html'
		plotly.offline.plot(
			self.plotly_figure,
			filename = file_name,
			auto_open = auto_open,
			include_plotlyjs = include_plotlyjs,
			**kwargs
		)
	
	def draw_layout(self):
		# Overriding this method as specified in the class Figure.
		if self.show_title == True and self.title != None:
			self.plotly_figure.update_layout(title = self.title)
		self.plotly_figure.update_layout(
			xaxis_title = self.xlabel,
			yaxis_title = self.ylabel,
		)
		# Axes scale:
		if self.xscale in [None, 'lin']:
			pass
		elif self.xscale == 'log':
			self.plotly_figure.update_layout(xaxis_type = 'log')
		if self.yscale in [None, 'lin']:
			pass
		elif self.yscale == 'log':
			self.plotly_figure.update_layout(yaxis_type = 'log')
		
		if self.aspect == 'equal':
			self.plotly_figure.update_yaxes(
				scaleanchor = "x",
				scaleratio = 1,
			)
		
		if self.subtitle != None:
			self.plotly_figure.add_annotation(
				text = self.subtitle.replace('\n','<br>'),
				xref = "paper", 
				yref = "paper",
				x = .5, 
				y = 1,
				align = 'left',
				arrowcolor="#ffffff",
				font=dict(
					family="Courier New, monospace",
					color="#999999"
				),
			)
	
	def draw_trace(self, trace):
		# Overriding this method as specified in the class Figure.
		traces_drawing_methods = {
			Histogram: self._draw_histogram,
		}
		if type(trace) not in traces_drawing_methods:
			raise RuntimeError(f"Don't know how to draw a <{type(trace)}> trace...")
		traces_drawing_methods[type(trace)](trace)
	
	# Methods that draw each of the traces (for internal use only) -----
	
	
	def _draw_histogram(self, histogram, showlegend=True, **kwargs):
		print(kwargs)
		if not isinstance(histogram, Histogram):
			raise TypeError(f'<histogram> must be an instance of {Histogram}, received object of type {type(histogram)}.')
		x = np.array(histogram.x) # Make a copy to avoid touching the original data.
		x[0] = x[1] - (x[3]-x[1]) # Plotly does not plot points in infinity.
		x[-1] = x[-2] + (x[-2]-x[-4]) # Plotly does not plot points in infinity.
		legendgroup = str(np.random.rand(3))
		# The following trace is the histogram lines ---
  
		self.plotly_figure.add_trace(
			go.Scatter(
				x = x, 
				y = histogram.y,
				opacity = histogram.alpha,
				mode = 'lines',
				line = dict(
					dash = map_linestyle_to_Plotly_linestyle(histogram.linestyle),
				),
				legendgroup = legendgroup,
				showlegend = False,
				hoverinfo='skip',
			), **histogram.kwargs
		)
		self.plotly_figure['data'][-1]['marker']['color'] = rgb2hexastr_color(histogram.color)
		self.plotly_figure['data'][-1]['line']['width'] = histogram.linewidth
		# The following trace adds the markers in the middle of each bin ---
		if histogram.marker is not None:
			self.plotly_figure.add_trace(
				go.Scatter(
					x = [x[2*i] + (x[2*i+1]-x[2*i])/2 for i in range(int(len(x)/2))],
					y = histogram.y[::2],
					name = histogram.label,
					mode = 'markers',
					marker_symbol = map_marker_to_Plotly_markers(histogram.marker),
					opacity = histogram.alpha,
					line = dict(
						dash = map_linestyle_to_Plotly_linestyle(histogram.linestyle),
					),
					legendgroup = legendgroup,
					hoverinfo = 'skip',
					showlegend = False,
				), **histogram.kwargs
			)
			self.plotly_figure['data'][-1]['marker']['color'] = rgb2hexastr_color(histogram.color)
		# The following trace adds the hover texts ---
		self.plotly_figure.add_trace(
			go.Scatter(
				x = [x[2*i] + (x[2*i+1]-x[2*i])/2 for i in range(int(len(x)/2))],
				y = histogram.y[::2],
				name = histogram.label,
				mode = 'lines',
				marker_symbol = map_marker_to_Plotly_markers(histogram.marker),
				opacity = histogram.alpha,
				line = dict(
					dash = map_linestyle_to_Plotly_linestyle(histogram.linestyle),
				),
				legendgroup = legendgroup,
				showlegend = False,
				text = [f'Bin: (-∞, {histogram.bin_edges[0]})<br>Count: {histogram.bin_counts[0]}'] + [f'Bin: [{histogram.bin_edges[i]}, {histogram.bin_edges[i+1]})<br>Count: {histogram.bin_counts[i+1]}' for i in range(len(histogram.bin_edges)-1)] + [f'Bin: [{histogram.bin_edges[-1]},∞)<br>Count: {histogram.bin_counts[-1]}'],
				hovertemplate = "%{text}",
			), **histogram.kwargs
		)
		self.plotly_figure['data'][-1]['marker']['color'] = rgb2hexastr_color(histogram.color)
		self.plotly_figure['data'][-1]['line']['width'] = 0
		# The following trace is to add the item in the legend ---
		self.plotly_figure.add_trace(
			go.Scatter(
				x = [float('NaN')],
				y = [float('NaN')],
				name = histogram.label,
				mode = translate_marker_and_linestyle_to_Plotly_mode(histogram.marker, histogram.linestyle),
				marker_symbol = map_marker_to_Plotly_markers(histogram.marker),
				opacity = histogram.alpha,
				showlegend = histogram.showlegend,
				line = dict(
					dash = map_linestyle_to_Plotly_linestyle(histogram.linestyle),
				),
				legendgroup = legendgroup,
			), **histogram.kwargs
		)
		self.plotly_figure['data'][-1]['marker']['color'] = rgb2hexastr_color(histogram.color)
		self.plotly_figure['data'][-1]['line']['width'] = histogram.linewidth
	
		
def translate_marker_and_linestyle_to_Plotly_mode(marker, linestyle):
	"""<marker> and <linestyle> are each one and only one of the valid
	options for each object."""
	if marker is None and linestyle != 'none':
		mode = 'lines'
	elif marker is not None and linestyle != 'none':
		mode = 'lines+markers'
	elif marker is not None and linestyle == 'none':
		mode = 'markers'
	else:
		mode = 'lines'
	return mode

def map_marker_to_Plotly_markers(marker):
	markers_map = {
		'.': 'circle',
		'+': 'cross',
		'x': 'x',
		'o': 'circle-open',
		'*': 'star',
		None: None
	}
	return markers_map[marker]

def map_linestyle_to_Plotly_linestyle(linestyle):
	linestyle_map = {
		'solid': None,
		None: None,
		'none': None,
		'dashed': 'dash',
		'dotted':  'dot',
	}
	return linestyle_map[linestyle]

def rgb2hexastr_color(rgb_color: tuple):
	# Assuming that <rgb_color> is a (r,g,b) tuple.
	color_str = '#'
	for rgb in rgb_color:
		color_hex_code = hex(int(rgb*255))[2:]
		if len(color_hex_code) < 2:
			color_hex_code = f'0{color_hex_code}'
		color_str += color_hex_code
	return color_str
