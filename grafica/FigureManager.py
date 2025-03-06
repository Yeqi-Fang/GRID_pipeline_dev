from .figure import Figure
from .PlotlyFigure import PlotlyFigure
import __main__
from pathlib import Path

class FigureManager:
	def __init__(self):
		self.figures = [] # Figures of class "from .figure import Figure" are stored here.
		self._unsaved_figures = [] # Figures that were not yet saved by the `save_unsaved` method.
		self.plotters = {} # Plotters to create figures are stored here.
		BUILT_IN_PLOTTERS = {
			'plotly': PlotlyFigure
		}
		for name, plotter in BUILT_IN_PLOTTERS.items():
			self.add_plotter(plotter = plotter, name = name)
		self._default_plotter_name = 'plotly'
	
	@property
	def default_plotter(self):
		return self.plotters[self._default_plotter_name]
	@default_plotter.setter
	def default_plotter(self, plotter_name):
		if plotter_name not in self.plotters:
			raise ValueError(f'<plotter_name> must be one of {set(self.plotters.keys())}.')
		self._default_plotter_name = plotter_name
	
	def add_plotter(self, plotter, name):
		if not issubclass(plotter, Figure):
			raise TypeError(f'<plotter> must be a subclass of Figure. Received {plotter}.')
		if not isinstance(name, str):
			raise TypeError(f'<name> must be a string, received {name} of type {type(name)}.')
		self.plotters[name] = plotter
	
	def new(self, plotter_name=None, subplots=False, **kwargs):
		if plotter_name is None:
			plotter = self.default_plotter
		elif plotter_name not in self.plotters:
			raise ValueError(f'Unknown <plotter_name> "{plotter_name}".')
		else:
			plotter = self.plotters[plotter_name]
		fig = plotter(subplots, **kwargs)
		self.figures.append(fig)
		self._unsaved_figures.append(fig)
		fig.set(**kwargs)
		return fig
	
	def show(self):
		for fig in self.figures:
			fig.show()
	
	def save_unsaved(self, mkdir=False):
		if mkdir == False: # Save the plots in the current working directory.
			directory = './'
		else:
			if mkdir == True: # I create a directory automatically with the name of the script, in the same place as the script.
				directory = __main__.__file__.replace('.py', '') + '_plots/'
			else: # I assume mkdir is a path to a directory where to save the plots.
				directory = str(mkdir)
			Path(directory).mkdir(parents=True, exist_ok=True)
		for idx,fig in enumerate(self._unsaved_figures):
			file_name = fig.title if fig.title is not None else f'figure_{idx+1}'
			fig.save(file_name = str(Path(directory)/Path(file_name)))
		self._unsaved_figures = []
