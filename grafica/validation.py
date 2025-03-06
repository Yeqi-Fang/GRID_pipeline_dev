def validate_label(label):
	if label is None:
		return None
	if not isinstance(label, str):
		raise TypeError(f'<label> must be a string, received {label} of type {type(label)}.')
	return label

def validate_color(color):
	if color is None:
		raise ValueError(f'<color> cannot be None, it must have some value. The user should never see this error, so a color must be specified in the corresponding Figure method (e.g. Figure.scatter) if the user does not specifies any.')
	received_color = color
	try:
		color = tuple(color)
	except:
		raise TypeError(f'<color> must be an iterable of the form (r,g,b) where r,g and b are integer numbers from 0 to 255. Received {color}.')
	if len(color) != 3 or any({not 0<=i<=255 for i in color}):
		raise ValueError(f'<color> must contain 3 numbers ranging from 0 to 255, received {received_color}.')
	if sum(color) > 3: # This probably means that it was specified with rgb values in the range 0-255, which is an old pain. I convert this to values in 0-1.
		color = tuple([rgb/255 for rgb in color])
	return color

VALID_MARKERS = {'.','o','+','x','*', None}
def validate_marker(marker):
	if marker not in VALID_MARKERS:
		raise ValueError(f'<marker> must be one of {VALID_MARKERS}, received {marker}.')
	return marker

VALID_LINESTYLES = {'solid','dotted','dashed', 'none', None}
def validate_linestyle(linestyle):
	if linestyle not in VALID_LINESTYLES:
		raise ValueError(f'<linestyle> must be one of {VALID_LINESTYLES}, received {linestyle}.')
	return linestyle

def validate_linewidth(linewidth):
	if linewidth is None: # Use the defaultl value.
		return 2
	received_linewidth = linewidth
	try:
		linewidth = float(linewidth)
	except:
		raise TypeError(f'<linewidth> must be a float number, received {received_linewidth} of type {type(received_linewidth)}.')
	if linewidth < 0:
		raise ValueError(f'<linewidth> must be a positive number, received {linewidth}.')
	return linewidth

def validate_alpha(alpha):
	if alpha is None: # Use the default value
		return 1
	received_alpha = alpha
	try:
		alpha = float(alpha)
	except:
		raise TypeError(f'<alpha> must be a float number, received {received_alpha} of type {type(alpha)}.')
	if not 0 <= alpha <= 1:
		raise ValueError(f'<alpha> must be a positive number, received {alpha}.')
	return alpha

VALIDATION_FUNCTIONS_MAPPING = {
	'label': validate_label,
	'color': validate_color,
	'marker': validate_marker,
	'linestyle': validate_linestyle,
	'linewidth': validate_linewidth,
	'alpha': validate_alpha,
}
def validate_kwargs(kwargs2validate, kwargs):
	"""
	kwargs2validate: An iterable with the names of the kwargs that have 
		to be validated from <kwargs>.
	kwargs: A dictionary with the kwargs to validate.
	
	- If an argument is in <kwargs2validate> and it is also in <kwargs> then
	the corresponding validating function (according to the mapping below)
	is called to check whether it is correct. 
	- If an argument is in <kwargs2validate> and it is NOT in <kwargs>, a
	None value is passed to the corresponding validation function and it
	will determine whether this is fine or not.
	- Arguments in <kwargs> that are not in <kwargs2validate> will raise
	an error.
	"""
	for arg in kwargs:
		if arg not in kwargs2validate:
			raise ValueError(f'Wrong key word arguments <{arg}>.')
	for arg in kwargs2validate:
		kwargs[arg] = VALIDATION_FUNCTIONS_MAPPING[arg](kwargs.get(arg))
	return kwargs
