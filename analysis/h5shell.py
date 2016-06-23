"""
	Utilities for handling HDF5 files.

	Supplement functionalities for h5py.File class,\
	allowing to browse and manipulate groups, datasets\
	quickly with Unix-like functions.

	To initiate, use e.g. f = h5shell(filename, 'w')

	Common flags:
	-r : Recursive, act on the current group and its subgroups
	-p : Print return values to screen
"""
import h5py

class h5shell(h5py.File):
	""" Supplement functionalities for h5py.File class,\
	allowing to browse and manipulate groups, datasets\
	quickly with Unix-like functions.

	To initiate, use e.g. f = h5shell(filename, 'w')
	"""
	def __init__(self, filename, mode=None, driver=None,
                 libver=None, userblock_size=None, swmr=False, **kwds):
		super(h5shell, self).__init__(filename, mode=None, driver=None,
                 libver=None, userblock_size=None, swmr=False, **kwds)
		self._HEAD = self

	def pwd(self):
		""" Return the name of the current group """
		return self._HEAD.name

	def ls(self, *args):
		""" List the children of a group.

		Flags:
		-r : Recursive, list all its subgroups and datasets
		-p : Print return values to screen
		-t : If print is enabled, print as a tree

		Examples:
		f.ls('G1') --> list items in subgroup named 'G1' inside the current group
		f.ls(grp1, '-r') --> list items recursively in group object grp1
		f.ls('.. -pt') --> list items in the parent group and print a tree layout
		f.ls('/ -r') --> list all items in the file

		"""
		grp, flag = _filter_args(*args)
		if grp is None:
			grp = self._HEAD
		if isinstance(grp, str): grp = self._get_item(grp)

		ops = _options(flag)
		if ops['r']: # recursive
			subgrps = []
			grp.visit(lambda x: subgrps.append(x))
		else:
			subgrps = [key for key in grp.keys()]

		if ops['p']: # print out
			_display(subgrps, tree=ops['t'])
		return subgrps

	def cd(self, grp=None):
		""" Switch the working locaiton to another group.

		Return a group object.

		Examples:
		f.cd('G1') --> switch to subgroup named 'G1' inside the current group
		f.cd(grp1) --> switch to group object grp1
		f.cd('..') --> switch to the parent group
		f.cd() or f.cd('/') --> switch to the root group

		"""
		if grp is None: grp = self
		if isinstance(grp, str): grp = self._get_item(grp)
		self._HEAD = grp
		return grp

	def mkdir(self, grp_name):
		""" Create a new group.

		Return the new group object.

		Examples:
		f.mkdir('G12') --> create new subgroup 'G12' inside the current group
		f.mkdir('/G2') --> create new group 'G2' in the root group
		"""
		if grp_name[0]=='/':
			grp_cur = self
		else:
			grp_cur = self._HEAD
		return grp_cur.create_group(grp_name)

	def rm(self, grp):
		""" Remove a group or dataset.

		Examples:
		f.rm('G1') --> remove the subgroup 'G1' in the current group
		f.rm(grp1) --> remove the group object grp1
		"""
		grp = self._get_item(grp)
		del self[grp.name]

	def touch(self, dset_name, **kwargs):
		""" Create a new dataset. All keyword arguments will be passed to
		h5py.Group.create_dataset() function.
		"""
		mark = dset_name.rfind('/')
		if mark==-1:
			grp = self._HEAD
			dname = dset_name
		else:
			path = dset_name[:mark]
			grp = self._get_item(path)
			dname = dset_name[mark+1:]
		dset = grp.create_dataset(dname,**kwargs)
		return dset

	def grep(self, key, *args):
		""" Search for 'key' in group. Additional arguments will be passed\
		to the function h5shell.ls().

		Argument 'key' can be a string or boolean function. If 'key' is a string, will search\
		for group names which contain 'key'. If 'key' is a boolean function taking\
		a group/dataset object as the only argument, then will search for groups\
		upon which key() return True.

		Examples:
		f.grep('D','G1') --> search for 'D' in subgroup named 'G1' inside the current group
		f.grep('D',grp1, '-r') --> search for 'D' recursively in group object grp1
		f.grep('D','.. -pt') --> search for 'D' in the parent group and print a tree layout
		f.grep('D','/ -r') --> search for 'D' in all the items in the file
		f.grep(lambda x: 'D' in x.name) is the same as f.grep('D')
		"""
		grps = self.ls(*args)
		if isinstance(key,str):
			return [grp for grp in grps if key in grp]
		else:
			return [grp for grp in grps if key(self[grp])]

	def cp(self, src, des, **kwargs):
		""" Copy object(s)

		Examples:
		f.cp('G1','G2') --> copy 'G1' to new 'G2'
		f.cp('G1/','G2') --> copy all content of 'G1' into 'G2'
		f.cp('G1','G2/') --> copy G1 into 'G2'
		"""
		def copy_in(source, dest, **kwargs):
			""" Copy an object to a new object of same name in destination """
			des_name = dest.name + '/'
			srcs = source.name.split('/')
			des_name = des_name + srcs[-1]
			# Avoid overwrite existing object
			while des_name in self:
				des_name += '-copy'
			self.copy(source, des_name, **kwargs)

		if isinstance(src,str):
			src_obj = self._get_item(src)
		else:
			src_obj = src
		if isinstance(des,str):
			des_obj = self._get_item(des)
		else:
			des_obj = des

		if isinstance(src,str) and src[-1]=='/':
			# Copy all items in src to des
			for item in self.ls(src_obj):
				try:
					copy_in(src_obj[item], des_obj, **kwargs)
				except:
					print("Cannot copy {}. Ignored.".format(item))

		elif isinstance(des,str) and des[-1]!='/':
			if des[0]!='/':
				des = self._HEAD.name + '/' + des
			while des in self:
				des += '-copy'
			self.copy(src_obj, des, **kwargs)
		else:
			copy_in(src_obj, des_obj, **kwargs)

	def cat(self,*args):
		""" View details about an object
		"""
		target, flags = _filter_args(*args)
		if target is None:
			target = self._HEAD
		if isinstance(target,str):
			target = self._get_item(target)

		items = {}
		items['Type'] = _get_type(target)
		if items['Type']=='Group':
			groups = []
			dsets = []
			unknowns  =[]
			for item in target.values():
				if _get_type(item)=="Dataset":
					dsets.append(item.name)
				elif _get_type(item)=="Group":
					groups.append(item.name)
				else:
					unknowns.append([item.name,_get_type(item)])
			if len(groups)>0: items['Groups'] = groups
			if len(dsets)>0: items['Datasets'] = dsets
			if len(unknowns)>0: items['Unknowns'] = unknowns
		elif items['Type']=='Dataset':
			items['Shape'] = str(target.shape)
			items['Type'] = str(target.dtype)
		attrs = {}
		for k,v in target.attrs.items():
			attrs[k] = v
		items['Attributes'] = attrs
		#TODO: Add print functionality
		return items

	def _get_item(self, grp_name):
		""" Return an object from a string """
		if grp_name[0]=='/':
			grp_cur = self
			grp_name = grp_name[1:]
		else:
			grp_cur = self._HEAD

		path = grp_name.split('/')
		for seg in path:
			if seg=='.':
				pass # do nothing
			elif seg=='..':
				grp_cur = grp_cur.parent
			elif seg in grp_cur:
				grp_cur = grp_cur[seg]
		return grp_cur

	def __repr__(self):
		return '<h5shell>-' + super(h5shell,self).__repr__()


#=======================
#   Internal functions
#=======================

def _options(flg_str):
	""" Return a dictionary """
	flg_str = flg_str.lower()
	opts = {'r':False,	# Recursive
			'p':False,	# Print
			't':False,	# Tree
			'a':False	# All
			}
	for k in opts.keys():
		if flg_str.find(k) > -1: opts[k] = True
	return opts

def _filter_args(*args):
	""" Analyze the arguments """
	target = None
	flag = ""
	if len(args)>1:
		target = args[0]
		flag = flag.join([k for k in args[1:] if isinstance(k,str)])
	elif len(args)==1:
		if isinstance(args[0],str):
			flag = args[0]
			fs = flag.split()
			if fs[0][0].isalnum() or fs[0][0] in ['/','.']:
				target = fs[0]
				flag = "".join(fs[1:])
		else: target = args[0]
	return target, flag

def _display(items, tree=False, info=False):
	""" Display a list of items """
	if not tree:
		for i,item in enumerate(items):
			print("{}  {}".format(i,item))
	else:
		items = sorted(items)
		for i,item in enumerate(items):
			compos = item.split('/')
			disp = ''
			for j in range(len(compos)-1):
				disp = disp + '|   '
			disp = disp + '|__ ' + compos[-1]
			# if disp[1]=='|': disp = disp[1:]
			print("{}  {}".format(i,disp))

def _get_type(item):
	if type(item)==h5py._hl.dataset.Dataset:
		return "Dataset"
	elif type(item)==h5py._hl.group.Group:
		return "Group"
	else:
		return str(type(item))