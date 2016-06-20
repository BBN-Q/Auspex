"""
	Utilities for handling HDF5 files
"""
import h5py

class h5shell(h5py.File):
	def __init__(self, filename, mode=None, driver=None, 
                 libver=None, userblock_size=None, swmr=False, **kwds):
		super(h5shell, self).__init__(filename, mode=None, driver=None, 
                 libver=None, userblock_size=None, swmr=False, **kwds)
		self._HEAD = self

	def pwd(self):
		""" Return current group """
		return self._HEAD.name

	def ls(self, *args):
		""" List the child(ren) of a group """
		grp, flag = filter_args(*args)
		if grp is None:
			grp = self._HEAD
		if isinstance(grp, str): grp = self.get_group(grp)
		
		ops = options(flag)
		if ops['r']:
			subgrps = []
			grp.visit(lambda x: subgrps.append(x))
		else:
			subgrps = [key for key in grp.keys()]
		
		if ops['p']:
			display(subgrps, tree=ops['t'])
		return subgrps

	def cd(self, grp=None):
		""" Switch to another group """
		if grp is None: grp = self
		if isinstance(grp, str): grp = self.get_group(grp)
		self._HEAD = grp
		return grp

	def mkdir(self, grp_name):
		""" Create a new group """
		if grp_name[0]=='/':
			grp_cur = self
		else:
			grp_cur = self._HEAD
		return grp_cur.create_group(grp_name)

	def rm(self, grp):
		""" Remove a group or file """
		grp = self.get_group(grp)
		del self[grp.name]

	def touch(self, dset_name, **kwargs):
		""" Create a new dataset """
		mark = dset_name.rfind('/')
		if mark==-1:
			grp = self._HEAD
			dname = dset_name
		else:
			path = dset_name[:mark]
			grp = self.get_group(path)
			dname = dset_name[mark+1:]
		dset = grp.create_dataset(dname,**kwargs)
		return dset

	def get_group(self, grp_name):
		""" Return a group instance from a string """
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
			else:
				print("Group '{}' not found. Ignored.".format(seg))

		return grp_cur

	def __repr__(self):
		rep = '<h5shell>-' + super(h5shell,self).__repr__()
		return rep

#=======================
#   Internal functions
#=======================

def options(flg_str):
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

def filter_args(*args):
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

def display(items, tree=False, info=False):
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
