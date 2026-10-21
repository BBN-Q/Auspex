import pstats
import glob

s = pstats.Stats()

for fname in glob.glob('*.prof'):
	s.add(fname)
	# print(fname)

s.dump_stats("combined.prof")