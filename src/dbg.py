'''
These are debugging wrapper functions for functions we don't want being called
when the debug command line option isn't called.
'''

import sys
import getopt

# Debug Constant
DEBUG = False

argv = sys.argv[1:]
try:
  opts, args = getopt.getopt(argv, "d",
              ["DEBUG"])
except:
  print("Error grabbing arguments")

for opt, arg in opts:
  if opt in ['-d', '--DEBUG']:
    DEBUG = True

def dbg_print(*args, **kwargs):
  if (DEBUG):
    print(" ".join(map(str,args)), **kwargs)

def dbg_assert(*c):
  if DEBUG:
    assert(c)