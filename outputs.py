
import globals
import os
#-------------------------------------------------------------------
def openFile(filename):
	if os.path.exists(filename):
	    os.remove(filename)

	globals.file = open(filename, 'a')
#-------------------------------------------------------------------
def closeFile():
	# Close a file
	globals.file.close()