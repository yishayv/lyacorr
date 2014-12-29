#!/usr/bin/python

import sys, getopt, os

#convert a spectra csv file with a 1-line header into a binary npy format.

def print_usage(program_name):
    print program_name, '-i <inputfile> -o <outputfile>'

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv[1:],"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print argv[0], '-i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print argv[0], '-i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    if (not inputfile) or (not outputfile):
        print_usage(argv[0])
        sys.exit(2)
#   print 'Input file is "' + inputfile + '"'
#   print 'Output file is "' + outputfile + '"'
    if (not os.path.isfile(inputfile)):
        print 'Input file not found'
        sys.exit(2)
    
    import numpy as np
    spectra = np.genfromtxt(inputfile, delimiter=',')
    np.save(outputfile, spectra)

if __name__ == "__main__":
   main(sys.argv)
   
