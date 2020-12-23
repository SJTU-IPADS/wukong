import sys
import os
import getopt

def usage():
    print('usage:')
    print(' python convert_nt.py -i <input directory> -o <output directory>')

def convert(input_dir, output_dir):
    os.system('g++ generate_data.cpp -o generate_data -std=c++11')
    #Keep run generate_data project until it succeeds.
    os.system('./generate_data ' + input_dir + ' ' + output_dir)
    while (os.path.exists(output_dir + '/log_commit') == False):
        print('Failure occurred to generate_data project, try and recover ...')
        os.system('./generate_data ' + input_dir + ' ' + output_dir)

    os.system('rm ' + output_dir + '/log*')

def main():    
#get args
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["help", "input=", "output="])
    except getopt.GetoptError:  
        usage()
        sys.exit(2)

    input_dir = None
    output_dir = None
    for o, a in opts:
        if o in ("-h", "-help"):
            usage()
            sys.exit()
        elif o in ("-i", "--input"):
            input_dir = a
        elif o in ("-o", "--output"):
            output_dir = a
        else:
            assert False

    if input_dir == None or os.path.exists(input_dir) == False:
        usage()
        sys.exit()

    if os.path.exists(output_dir) == False:
        try:
            os.mkdir(output_dir)
        except:
            print('Failed to create output directory: ' + output_dir)
            sys.exit()

    convert(input_dir, output_dir)

    print('Convert from N-Triples to ID-Triples is done.\n')

if __name__ == "__main__":
    main()

