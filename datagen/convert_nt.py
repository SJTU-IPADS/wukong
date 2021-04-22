import sys
import os
import getopt

def usage():
    print('usage:')
    print(' python convert_nt.py -i <input directory> -o <output directory>')

def is_commit(output_dir):
    log_file = output_dir + '/log_commit'
    if (os.path.exists(log_file) == False):
        return False;
    for line in open (log_file):
        if line.find('Commit.') != -1:
            return True;
    return False;

def convert(input_dir, output_dir):
    os.system('g++ generate_data.cpp -o generate_data -std=c++11 -I ./ -O2')
    #Keep run generate_data project until it succeeds.
    if (is_commit(output_dir) == False):
        os.system('./generate_data ' + input_dir + ' ' + output_dir)
        while (is_commit(output_dir) == False):
            print('Failure occurred to generate_data project, try and recover ...')
            os.system('./generate_data ' + input_dir + ' ' + output_dir)

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

    if input_dir == None or os.path.exists(input_dir) == False or output_dir == None :
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

    os.system('rm ' + output_dir + '/log*')

if __name__ == "__main__":
    main()

