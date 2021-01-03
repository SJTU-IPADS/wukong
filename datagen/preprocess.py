import sys
import os
import getopt

def usage():
    print('usage:')
    print(' python preprocess.py -i <input directory> -o <output directory> [-p <partitions>, 1024 as default]')

def is_commit(output_dir):
    log_file = output_dir + '/log_commit'
    if (os.path.exists(log_file) == False):
        return False;
    for line in open (log_file):
        if line.find('commit') != -1:
            return True;
    return False;

def convert(input_dir, output_dir, partitions):
    os.system('g++ preprocess.cpp -o preprocess -std=c++11 -I ./ -O2')
    #Keep run generate_data project until it succeeds.
    if (is_commit(output_dir) == False):
        os.system('./preprocess ' + input_dir + ' ' + output_dir + ' ' + partitions)
        while (is_commit(output_dir) == False):
            print('Failure occurred to generate_data project, try and recover ...')
            os.system('./preprocess ' + input_dir + ' ' + output_dir + ' ' + partitions)

def main():    
#get args
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["help", "input=", "output="])
    except getopt.GetoptError:  
        usage()
        sys.exit(2)

    input_dir = None
    output_dir = None
    partitions = 1024
    for o, a in opts:
        if o in ("-h", "-help"):
            usage()
            sys.exit()
        elif o in ("-i", "--input"):
            input_dir = a
        elif o in ("-o", "--output"):
            output_dir = a
        elif o in ("-p", "--partitions"):
            partitions = a
        else:
            assert False

    if partitions < 1 or partitions > 10 * 1024 * 1024 :
        print('Number of Partitions is too small or too large.')
        usage()
        sys.exit()

    if input_dir == None or os.path.exists(input_dir) == False or output_dir == None :
        usage()
        sys.exit()

    if os.path.exists(output_dir) == False:
        try:
            os.mkdir(output_dir)
        except:
            print('Failed to create output directory: ' + output_dir)
            sys.exit()

    convert(input_dir, output_dir, partitions)
    print('Preprocess is done.\n')
    os.system('rm ' + output_dir + '/log*')


if __name__ == "__main__":
    main()

