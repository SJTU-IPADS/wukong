import sys
import os
import getopt

def usage():
    print('usage:')
    print(' python convert_rdf.py [Options] -i <input directory> -o <output directory> -s <data size> -p <input prefix> -w <output prefix>')
    print('Options:')
    print('  -r/--remove    Remove input files in input directory')
    
#get jena
def get_jena():
    jena = 'apache-jena/bin/riot'
    if os.path.exists(jena) == True:
        return jena

    source = 'apache-jena-3.17.0.tar.gz'
    try_cnt = 0
    while os.path.exists(source) == False and try_cnt < 5:
        os.system('wget http://archive.apache.org/dist/jena/binaries/' + source)
        try_cnt += 1

    if os.path.exists(source) == False:
        print('Failed to get apache jena, please check the source. Or download in current path manually before using convert tool.')
        sys.exit()

    os.system('mkdir ./apache-jena & tar zxvf ' + source + ' -C ./apache-jena --strip-components 1')
    #os.remove(source)
    return jena
 
def main():
#get args
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:s:p:w:r", ["help", "input=", "output=", "size=", "input_prefix=", "output_prefix=" "remove"])
    except getopt.GetoptError:  
        usage()
        sys.exit(2)
    input_dir = None
    output_dir = None
    input_prefix = None
    output_prefix = None
    size = 0
    remove_input = False
    for o, a in opts:
        if o in ("-h", "-help"):
            usage()
            sys.exit()
        elif o in ("-i", "--input"):
            input_dir = a
        elif o in ("-o", "--output"):
            output_dir = a
        elif o in ("-s", "--size"):
            size = a
        elif o in ("-p", "--input_prefix"):
            input_prefix = a
        elif o in ("-w", "--output_prefix"):
            output_prefix = a
        elif o in ("-r", "--remove"):
            remove_input = True
        else:
            assert False
    if size == 0:
        print('Size of dataset should not be 0.')
        usage()
        sys.exit()

    if input_prefix == None or output_prefix == None:
        usage()
        sys.exit()

    if input_dir == None or os.path.exists(input_dir) == False:
        usage()
        sys.exit()
    
    if os.path.exists(output_dir) == False:
        try:
            os.mkdir(output_dir)
        except:
            print('Failed to create output directory: ' + output_dir)
            sys.exit()

    jena = get_jena()
    
    #transfer to NT format
    for i in range(0, int(size)):
        rdf_file = input_dir + '/' + input_prefix + str(i) + '_*.owl'
        uni_file = output_dir + '/' + output_prefix + str(i) + '.nt'

        try:
            os.system(jena + ' --output=N-Triples ' + rdf_file + ' >> ' + uni_file)
            if remove_input:
                os.system('rm ' + rdf_file)
        except:
            print('Convert error occurred.')
            sys.exit()

        print('Transferred ' + rdf_file)

    print('Convert from RDF data to NT format data is done.\n')
    try:
        os.system('rm -rf apache-jena')
    except:
        pass

if __name__ == "__main__":
    main()

