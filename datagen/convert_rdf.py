import sys
import os
import getopt

def usage():
    print('usage:')
    print(' python convert_rdf.py [Options] -i <input directory> -o <output directory> -s <data size>')
    print('Options:')
    print('  -r/--remove    Remove input files and directory')
    
#get jena
def get_jena():
    jena = 'apache-jena/bin/riot'
    if os.path.exists(jena) == False:
        os.system('wget http://mirrors.tuna.tsinghua.edu.cn/apache/jena/binaries/apache-jena-3.13.1.tar.gz')
        os.system('mkdir ./apache-jena & tar zxvf apache-jena-3.13.1.tar.gz -C ./apache-jena --strip-components 1')
        os.system('rm apache-jena-3.13.1.tar.gz')
    return jena
 
def main():
#get args
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:s:r", ["help", "input=", "output=", "size=", "remove"])
    except getopt.GetoptError:  
        usage()
        sys.exit(2)
    input_dir = None
    output_dir = None
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
        elif o in ("-r", "--remove"):
            remove_input = True
        else:
            assert False 
    if input_dir == None or os.path.exists(input_dir) == False:
        usage()
        sys.exit()

    jena = get_jena()

    if os.path.exists(output_dir) == False:
        os.system('mkdir ' + output_dir)
    
    #transfer to NT format
    for i in range(0, int(size)):
        rdf_file = input_dir + '/University' + str(i) + '_*.owl'
        uni_file = output_dir + '/uni' + str(i) + '.nt'
        os.system(jena + ' -output=N-Triples ' + rdf_file + ' >> ' + uni_file)
        if remove_input:
            os.system('rm ' + rdf_file)
        print('transfer uni' + str(i))

    print('Convert from RDF data to NT format data is done.\n')

if __name__ == "__main__":
    main()

