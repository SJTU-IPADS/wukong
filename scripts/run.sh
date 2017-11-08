#!/bin/sh
#
# 1. List all nodes at the file 'mpd.host'
# 2. Share the input RDF data among all nodes through DFS (e.g., NFS or HDFS)
# 3. Configure wukong at the file 'config'
# 4. Edit the file 'core.bind' to control the thread binding.

# NOTE: the hostfile of mpiexec must match that of wukong (i.e., mpd.hosts)

#resort the args
ARGS=`getopt -o "c:" -l "command:" -n "run.sh" -- "$@"` 
eval set -- "${ARGS}"

#parser arg
server_num=1
direct_command=
while true; do
    case "${1}" in
        -c|--command)
        shift;
        if  [ -n "${1}" ] ; then
           	direct_command=${1}
            shift;
        fi
        ;;
        --)
        shift;
        break;
        ;;
    esac
done

for arg do 
    server_num=$arg ;
done
 
if [ -z "$direct_command" ]; 
then
	${WUKONG_ROOT}/deps/openmpi-1.6.5-install/bin/mpiexec -x CLASSPATH -x LD_LIBRARY_PATH -hostfile mpd.hosts -n $server_num ../build/wukong config mpd.hosts -b core.bind
else
	${WUKONG_ROOT}/deps/openmpi-1.6.5-install/bin/mpiexec -x CLASSPATH -x LD_LIBRARY_PATH -hostfile mpd.hosts -n $server_num ../build/wukong config mpd.hosts -b core.bind -c "$direct_command"
fi 
