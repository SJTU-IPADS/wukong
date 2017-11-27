#!/bin/sh
#
# 1. List all nodes at the file 'mpd.host'
# 2. Share the input RDF data among all nodes through DFS (e.g., NFS or HDFS)
# 3. Configure wukong at the file 'config'
# 4. Edit the file 'core.bind' to control the thread binding.
#

#resort the args
ARGS=`getopt -o "c:" -l "command:" -n "run.sh" -- "$@"` 
eval set -- "${ARGS}"

# the number of servers
num_servers=1

# the command of one-shot execution on wukong
oneshot_cmd=   
while true; do
    case "${1}" in
        -c|--command)
        shift;
        if  [ -n "${1}" ] ; then
           	oneshot_cmd=${1}
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
    num_servers=$arg ;
done
 
# NOTE: the hostfile of mpiexec must match that of wukong (i.e., mpd.hosts)
if [ -z "$oneshot_cmd" ]; 
then
	${WUKONG_ROOT}/deps/openmpi-1.6.5-install/bin/mpiexec -x CLASSPATH -x LD_LIBRARY_PATH -hostfile mpd.hosts -n $num_servers ../build/wukong config mpd.hosts -b core.bind
else
	${WUKONG_ROOT}/deps/openmpi-1.6.5-install/bin/mpiexec -x CLASSPATH -x LD_LIBRARY_PATH -hostfile mpd.hosts -n $num_servers ../build/wukong config mpd.hosts -b core.bind -c "$oneshot_cmd"
fi 
