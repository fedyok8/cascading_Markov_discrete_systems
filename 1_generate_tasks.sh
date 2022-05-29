# ./waf --run "mm1-queue --PrintHelp"
# Program Options:
#     --lambda:          arrival rate (packets/sec) [1]
#     --mu:              departure rate (packets/sec) [2]
#     --initialPackets:  initial packets in the queue [0]
#     --numPackets:      number of packets to enqueue [0]
#     --queueLimit:      size of queue (number of packets) [100]
#     --quiet:           whether to suppress all output [false]

if [ -f tasks.txt ]
then
    rm tasks.txt
fi

lambda=1
numPackets=1000000
for mu in 1.001 1.01 1.05 1.1 1.2 1.3 1.5 1.7 2 2.5 3 4 5 7 10
do
    for queueLimit in 1 2 3 4 5 7 10 15 20 30 50 100 200 500 1000 10000 100000
    do
        outfilename="mm1q_mu-${mu}_queueSize-${queueLimit}.dat"
        # echo $outfilename
        cmd="./waf --run-no-build \\\"mm1-queue --numPackets=${numPackets} --lambda=${lambda} --mu=${mu} --queueLimit=${queueLimit} --outfilename=${outfilename}\\\""
        # echo $cmd
        echo $cmd >> tasks.txt
        # ./waf --run-no-build "mm1-queue \
        #     --numPackets=100000 \
        #     --lambda=${lambda} \
        #     --mu=${mu} \
        #     --queueLimit=${queueLimit} \
        #     --outfilename=${outfilename}"
    done
done

# ./waf --run "mm1-queue \
#     --numPackets=1000 \
#     --queueLimit=3 \
#     --RngRun=2 \
#     --outfilename=test1"
