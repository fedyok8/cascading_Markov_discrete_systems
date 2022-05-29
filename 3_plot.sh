if [ -f tasks_plot.txt ]
then
    rm tasks_plot.txt
fi

for data in dat/*
do
    cmd="/home/fedyok8/anaconda3/envs/plt/bin/python process.py ${data}"
    echo $cmd >> tasks_plot.txt
done

cat tasks_plot.txt | xargs -l -P 12 -I CMD bash -c CMD
