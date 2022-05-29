cat tasks.txt | xargs -l -P 12 -I CMD bash -c CMD
