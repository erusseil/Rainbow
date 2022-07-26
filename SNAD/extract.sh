division=10


for i in $(seq 0 $((division-1)))
do
   nohup python start_extraction.py $division $i > nohup/extract_$((i)).out 2>&1 &
   echo "Job $((i+1)) / $division started"
done