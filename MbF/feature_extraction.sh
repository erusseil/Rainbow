division=$1


for i in $(seq 0 $((division-1)))
do
   nohup python feature_extraction.py $division $i $2 $3 $4 > nohup/extract_$((i)).out 2>&1 &
done