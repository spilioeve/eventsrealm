ks=(1 2 3 5 10 20 41)
for i in 0 1 2 3 4 5 6
do
    bash /gscratch/argon/artidoro/eva_transformers/examples/pytorch/summarization/run_T5_script_k1_test.sh ${ks[$i]} i &
done