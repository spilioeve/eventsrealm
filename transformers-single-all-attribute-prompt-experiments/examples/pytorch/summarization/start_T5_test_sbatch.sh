# for k in 1 2 3 5 10 20 41
for k in 15 25
do
    sbatch /gscratch/argon/artidoro/eva_transformers/examples/pytorch/summarization/run_T5_script_k5_test_sbatch.sh $k
done