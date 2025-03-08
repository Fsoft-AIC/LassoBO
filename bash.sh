#!/bin/bash
seed_start=2015
seed_end=2020

func_list=(hartmann6_300 levy15_300 ackley15_300)
max_samples=350
root_dir=all_logs

for func in ${func_list[@]}
do
    # lasso-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lassobo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # mcts-vs-turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python mcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --turbo_max_evals=5 \
            --ipt_solver=turbo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done



    # random search
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 random_search.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # vanilla bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 vanilia_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        } 
    done
    
    # dropout
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 dropout.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=10 \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

   

    # saasbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 saasbo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 turbo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # hesbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 ax_embedding_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=10 \
            --strategy=hesbo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # alebo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 ax_embedding_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=10 \
            --strategy=alebo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
done
