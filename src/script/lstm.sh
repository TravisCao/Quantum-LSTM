for i in {1..20}; do \
    python src/run_lstm.py --seed $i --data period1 --hidden_dim 3 & 
    python src/run_lstm.py --seed $i --data period2 --hidden_dim 3 &
    wait
done
