checkpoint_number=10
pred_len=24
timestamp=`date +%Y%m%d%H%M%S`
logfile="Logs/${timestamp}.log"
rm Logs/*.log
#"energy" "stocks"  "mujoco"   "solar"
for dataset_name in  "stocks"   "mujoco" 
do 
#echo "Running prediction for dataset: $dataset_name"


python -m pdb  main.py --name ${dataset_name}  --config_file ./Config/${dataset_name}.yaml --gpu 1 --train  2>&1  | tee -a ${logfile}
 

#echo "Running prediction for dataset: $dataset_name"   2>&1 |tee -a  ${logfile}

#python  main.py --name ${dataset_name} --config_file ./Config/${dataset_name}.yaml \
#--gpu 1 --sample 1 --milestone $checkpoint_number --tensorboard  --mode predict --pred_len ${pred_len}  2>&1 |tee -a  ${logfile}
done 





