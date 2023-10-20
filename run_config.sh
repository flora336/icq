for task in snli
#task=copa
do
	log_dir=/home/shanshan/icq/prob/log/${task}
	out_dir=/home/shanshan/icq/prob/data/${task}
	in_dir=/home/shanshan/generate_noise/ESIM/data/dataset/${task^^}/
	for model_name in bert
	do
		#model_name=roberta

		model_file=/home/shanshan/generate_noise/transformers/output_dir/${task^^}/original_1_${model_name}_base/test_results_original.txt
		python test.py \
			--task ${task} \
			--log_dir ${log_dir} \
			--train_inf ${in_dir}/train/original_train.csv \
			--train_outf ${out_dir}/train.csv \
			--train_features_file ${log_dir}/train_features.pkl \
			--test_features_file ${log_dir}/test_features.pkl \
			--test_inf ${in_dir}/test/original_test.csv \
			--test_outf ${out_dir}/test.csv \
			--features sentiment overlap negation typo tense ner word pronoun \
			--model_res_file ${model_file} \
			--model_name ${model_name} \
			--change_type delete \
			--feature_split True
		wait
	done
	wait
done

			#--feature_split True
			#--lemmatize False
