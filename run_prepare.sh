for task in roc
#task=copa
do
	log_dir=/home/shanshan/icq/prob/log/${task}
	out_dir=/home/shanshan/icq/prob/data/${task}
	in_dir=/home/shanshan/generate_noise/ESIM/data/dataset/${task^^}/
	change_type=substitute
	for model_name in bert
	do
		#model_name=roberta
		rm -rf ${log_dir}/
		model_file=/home/shanshan/generate_noise/transformers/output_dir/${task^^}/original_1_${model_name}_base/test_results_original.txt
		python ./prepare_data.py \
			--task ${task} \
			--log_dir ${log_dir} \
			--train_inf ${in_dir}/train/original_train.csv \
			--train_outf ${out_dir}/train.csv \
			--train_features_file ${log_dir}/train_features.pkl \
			--test_features_file ${log_dir}/test_features.pkl \
			--test_inf ${in_dir}/test/original_test.csv \
			--test_outf ${out_dir}/test.csv \
			--features sentiment overlap negation tense ner word pronoun \
			--cueness_types lmi pmi condition frequency\
			--change_type ${change_type}  \
			--merge_changed_file ${out_dir}/${change_type}.csv \
			--feature_split True
		wait
	done
	wait
done

			#--features sentiment overlap negation typo tense ner word pronoun \
			#--lemmatize False
			#--feature_split True
			#--features sentiment overlap negation typo tense ner word pronoun \
