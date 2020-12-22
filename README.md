# embedded-system-2020-project

Term Project for Embedded Systems and Applications (SNU, 2020 Fall)

Quantize BERT for Question Answering (SQuAD 2.0 task)

All experiments were executed on `Ryzen 5600X w/ 32GB RAM, and Single RTX 3080`.

# How to run

First install packages

```sh
pip install -r requirements.txt -r requirements-dev.txt
```

## Train

### Full Precision Model

```sh
python run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad_v2 \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir <output_dir> \
    --per_device_eval_batch_size=4  \
    --per_device_train_batch_size=12   \
    --save_steps 10000
```

### Quantized Model

Before training, change bits of [QuantizationConfig](./quant_util.py). And then run below.

```sh
python train_quantized_bert.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad_v2 \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir <output_dir> \
    --per_device_eval_batch_size=4  \
    --per_device_train_batch_size=10   \
    --save_steps 10000
```

## Evaluation

### Full Precision Model

```sh
python run_qa.py \
    --model_name_or_path ./models/ \
    --dataset_name squad_v2 \
    --do_eval \
    --version_2_with_negative \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir <output_dir> \
    --per_device_eval_batch_size=256
```


### Quantized Model

Before evaluation, change bits of [QuantizationConfig](./quant_util.py), and set `"requantize_output": False`. And then run below.

```sh
python train_quantized_bert.py \
    --model_name_or_path ./qat_int4_models/ \
    --dataset_name squad_v2 \
    --do_eval \
    --version_2_with_negative \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir <output_dir> \
    --per_device_eval_batch_size=256
```
