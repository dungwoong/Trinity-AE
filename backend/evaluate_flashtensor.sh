#!/bin/bash

echo "**********************[LLaMA Vanilla]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m llama --t vanilla --n 946 --baseline flashtensor
    sleep 1
done
echo "**********************[LLaMA PreNorm]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m llama --t prenorm --n 579 --baseline flashtensor
    sleep 1
done
echo "**********************[LLaMA QKNorm]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m llama --t qknorm --n 2321 --baseline flashtensor
    sleep 1
done
echo "**********************[LLaMA KeyFormer]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m llama --t keyformer --n 2754 --baseline flashtensor
    sleep 1
done
echo "**********************[LLaMA RoCo]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m llama --t roco --n 1753 --baseline flashtensor
    sleep 1
done

echo "**********************[Falcon Vanilla]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m falcon --t vanilla --n 1562 --baseline flashtensor
    sleep 1
done
echo "**********************[Falcon PreNorm]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m falcon --t prenorm --n 579 --baseline flashtensor
    sleep 1
done
echo "**********************[Falcon QKNorm]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m falcon --t qknorm --n 3690 --baseline flashtensor
    sleep 1
done
echo "**********************[Falcon KeyFormer]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m falcon --t keyformer --n 3323 --baseline flashtensor
    sleep 1
done
echo "**********************[Falcon RoCo]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m falcon --t roco --n 1620 --baseline flashtensor
    sleep 1
done