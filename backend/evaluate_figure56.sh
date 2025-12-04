#!/bin/bash

echo "**********************[LLaMA RoCo 2470]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t roco --n 2470 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA RoCo 4775]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t roco --n 4775 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA RoCo 1620]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t roco --n 1620 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA RoCo 5446]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t roco --n 5446 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA RoCo 1753]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t roco --n 1753 --baseline trinity
    sleep 1
done


echo "**********************[LLaMA KeyFormer 2430]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t keyformer --n 2430 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA KeyFormer 1799]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t keyformer --n 1799 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA KeyFormer 3323]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t keyformer --n 3323 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA KeyFormer 1813]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t keyformer --n 1813 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA KeyFormer 2682]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t keyformer --n 2682 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA KeyFormer 4358]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t keyformer --n 4358 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA KeyFormer 7039]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t keyformer --n 7039 --baseline trinity
    sleep 1
done

echo "**********************[LLaMA KeyFormer 4348]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 2 --m llama --t keyformer --n 4348 --baseline trinity
    sleep 1
done