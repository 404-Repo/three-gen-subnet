Run judge with
```shell
vllm serve THUDM/GLM-4.1V-9B-Thinking --gpu-memory-utilization 0.9 --max-model-len 8096 --trust-remote-code --tensor-parallel-size 1 --port 4000 --api-key d5aed12f-b178-4f53-a956-7b3324dbe065
```