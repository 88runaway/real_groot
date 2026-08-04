hf download nvidia/Cosmos-Reason2-2B \
  --local-dir /mnt/netdata/Team/Personal/chenyiyang/zjb/ckpts/cosmos_reason2_2b \
  --max-workers 12

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com

curl -I https://hf-mirror.com

hf download nvidia/Cosmos-Reason2-2B \
  --local-dir /mnt/netdata/Team/Personal/chenyiyang/zjb/ckpts/cosmos_reason2_2b \
  --max-workers 4

