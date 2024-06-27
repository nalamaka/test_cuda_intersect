# wget http://snap.stanford.edu/data/cit-Patents.txt.gz
# gzip -d cit-Patents.txt.gz
# cp ../../Preprocess/fromDirectToUndirect .
# cp ../../Preprocess/preprocess .
# cp ../../Preprocess/partition .
# ./fromDirectToUndirect cit-Patents.txt
# ./fromDirectToUndirect as-skitter.txt
# ./fromDirectToUndirect ../../../Gpu-SubgraphIsomorphism/data/graph/Theory-3-4-5-9-16-25-B1k.txt
./fromDirectToUndirect ../../../Gpu-SubgraphIsomorphism/data/graph/Theory-5-9-16-25-81-B1k.txt
./preprocess