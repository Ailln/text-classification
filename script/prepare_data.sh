#!/bin/bash

# 准备数据
echo -e "## Prepare data.\n"


if [ ! -d "../data/THUCNews-5_2000" ]; then
    if [ ! -f "../data/THUCNews-5_2000.zip" ];then
        echo ">> Download dataset..."
        wget -c https://cnlp.dovolopor.com/downloads/THUCNews-5_2000.zip -O ../data/THUCNews-5_2000.zip
        echo ">> Unzip dataset..."
        unzip ../data/THUCNews-5_2000.zip -d ../data/
        rm ../data/THUCNews-5_2000.zip
    else
        echo "!! file is exists."
    fi
else
    echo "!! data/THUCNews-5_2000 dir is exists."
fi

echo -e ">> Prepare data successful.\n"
