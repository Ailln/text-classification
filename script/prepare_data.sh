#!/bin/bash

echo `date`
echo -e "\n## Prepare data.\n"


data_dir="./data/THUCNews-5_2000/"

if [ ! -d ${data_dir} ]; then
    if [ ! -f "./data/THUCNews-5_2000.zip" ];then
        echo ">> 1 Download dataset..."
        wget -c https://cnlp.dovolopor.com/downloads/THUCNews-5_2000.zip -O ./data/THUCNews-5_2000.zip
        echo ">> 2 Unzip dataset..."
        unzip ./data/THUCNews-5_2000.zip -d ./data/
        rm ./data/THUCNews-5_2000.zip
    else
        echo "!! file is exists."
    fi
else
    echo -e "!! ${data_dir} dir is exists."
fi

train_dir=${data_dir}"train/"
validate_dir=${data_dir}"validate/"
test_dir=${data_dir}"test/"
if [ ! -d ${train_dir} -a ! -d ${validate_dir} -a ! -d ${test_dir} ]; then
    echo -e "\n>> 3 split train validate and test data.\n"

    mkdir ${train_dir}
    mkdir ${validate_dir}
    mkdir ${test_dir}

    class_arr=(体育 房产 时政 游戏 股票)
    for class_name in ${class_arr[@]}; do
        echo ">> split "${class_name}"..."

        num=0
        tmp_arr=()
        for file in `ls ${data_dir}`; do
            if [ ${file:0:2} = ${class_name} ]; then
                tmp_arr[num]=${file}
                if [ ${num} -lt 1200 ]; then
                    mv ${data_dir}${file} ${train_dir}
                elif [ ${num} -ge 1200 -a ${num} -lt 1600 ]; then
                    mv ${data_dir}${file} ${validate_dir}
                else
                    mv ${data_dir}${file} ${test_dir}
                fi
                let num++
            fi
        done
    done
else
    echo -e "!! ${train_dir} dir is exists."
    echo -e "!! ${validate_dir} dir is exists."
    echo -e "!! ${test_dir} dir is exists."
fi

echo -e "\n## Successful."