# 数据集
dataset_dic = {
    'RAF0':'--dataset RAF-DB --subset base',
    'RAFf':'--dataset RAF-DB --subset flip',
}

# 模型
model_cmd = {
    'base':"--model TCR_0 -n_b 2 -pt gap ",

    'TCR0a': "--model TCR_0 -n_b 2 -pt avg ",
    'TCR1a': "--model TCR_1 -n_b 2 -pt avg ",
    'TCR2a': "--model TCR_2 -n_b 2 -pt avg ",
    'TCR3a': "--model TCR_3 -n_b 2 -pt avg ",

    'TCR0v': "--model TCR_0 -n_b 2 -pt vit ",
    'TCR1v': "--model TCR_1 -n_b 2 -pt vit ",
    'TCR2v': "--model TCR_2 -n_b 2 -pt vit ",
    'TCR3v': "--model TCR_3 -n_b 2 -pt vit ",
} 

# encoder layer
encoder_cmd = {
    'e0': "-n_h 8 -dp 0.0", # -bi
}

# 预训练模型
pretrain_cmd = {
    'p0': "--pretrain  MS_Celeb_1M",
}

# 学习率 & batch size
lr_batch_cmd = {
    'l0': "-b 128 --lr 0.001 -lrr 0 -lrs 0 -lri 0",
}

# 基本参数
basic_cmd = {
    'b0': " --momentum 0.9 --wd 0",
}

# 环境参数
device_cmd = {
    'd0': "--epochs 150 -mi_b 64 -j 8",
}



# 模板dictionary
scheme_dict = {
    'Baseline':{
        '0': 'RAF0_base_e0_p0_l0_b0_d0', # 
        '1': 'RAFf_base_e0_p0_l0_b0_d0', # 
    },

    'TC_Res':{
            '0': 'RAF0_TCR2v_e0_p0_l0_b0_d0', #
            '1': 'RAFf_TCR2v_e0_p0_l0_b0_d0', #  
        },
}

# 命令集dictionary
command_dict = {**dataset_dic, **model_cmd, **encoder_cmd, **pretrain_cmd,
                **lr_batch_cmd, **basic_cmd, **device_cmd}

